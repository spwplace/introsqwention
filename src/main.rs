mod api;
mod db;
mod distributed;
mod experiments;
mod prompts;
mod state;
mod steering;
mod worker;

use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use candle_core::{DType, Device};
use clap::Parser;
use tower_http::services::ServeDir;
use tracing_subscriber::EnvFilter;

use state::SharedState;

#[derive(Parser)]
#[command(name = "introspect", about = "Introspection server + dashboard")]
struct Args {
    /// HuggingFace model ID or local path to model directory
    #[arg(short, long)]
    model: String,

    /// Port to serve on
    #[arg(short, long, default_value = "3131")]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Data type for model weights (f16, bf16, f32)
    #[arg(long, default_value = "f16")]
    dtype: String,

    /// Path to static files directory
    #[arg(long, default_value = "static")]
    static_dir: PathBuf,

    /// Path to SQLite database for persistence
    #[arg(long, default_value = "data/introsqwention.db")]
    db: PathBuf,

    /// Tensor parallelism world size (number of GPUs).
    /// Only effective with --features nccl.
    #[arg(long, default_value = "1")]
    tp_size: usize,
}

fn parse_dtype(s: &str) -> Result<DType> {
    match s.to_lowercase().as_str() {
        "f16" | "float16" => Ok(DType::F16),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f32" | "float32" => Ok(DType::F32),
        other => anyhow::bail!("Unknown dtype: {}", other),
    }
}

fn select_device(rank: usize) -> Result<Device> {
    #[cfg(feature = "metal")]
    {
        let _ = rank;
        tracing::info!("Using Metal device");
        return Ok(Device::new_metal(0)?);
    }

    #[cfg(feature = "cuda")]
    {
        tracing::info!("Using CUDA device {rank}");
        return Ok(Device::new_cuda(rank)?);
    }

    #[allow(unreachable_code)]
    {
        let _ = rank;
        tracing::info!("Using CPU device");
        Ok(Device::Cpu)
    }
}

// ── Worker entry point (NCCL TP only) ───────────────────────────────

#[cfg(feature = "nccl")]
fn worker_main() -> Result<()> {
    // Workers reuse stderr tracing from the spawning env
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .with_writer(std::io::stderr)
        .init();

    let (tp, meta) = distributed::nccl_setup::init_worker()?;
    let args = Args::parse();
    let dtype = parse_dtype(&args.dtype)?;

    tracing::info!("Worker rank {} loading model: {}", tp.rank, args.model);
    let model = mistralrs_core::introspection::IntrospectionModel::load(
        &args.model,
        meta.device,
        dtype,
        tp.comm,
    )?;

    tracing::info!("Worker rank {} entering command loop", tp.rank);
    worker::worker_loop(model)
}

// ── Master entry point ──────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // If we're a TP worker, run the worker path (blocks forever)
    #[cfg(feature = "nccl")]
    if distributed::nccl_setup::is_worker() {
        return worker_main();
    }

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();
    let dtype = parse_dtype(&args.dtype)?;

    // Initialize tensor parallelism (if requested)
    let tp_config = if args.tp_size > 1 {
        #[cfg(not(feature = "nccl"))]
        anyhow::bail!("--tp-size > 1 requires building with --features nccl");

        #[cfg(feature = "nccl")]
        {
            let device = select_device(0)?;
            distributed::nccl_setup::init_master(args.tp_size, &device)?
        }
    } else {
        distributed::TpConfig::single()
    };

    let device = select_device(tp_config.rank)?;

    tracing::info!("Loading model: {}", args.model);
    let model = mistralrs_core::introspection::IntrospectionModel::load(
        &args.model,
        device,
        dtype,
        tp_config.comm,
    )?;

    let model_info = model.model_info();

    // Worker coordinator (Some when running TP with multiple ranks)
    let worker_coordinator = if args.tp_size > 1 {
        Some(worker::WorkerCoordinator::new(args.tp_size))
    } else {
        None
    };

    // Initialize database and load persisted data
    let db_path = args.db.clone();
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    db::init_db(&db_path).map_err(|e| anyhow::anyhow!("DB init failed: {}", e))?;

    let steering_vectors = db::load_steering_vectors(&db_path).unwrap_or_default();
    let experiments = db::load_experiments(&db_path).unwrap_or_default();
    if !steering_vectors.is_empty() {
        tracing::info!(
            "Loaded {} steering vectors from DB",
            steering_vectors.len()
        );
    }
    if !experiments.is_empty() {
        tracing::info!("Loaded {} experiments from DB", experiments.len());
    }

    let shared_state = Arc::new(SharedState {
        model: Mutex::new(model),
        model_info,
        experiments: RwLock::new(experiments),
        steering_vectors: RwLock::new(steering_vectors),
        db_path,
        worker_coordinator,
    });

    let static_dir = args.static_dir.clone();
    if static_dir.exists() {
        tracing::info!("Serving static files from: {}", static_dir.display());
    } else {
        tracing::warn!(
            "Static directory not found: {} — dashboard unavailable, API still works",
            static_dir.display()
        );
    }

    let app = Router::new()
        // GET endpoints
        .route("/api/model_info", get(api::api_model_info))
        .route("/api/experiments", get(api::api_experiments))
        .route("/api/experiments/{id}", get(api::api_experiment))
        .route("/api/steering_vectors", get(api::api_steering_vectors))
        // POST endpoints
        .route("/api/tokenize", post(api::api_tokenize))
        .route("/api/forward", post(api::api_forward))
        .route(
            "/api/set_steering_vector",
            post(api::api_set_steering_vector),
        )
        .route(
            "/api/clear_steering_vectors",
            post(api::api_clear_steering_vectors),
        )
        .route("/api/train", post(api::api_train))
        .route(
            "/api/apply_steering_vector",
            post(api::api_apply_steering_vector),
        )
        // Experiment runners
        .route("/api/run/logit_diff", post(api::api_run_logit_diff))
        .route(
            "/api/run/control_questions",
            post(api::api_run_control_questions),
        )
        .route(
            "/api/run/logit_lens_comparison",
            post(api::api_run_logit_lens_comparison),
        )
        .route("/api/run/top_of_mind", post(api::api_run_top_of_mind))
        .route(
            "/api/run/concept_activation",
            post(api::api_run_concept_activation),
        )
        .route(
            "/api/run/layer_type_lens",
            post(api::api_run_layer_type_lens),
        )
        .route(
            "/api/run/steering_survival",
            post(api::api_run_steering_survival),
        )
        .route("/api/run/cka", post(api::api_run_cka))
        .route(
            "/api/run/routing_analysis",
            post(api::api_run_routing_analysis),
        )
        .route("/api/run/causal_tracing", post(api::api_run_causal_tracing))
        .route(
            "/api/run/gdn_state_stats",
            post(api::api_run_gdn_state_stats),
        )
        // Code-native detection experiments
        .route(
            "/api/run/code_logit_diff",
            post(api::api_run_code_logit_diff),
        )
        .route(
            "/api/run/code_gen_detection",
            post(api::api_run_code_gen_detection),
        )
        .route(
            "/api/run/concept_identification",
            post(api::api_run_concept_identification),
        )
        .route(
            "/api/run/discrimination_matrix",
            post(api::api_run_discrimination_matrix),
        )
        .with_state(shared_state)
        .fallback_service(ServeDir::new(&static_dir).append_index_html_on_directories(true));

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Starting server at http://{}", addr);
    tracing::info!("  Dashboard: http://{}/", addr);
    tracing::info!("  API: http://{}/api/model_info", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

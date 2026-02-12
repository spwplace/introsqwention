mod api;
mod db;
mod experiments;
mod prompts;
mod state;
mod steering;

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
}

fn parse_dtype(s: &str) -> Result<DType> {
    match s.to_lowercase().as_str() {
        "f16" | "float16" => Ok(DType::F16),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f32" | "float32" => Ok(DType::F32),
        other => anyhow::bail!("Unknown dtype: {}", other),
    }
}

fn select_device() -> Result<Device> {
    #[cfg(feature = "metal")]
    {
        tracing::info!("Using Metal device");
        return Ok(Device::new_metal(0)?);
    }

    #[cfg(feature = "cuda")]
    {
        tracing::info!("Using CUDA device 0");
        return Ok(Device::new_cuda(0)?);
    }

    #[allow(unreachable_code)]
    {
        tracing::info!("Using CPU device");
        Ok(Device::Cpu)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();
    let dtype = parse_dtype(&args.dtype)?;
    let device = select_device()?;

    tracing::info!("Loading model: {}", args.model);
    let model =
        mistralrs_core::introspection::IntrospectionModel::load(&args.model, device, dtype)?;

    let model_info = model.model_info();

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

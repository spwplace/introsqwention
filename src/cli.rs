use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use serde_json::Value;

#[derive(Parser)]
#[command(
    name = "introspect-cli",
    about = "CLI client for the introspection server"
)]
struct Cli {
    /// Server URL
    #[arg(long, default_value = "http://127.0.0.1:3131")]
    server: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show model info
    Info,
    /// Tokenize text
    Tokenize {
        /// Text to tokenize
        text: String,
    },
    /// Forward pass with logit lens
    Forward {
        /// Input text
        text: String,
        /// Number of top tokens per layer
        #[arg(long, default_value = "5")]
        top_k: usize,
    },
    /// Train a steering vector for a concept
    Train {
        /// Concept name
        concept: String,
        /// Number of training suffixes
        #[arg(long)]
        suffixes: Option<usize>,
        /// Comma-separated layer indices
        #[arg(long, value_delimiter = ',')]
        layers: Option<Vec<usize>>,
    },
    /// List trained steering vectors
    Vectors,
    /// Apply a trained steering vector to the model
    Apply {
        /// Concept name
        concept: String,
        /// Scale factor
        #[arg(long, default_value = "8.0")]
        scale: f64,
        /// Comma-separated layer indices
        #[arg(long, value_delimiter = ',')]
        layers: Option<Vec<usize>>,
    },
    /// Clear all steering vectors from the model
    Clear,
    /// List experiments
    Experiments,
    /// Get experiment details
    Experiment {
        /// Experiment ID
        id: String,
    },
    /// Run an experiment
    Run {
        #[command(subcommand)]
        experiment: RunCommand,
    },
}

#[derive(Subcommand)]
enum RunCommand {
    /// Run logit diff experiment
    LogitDiff {
        concept: String,
        #[arg(long, default_value = "8.0")]
        scale: f64,
        #[arg(long, value_delimiter = ',')]
        layers: Option<Vec<usize>>,
        #[arg(long, default_value = "with_info")]
        variant: String,
    },
    /// Run control questions experiment
    Control {
        concept: String,
        #[arg(long, default_value = "8.0")]
        scale: f64,
        #[arg(long, value_delimiter = ',')]
        layers: Option<Vec<usize>>,
        #[arg(long, default_value = "with_info")]
        variant: String,
    },
    /// Run logit lens comparison
    LensCompare {
        concept: String,
        #[arg(long, default_value = "8.0")]
        scale: f64,
        #[arg(long, value_delimiter = ',')]
        layers: Option<Vec<usize>>,
        #[arg(long, value_delimiter = ',', default_value = "yes,no")]
        tracked_tokens: Vec<String>,
        #[arg(long, default_value = "with_info")]
        variant: String,
    },
    /// Run top-of-mind generation
    TopOfMind {
        concept: String,
        #[arg(long, default_value = "8.0")]
        scale: f64,
        #[arg(long, value_delimiter = ',')]
        layers: Option<Vec<usize>>,
    },
    /// Measure concept activation in text
    ConceptActivation {
        concept: String,
        text: String,
        #[arg(long, value_delimiter = ',')]
        layers: Option<Vec<usize>>,
    },
    /// Run layer type lens
    LayerTypeLens { text: String },
    /// Run steering survival experiment
    SteeringSurvival {
        concept: String,
        #[arg(long, value_delimiter = ',')]
        injection_layers: Vec<usize>,
        #[arg(long, default_value = "8.0")]
        scale: f64,
    },
    /// Run CKA analysis
    Cka,
    /// Run routing analysis
    RoutingAnalysis {
        text: String,
        #[arg(long)]
        concept: Option<String>,
        #[arg(long, default_value = "8.0")]
        scale: f64,
    },
    /// Run causal tracing
    CausalTracing {
        clean_text: String,
        corrupted_text: String,
    },
    /// Run GDN state statistics
    GdnStateStats { text: String },
    /// Run code-native logit diff (Phase 1)
    CodeLogitDiff {
        concept: String,
        #[arg(long, default_value = "8.0")]
        scale: f64,
        #[arg(long, value_delimiter = ',')]
        layers: Option<Vec<usize>>,
        #[arg(long, default_value = "10")]
        top_k: usize,
    },
    /// Run code-native generation detection (Phase 2)
    CodeGen {
        concept: String,
        #[arg(long, default_value = "8.0")]
        scale: f64,
        #[arg(long, value_delimiter = ',')]
        layers: Option<Vec<usize>>,
        #[arg(long, default_value = "32")]
        max_tokens: usize,
    },
    /// Run concept identification (Phase 3)
    ConceptId {
        /// Comma-separated concept names
        #[arg(value_delimiter = ',')]
        concepts: Vec<String>,
        #[arg(long, default_value = "8.0")]
        scale: f64,
        #[arg(long, value_delimiter = ',')]
        layers: Option<Vec<usize>>,
    },
    /// Run discrimination matrix (Phase 4)
    Discrimination {
        /// Comma-separated concept names
        #[arg(value_delimiter = ',')]
        concepts: Vec<String>,
        #[arg(long, default_value = "8.0")]
        scale: f64,
        #[arg(long, value_delimiter = ',')]
        layers: Option<Vec<usize>>,
    },
    /// Run code-native full pipeline: code-logit-diff + code-gen + concept-id
    CodeFull {
        concept: String,
        #[arg(long, default_value = "8.0")]
        scale: f64,
        #[arg(long, value_delimiter = ',')]
        layers: Option<Vec<usize>>,
    },
    /// Run the full vgel pipeline: train + logit-diff + control + lens-compare + top-of-mind
    Full {
        concept: String,
        #[arg(long, default_value = "8.0")]
        scale: f64,
        #[arg(long)]
        suffixes: Option<usize>,
        #[arg(long, value_delimiter = ',')]
        layers: Option<Vec<usize>>,
    },
}

fn main() {
    let cli = Cli::parse();
    let client = match Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()
    {
        Ok(client) => client,
        Err(e) => {
            eprintln!("Error: failed to create HTTP client: {}", e);
            std::process::exit(1);
        }
    };

    let result = match cli.command {
        Command::Info => cmd_info(&client, &cli.server),
        Command::Tokenize { text } => cmd_tokenize(&client, &cli.server, &text),
        Command::Forward { text, top_k } => cmd_forward(&client, &cli.server, &text, top_k),
        Command::Train {
            concept,
            suffixes,
            layers,
        } => cmd_train(&client, &cli.server, &concept, suffixes, layers),
        Command::Vectors => cmd_vectors(&client, &cli.server),
        Command::Apply {
            concept,
            scale,
            layers,
        } => cmd_apply(&client, &cli.server, &concept, scale, layers),
        Command::Clear => cmd_clear(&client, &cli.server),
        Command::Experiments => cmd_experiments(&client, &cli.server),
        Command::Experiment { id } => cmd_experiment(&client, &cli.server, &id),
        Command::Run { experiment } => cmd_run(&client, &cli.server, experiment),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn cmd_info(client: &Client, server: &str) -> anyhow::Result<()> {
    let resp: Value = client
        .get(format!("{}/api/model_info", server))
        .send()?
        .json()?;
    println!("{}", serde_json::to_string_pretty(&resp)?);
    Ok(())
}

fn cmd_tokenize(client: &Client, server: &str, text: &str) -> anyhow::Result<()> {
    let resp: Value = client
        .post(format!("{}/api/tokenize", server))
        .json(&serde_json::json!({"text": text}))
        .send()?
        .json()?;
    let count = resp["count"].as_u64().unwrap_or(0);
    println!("{} tokens", count);
    if let Some(ids) = resp["token_ids"].as_array() {
        let tokens = resp["tokens"].as_array();
        for (i, id) in ids.iter().enumerate() {
            let tok = tokens
                .and_then(|t| t.get(i))
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            println!("  {:>4}: {:>6} = {:?}", i, id, tok);
        }
    }
    Ok(())
}

fn cmd_forward(client: &Client, server: &str, text: &str, top_k: usize) -> anyhow::Result<()> {
    let resp: Value = client
        .post(format!("{}/api/forward", server))
        .json(&serde_json::json!({"text": text, "top_k": top_k}))
        .send()?
        .json()?;
    if let Some(layers) = resp["layers"].as_array() {
        for layer in layers {
            let idx = layer["layer_idx"].as_u64().unwrap_or(0);
            let ltype = layer["layer_type"].as_str().unwrap_or("?");
            print!("L{:>2} ({:>14}):", idx, ltype);
            if let Some(tops) = layer["top_tokens"].as_array() {
                for tp in tops.iter().take(5) {
                    let tok = tp["token"].as_str().unwrap_or("?");
                    let prob = tp["probability"].as_f64().unwrap_or(0.0);
                    print!("  {:.4} {:?}", prob, tok);
                }
            }
            println!();
        }
    }
    if let Some(id) = resp["experiment_id"].as_str() {
        println!("\nExperiment saved: {}", id);
    }
    Ok(())
}

fn cmd_train(
    client: &Client,
    server: &str,
    concept: &str,
    suffixes: Option<usize>,
    layers: Option<Vec<usize>>,
) -> anyhow::Result<()> {
    let mut body = serde_json::json!({"concept": concept});
    if let Some(n) = suffixes {
        body["num_suffixes"] = serde_json::json!(n);
    }
    if let Some(l) = &layers {
        body["layers"] = serde_json::json!(l);
    }

    let resp = client
        .post(format!("{}/api/train", server))
        .json(&body)
        .header("Accept", "text/event-stream")
        .send()?;

    // Start with a spinner — progress bar appears once first event arrives
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    spinner.set_message(format!(
        "Training '{}' — waiting for first pair to complete...",
        concept
    ));
    spinner.enable_steady_tick(std::time::Duration::from_millis(120));

    let bar_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} pairs ({eta})")
        .unwrap()
        .progress_chars("=>-");

    let mut using_bar = false;
    let mut buf = String::new();
    let reader = std::io::BufRead::lines(std::io::BufReader::new(resp));
    for line in reader {
        let line = line?;
        if line.starts_with("event: ") {
            buf = line[7..].to_string();
        } else if line.starts_with("data: ") {
            let data = &line[6..];
            match buf.as_str() {
                "progress" => {
                    if let Ok(v) = serde_json::from_str::<Value>(data) {
                        let done = v["done"].as_u64().unwrap_or(0);
                        let total = v["total"].as_u64().unwrap_or(0);
                        if !using_bar {
                            spinner.finish_and_clear();
                            spinner.set_length(total);
                            spinner.set_style(bar_style.clone());
                            using_bar = true;
                        }
                        spinner.set_length(total);
                        spinner.set_position(done);
                    }
                }
                "complete" => {
                    spinner.finish_and_clear();
                    if let Ok(v) = serde_json::from_str::<Value>(data) {
                        println!(
                            "Trained '{}': {} pairs, {} layers",
                            v["concept"].as_str().unwrap_or("?"),
                            v["num_pairs"].as_u64().unwrap_or(0),
                            v["num_layers"].as_u64().unwrap_or(0),
                        );
                    }
                }
                "error" => {
                    spinner.finish_and_clear();
                    if let Ok(v) = serde_json::from_str::<Value>(data) {
                        anyhow::bail!("{}", v["error"].as_str().unwrap_or("unknown error"));
                    }
                }
                _ => {}
            }
            buf.clear();
        }
    }
    Ok(())
}

fn cmd_vectors(client: &Client, server: &str) -> anyhow::Result<()> {
    let resp: Value = client
        .get(format!("{}/api/steering_vectors", server))
        .send()?
        .json()?;
    if let Some(arr) = resp.as_array() {
        if arr.is_empty() {
            println!("No steering vectors trained yet.");
        } else {
            println!("{} steering vectors:", arr.len());
            for v in arr {
                println!(
                    "  '{}': concept='{}', {} training pairs",
                    v["name"].as_str().unwrap_or("?"),
                    v["concept"].as_str().unwrap_or("?"),
                    v["num_training_pairs"].as_u64().unwrap_or(0),
                );
            }
        }
    }
    Ok(())
}

fn cmd_apply(
    client: &Client,
    server: &str,
    concept: &str,
    scale: f64,
    layers: Option<Vec<usize>>,
) -> anyhow::Result<()> {
    let mut body = serde_json::json!({"name": concept, "scale": scale});
    if let Some(l) = &layers {
        body["layers"] = serde_json::json!(l);
    }
    let resp: Value = client
        .post(format!("{}/api/apply_steering_vector", server))
        .json(&body)
        .send()?
        .json()?;
    println!(
        "Applied '{}' at scale {} on layers {:?}",
        concept,
        resp["scale"].as_f64().unwrap_or(scale),
        resp["layers"]
    );
    Ok(())
}

fn cmd_clear(client: &Client, server: &str) -> anyhow::Result<()> {
    let _: Value = client
        .post(format!("{}/api/clear_steering_vectors", server))
        .json(&serde_json::json!({}))
        .send()?
        .json()?;
    println!("All steering vectors cleared.");
    Ok(())
}

fn cmd_experiments(client: &Client, server: &str) -> anyhow::Result<()> {
    let resp: Value = client
        .get(format!("{}/api/experiments", server))
        .send()?
        .json()?;
    if let Some(arr) = resp.as_array() {
        if arr.is_empty() {
            println!("No experiments yet.");
        } else {
            println!("{} experiments:", arr.len());
            for exp in arr {
                println!(
                    "  {} [{}] {}",
                    exp["id"].as_str().unwrap_or("?"),
                    exp["status"].as_str().unwrap_or("?"),
                    exp["name"].as_str().unwrap_or("?"),
                );
            }
        }
    }
    Ok(())
}

fn cmd_experiment(client: &Client, server: &str, id: &str) -> anyhow::Result<()> {
    let resp: Value = client
        .get(format!("{}/api/experiments/{}", server, id))
        .send()?
        .json()?;
    println!("{}", serde_json::to_string_pretty(&resp)?);
    Ok(())
}

fn post_json(client: &Client, url: &str, body: &Value) -> anyhow::Result<Value> {
    let resp = client.post(url).json(body).send()?;
    let status = resp.status();
    let body: Value = resp.json()?;
    if !status.is_success() {
        let err = body["error"].as_str().unwrap_or("request failed");
        anyhow::bail!("{} (HTTP {})", err, status);
    }
    Ok(body)
}

fn print_experiment_id(resp: &Value) {
    if let Some(id) = resp["experiment_id"].as_str() {
        println!("Experiment saved: {}", id);
    }
}

fn cmd_run(client: &Client, server: &str, experiment: RunCommand) -> anyhow::Result<()> {
    match experiment {
        RunCommand::LogitDiff {
            concept,
            scale,
            layers,
            variant,
        } => {
            println!("Running logit diff for '{}'...", concept);
            let mut body = serde_json::json!({
                "concept": concept,
                "scale": scale,
                "user_turn1_variant": variant,
            });
            if let Some(l) = layers {
                body["layers"] = serde_json::json!(l);
            }
            let resp = post_json(client, &format!("{}/api/run/logit_diff", server), &body)?;
            let r = &resp["result"];
            println!(
                "P(yes): base {:.4}% | steered {:.4}% | random {:.4}%",
                r["base_p_yes"].as_f64().unwrap_or(0.0) * 100.0,
                r["steered_p_yes"].as_f64().unwrap_or(0.0) * 100.0,
                r["random_control_p_yes"].as_f64().unwrap_or(0.0) * 100.0,
            );
            println!(
                "Yes shift: {:+.4}% (95% CI [{:+.4}%, {:+.4}%])",
                r["mean_yes_shift"].as_f64().unwrap_or(0.0) * 100.0,
                r["yes_shift_ci95_low"].as_f64().unwrap_or(0.0) * 100.0,
                r["yes_shift_ci95_high"].as_f64().unwrap_or(0.0) * 100.0,
            );
            print_experiment_id(&resp);
        }
        RunCommand::Control {
            concept,
            scale,
            layers,
            variant,
        } => {
            println!("Running control questions for '{}'...", concept);
            let mut body = serde_json::json!({
                "concept": concept,
                "scale": scale,
                "user_turn1_variant": variant,
            });
            if let Some(l) = layers {
                body["layers"] = serde_json::json!(l);
            }
            let resp = post_json(
                client,
                &format!("{}/api/run/control_questions", server),
                &body,
            )?;
            let r = &resp["result"]["summary"];
            println!(
                "Accuracy: base {:.1}% -> steered {:.1}% ({:+.1}%)",
                r["base_accuracy"].as_f64().unwrap_or(0.0) * 100.0,
                r["steered_accuracy"].as_f64().unwrap_or(0.0) * 100.0,
                r["accuracy_shift"].as_f64().unwrap_or(0.0) * 100.0,
            );
            println!(
                "Yes shift: {:+.3}% (std {:.3}%)",
                r["mean_yes_shift"].as_f64().unwrap_or(0.0) * 100.0,
                r["std_yes_shift"].as_f64().unwrap_or(0.0) * 100.0,
            );
            print_experiment_id(&resp);
        }
        RunCommand::LensCompare {
            concept,
            scale,
            layers,
            tracked_tokens,
            variant,
        } => {
            println!("Running logit lens comparison for '{}'...", concept);
            let mut body = serde_json::json!({
                "concept": concept,
                "scale": scale,
                "tracked_tokens": tracked_tokens,
                "user_turn1_variant": variant,
            });
            if let Some(l) = layers {
                body["layers"] = serde_json::json!(l);
            }
            let resp = post_json(
                client,
                &format!("{}/api/run/logit_lens_comparison", server),
                &body,
            )?;
            println!("{}", serde_json::to_string_pretty(&resp["result"])?);
            print_experiment_id(&resp);
        }
        RunCommand::TopOfMind {
            concept,
            scale,
            layers,
        } => {
            println!("Running top-of-mind for '{}'...", concept);
            let mut body = serde_json::json!({"concept": concept, "scale": scale});
            if let Some(l) = layers {
                body["layers"] = serde_json::json!(l);
            }
            let resp = post_json(client, &format!("{}/api/run/top_of_mind", server), &body)?;
            let r = &resp["result"];
            println!("{}", r["generated_text"].as_str().unwrap_or(""));
            println!(
                "({} tokens, {})",
                r["num_tokens"].as_u64().unwrap_or(0),
                r["stop_reason"].as_str().unwrap_or("?"),
            );
            print_experiment_id(&resp);
        }
        RunCommand::ConceptActivation {
            concept,
            text,
            layers,
        } => {
            let mut body = serde_json::json!({"concept": concept, "text": text});
            if let Some(l) = layers {
                body["layers"] = serde_json::json!(l);
            }
            let resp = post_json(
                client,
                &format!("{}/api/run/concept_activation", server),
                &body,
            )?;
            let r = &resp["result"];
            println!(
                "Mean: {:.4}, Max: {:.4} (layer {})",
                r["mean_activation"].as_f64().unwrap_or(0.0),
                r["max_activation"].as_f64().unwrap_or(0.0),
                r["max_layer"].as_u64().unwrap_or(0),
            );
            print_experiment_id(&resp);
        }
        RunCommand::LayerTypeLens { text } => {
            let body = serde_json::json!({"text": text});
            let resp = post_json(
                client,
                &format!("{}/api/run/layer_type_lens", server),
                &body,
            )?;
            let r = &resp["result"];
            println!(
                "Mean GDN confidence: {:.4}, Mean Attn confidence: {:.4}",
                r["mean_gdn_confidence"].as_f64().unwrap_or(0.0),
                r["mean_attn_confidence"].as_f64().unwrap_or(0.0),
            );
            if let Some(layers) = r["layers"].as_array() {
                for entry in layers {
                    println!(
                        "  L{} ({:15}) -> {:.3}% \"{}\"",
                        entry["layer_idx"].as_u64().unwrap_or(0),
                        entry["layer_type"].as_str().unwrap_or("?"),
                        entry["top1_prob"].as_f64().unwrap_or(0.0) * 100.0,
                        entry["top1_token"].as_str().unwrap_or("?"),
                    );
                }
            }
            print_experiment_id(&resp);
        }
        RunCommand::SteeringSurvival {
            concept,
            injection_layers,
            scale,
        } => {
            let body = serde_json::json!({
                "concept": concept,
                "injection_layers": injection_layers,
                "scale": scale,
            });
            let resp = post_json(
                client,
                &format!("{}/api/run/steering_survival", server),
                &body,
            )?;
            println!("{}", serde_json::to_string_pretty(&resp["result"])?);
            print_experiment_id(&resp);
        }
        RunCommand::Cka => {
            println!("Running CKA analysis...");
            let body = serde_json::json!({});
            let resp = post_json(client, &format!("{}/api/run/cka", server), &body)?;
            let r = &resp["result"];
            println!(
                "{} layers",
                r["layer_labels"].as_array().map(|a| a.len()).unwrap_or(0)
            );
            if let Some(cka) = r["consecutive_cka"].as_array() {
                println!("Consecutive CKA:");
                for (i, v) in cka.iter().enumerate() {
                    println!("  {} -> {}: {:.4}", i, i + 1, v.as_f64().unwrap_or(0.0));
                }
            }
            print_experiment_id(&resp);
        }
        RunCommand::RoutingAnalysis {
            text,
            concept,
            scale,
        } => {
            let mut body = serde_json::json!({"text": text, "scale": scale});
            if let Some(c) = concept {
                body["concept"] = serde_json::json!(c);
            }
            let resp = post_json(
                client,
                &format!("{}/api/run/routing_analysis", server),
                &body,
            )?;
            println!("{}", serde_json::to_string_pretty(&resp["result"])?);
            print_experiment_id(&resp);
        }
        RunCommand::CausalTracing {
            clean_text,
            corrupted_text,
        } => {
            let body = serde_json::json!({
                "clean_text": clean_text,
                "corrupted_text": corrupted_text,
            });
            let resp = post_json(client, &format!("{}/api/run/causal_tracing", server), &body)?;
            let r = &resp["result"];
            println!(
                "Clean: \"{}\" ({:.4}), Corrupted: \"{}\" ({:.4})",
                r["clean_top_token"].as_str().unwrap_or("?"),
                r["clean_prob"].as_f64().unwrap_or(0.0),
                r["corrupted_top_token"].as_str().unwrap_or("?"),
                r["corrupted_prob"].as_f64().unwrap_or(0.0),
            );
            if let Some(layers) = r["layer_recovery"].as_array() {
                for lr in layers {
                    let recovery = lr["recovery"].as_f64().unwrap_or(0.0);
                    let bar_len = (recovery.clamp(0.0, 1.0) * 20.0) as usize;
                    let bar = format!("{}{}", "#".repeat(bar_len), "-".repeat(20 - bar_len));
                    println!(
                        "  L{:2} ({:15}) {} {:.1}%",
                        lr["layer_idx"].as_u64().unwrap_or(0),
                        lr["layer_type"].as_str().unwrap_or("?"),
                        bar,
                        recovery * 100.0
                    );
                }
            }
            print_experiment_id(&resp);
        }
        RunCommand::GdnStateStats { text } => {
            let body = serde_json::json!({"text": text});
            let resp = post_json(
                client,
                &format!("{}/api/run/gdn_state_stats", server),
                &body,
            )?;
            let r = &resp["result"];
            if let Some(layers) = r["layers"].as_array() {
                println!("Layer  ||S||_F    eff.rank  sigma_max  H_spec");
                for s in layers {
                    println!(
                        "  L{:2}  {:8.2}  {:6.2}    {:8.2}  {:.3}",
                        s["layer_idx"].as_u64().unwrap_or(0),
                        s["frobenius_norm"].as_f64().unwrap_or(0.0),
                        s["effective_rank"].as_f64().unwrap_or(0.0),
                        s["top_singular_value"].as_f64().unwrap_or(0.0),
                        s["spectral_entropy"].as_f64().unwrap_or(0.0),
                    );
                }
            }
            print_experiment_id(&resp);
        }
        RunCommand::CodeLogitDiff {
            concept,
            scale,
            layers,
            top_k,
        } => {
            println!("Running code-native logit diff for '{}'...", concept);
            let mut body = serde_json::json!({
                "concept": concept,
                "scale": scale,
                "top_k": top_k,
            });
            if let Some(l) = layers {
                body["layers"] = serde_json::json!(l);
            }
            let resp = post_json(
                client,
                &format!("{}/api/run/code_logit_diff", server),
                &body,
            )?;
            let r = &resp["result"];
            println!(
                "Coverage (P(True)+P(False)): base {:.4}% | steered {:.4}% | random {:.4}%",
                r["mean_base_coverage"].as_f64().unwrap_or(0.0) * 100.0,
                r["mean_steered_coverage"].as_f64().unwrap_or(0.0) * 100.0,
                r["mean_random_coverage"].as_f64().unwrap_or(0.0) * 100.0,
            );
            println!(
                "P(True): base {:.4}% | steered {:.4}% | random {:.4}%",
                r["base_p_true"].as_f64().unwrap_or(0.0) * 100.0,
                r["steered_p_true"].as_f64().unwrap_or(0.0) * 100.0,
                r["random_p_true"].as_f64().unwrap_or(0.0) * 100.0,
            );
            println!(
                "True shift: {:+.4}% (std {:.4}%)",
                r["mean_true_shift"].as_f64().unwrap_or(0.0) * 100.0,
                r["std_true_shift"].as_f64().unwrap_or(0.0) * 100.0,
            );
            if let Some(trials) = r["trials"].as_array() {
                println!("\nPer-template breakdown:");
                for t in trials {
                    println!(
                        "  {:20} ({:15}) coverage: base={:.3}% steered={:.3}%",
                        t["template"].as_str().unwrap_or("?"),
                        t["variant"].as_str().unwrap_or("?"),
                        t["base_coverage"].as_f64().unwrap_or(0.0) * 100.0,
                        t["steered_coverage"].as_f64().unwrap_or(0.0) * 100.0,
                    );
                }
            }
            print_experiment_id(&resp);
        }
        RunCommand::CodeGen {
            concept,
            scale,
            layers,
            max_tokens,
        } => {
            println!("Running code-native generation for '{}'...", concept);
            let mut body = serde_json::json!({
                "concept": concept,
                "scale": scale,
                "max_tokens": max_tokens,
            });
            if let Some(l) = layers {
                body["layers"] = serde_json::json!(l);
            }
            let resp = post_json(
                client,
                &format!("{}/api/run/code_gen_detection", server),
                &body,
            )?;
            let r = &resp["result"];
            if let Some(conditions) = r["conditions"].as_array() {
                for cond in conditions {
                    println!(
                        "  {:20} {:8} temp={:.1} -> detection rate: {:.1}%",
                        cond["template"].as_str().unwrap_or("?"),
                        cond["condition"].as_str().unwrap_or("?"),
                        cond["temperature"].as_f64().unwrap_or(0.0),
                        cond["detection_rate"].as_f64().unwrap_or(0.0) * 100.0,
                    );
                    if let Some(gens) = cond["generations"].as_array() {
                        for g in gens {
                            let text = g["generated_text"]
                                .as_str()
                                .unwrap_or("")
                                .chars()
                                .take(60)
                                .collect::<String>();
                            println!(
                                "    -> {:?} (parsed: {:?})",
                                text,
                                g["parsed_result"],
                            );
                        }
                    }
                }
            }
            print_experiment_id(&resp);
        }
        RunCommand::ConceptId {
            concepts,
            scale,
            layers,
        } => {
            println!("Running concept identification for {:?}...", concepts);
            let mut body = serde_json::json!({
                "concepts": concepts,
                "scale": scale,
            });
            if let Some(l) = layers {
                body["layers"] = serde_json::json!(l);
            }
            let resp = post_json(
                client,
                &format!("{}/api/run/concept_identification", server),
                &body,
            )?;
            let r = &resp["result"];
            if let Some(ids) = r["identifications"].as_array() {
                for id in ids {
                    println!(
                        "  injected: {:12} -> parsed: {:12} ({})",
                        id["injected_concept"].as_str().unwrap_or("?"),
                        id["parsed_concept"].as_str().unwrap_or("?"),
                        id["match_type"].as_str().unwrap_or("?"),
                    );
                }
            }
            print_experiment_id(&resp);
        }
        RunCommand::Discrimination {
            concepts,
            scale,
            layers,
        } => {
            println!("Running discrimination matrix for {:?}...", concepts);
            let mut body = serde_json::json!({
                "concepts": concepts,
                "scale": scale,
            });
            if let Some(l) = layers {
                body["layers"] = serde_json::json!(l);
            }
            let resp = post_json(
                client,
                &format!("{}/api/run/discrimination_matrix", server),
                &body,
            )?;
            let r = &resp["result"];
            println!(
                "Overall accuracy: {:.1}%",
                r["overall_accuracy"].as_f64().unwrap_or(0.0) * 100.0,
            );
            if let Some(acc) = r["per_concept_accuracy"].as_array() {
                for (i, a) in acc.iter().enumerate() {
                    let concept = concepts.get(i).map(|s| s.as_str()).unwrap_or("?");
                    println!(
                        "  {:12}: {:.1}%",
                        concept,
                        a.as_f64().unwrap_or(0.0) * 100.0,
                    );
                }
            }
            print_experiment_id(&resp);
        }
        RunCommand::CodeFull {
            concept,
            scale,
            layers,
        } => {
            println!("=== Code-native full pipeline for '{}' ===\n", concept);

            // Phase 1: Code logit diff
            println!("Phase 1/3: Code logit diff...");
            let mut body = serde_json::json!({
                "concept": concept,
                "scale": scale,
            });
            if let Some(ref l) = layers {
                body["layers"] = serde_json::json!(l);
            }
            let resp = post_json(
                client,
                &format!("{}/api/run/code_logit_diff", server),
                &body,
            )?;
            let r = &resp["result"];
            println!(
                "  Coverage: base {:.3}% -> steered {:.3}%",
                r["mean_base_coverage"].as_f64().unwrap_or(0.0) * 100.0,
                r["mean_steered_coverage"].as_f64().unwrap_or(0.0) * 100.0,
            );
            println!(
                "  P(True): base {:.3}% -> steered {:.3}% (shift {:+.3}%)",
                r["base_p_true"].as_f64().unwrap_or(0.0) * 100.0,
                r["steered_p_true"].as_f64().unwrap_or(0.0) * 100.0,
                r["mean_true_shift"].as_f64().unwrap_or(0.0) * 100.0,
            );
            println!();

            // Phase 2: Code gen detection
            println!("Phase 2/3: Code generation detection...");
            let mut body = serde_json::json!({
                "concept": concept,
                "scale": scale,
            });
            if let Some(ref l) = layers {
                body["layers"] = serde_json::json!(l);
            }
            let resp = post_json(
                client,
                &format!("{}/api/run/code_gen_detection", server),
                &body,
            )?;
            let r = &resp["result"];
            if let Some(conditions) = r["conditions"].as_array() {
                for cond in conditions {
                    if cond["condition"].as_str() == Some("steered") {
                        println!(
                            "  {} (temp={:.1}): detection rate {:.1}%",
                            cond["template"].as_str().unwrap_or("?"),
                            cond["temperature"].as_f64().unwrap_or(0.0),
                            cond["detection_rate"].as_f64().unwrap_or(0.0) * 100.0,
                        );
                    }
                }
            }
            println!();

            // Phase 3: Concept identification
            println!("Phase 3/3: Concept identification...");
            let mut body = serde_json::json!({
                "concepts": [concept],
                "scale": scale,
            });
            if let Some(ref l) = layers {
                body["layers"] = serde_json::json!(l);
            }
            let resp = post_json(
                client,
                &format!("{}/api/run/concept_identification", server),
                &body,
            )?;
            let r = &resp["result"];
            if let Some(ids) = r["identifications"].as_array() {
                for id in ids {
                    println!(
                        "  Injected: {:12} -> Parsed: {:12} ({})",
                        id["injected_concept"].as_str().unwrap_or("?"),
                        id["parsed_concept"].as_str().unwrap_or("?"),
                        id["match_type"].as_str().unwrap_or("?"),
                    );
                }
            }
            println!();

            println!("=== Code-native pipeline complete for '{}' ===", concept);
        }
        RunCommand::Full {
            concept,
            scale,
            suffixes,
            layers,
        } => {
            // Resolve default steering layers from model info if not specified
            let layers = match layers {
                Some(l) => Some(l),
                None => {
                    let info: Value = client
                        .get(format!("{}/api/model_info", server))
                        .send()?
                        .json()?;
                    let num_layers = info["num_layers"].as_u64().unwrap_or(64) as usize;
                    let start = num_layers / 3;
                    let end = 2 * num_layers / 3;
                    Some((start..end).map(|i| i + 1).collect())
                }
            };

            println!("=== Full pipeline for '{}' ===\n", concept);

            // 1. Train (only on default steering layers, not all layers)
            println!("Step 1/5: Training steering vector...");
            cmd_train(client, server, &concept, suffixes, layers.clone())?;
            println!();

            // 2. Logit diff
            println!("Step 2/5: Logit diff...");
            let mut body = serde_json::json!({
                "concept": concept,
                "scale": scale,
                "user_turn1_variant": "with_info",
            });
            if let Some(ref l) = layers {
                body["layers"] = serde_json::json!(l);
            }
            let resp = post_json(client, &format!("{}/api/run/logit_diff", server), &body)?;
            let r = &resp["result"];
            println!(
                "  P(yes): base {:.2}% -> steered {:.2}% (shift {:+.2}%)",
                r["base_p_yes"].as_f64().unwrap_or(0.0) * 100.0,
                r["steered_p_yes"].as_f64().unwrap_or(0.0) * 100.0,
                r["mean_yes_shift"].as_f64().unwrap_or(0.0) * 100.0,
            );
            println!();

            // 3. Control questions
            println!("Step 3/5: Control questions...");
            let mut body = serde_json::json!({
                "concept": concept,
                "scale": scale,
                "user_turn1_variant": "with_info",
            });
            if let Some(ref l) = layers {
                body["layers"] = serde_json::json!(l);
            }
            let resp = post_json(
                client,
                &format!("{}/api/run/control_questions", server),
                &body,
            )?;
            let s = &resp["result"]["summary"];
            println!(
                "  Accuracy: {:.1}% -> {:.1}% ({:+.1}%)",
                s["base_accuracy"].as_f64().unwrap_or(0.0) * 100.0,
                s["steered_accuracy"].as_f64().unwrap_or(0.0) * 100.0,
                s["accuracy_shift"].as_f64().unwrap_or(0.0) * 100.0,
            );
            println!();

            // 4. Logit lens comparison
            println!("Step 4/5: Logit lens comparison...");
            let mut body = serde_json::json!({
                "concept": concept,
                "scale": scale,
                "tracked_tokens": ["yes", "no"],
                "user_turn1_variant": "with_info",
            });
            if let Some(ref l) = layers {
                body["layers"] = serde_json::json!(l);
            }
            let _resp = post_json(
                client,
                &format!("{}/api/run/logit_lens_comparison", server),
                &body,
            )?;
            println!("  Done (see dashboard for layer-by-layer details)");
            println!();

            // 5. Top of mind
            println!("Step 5/5: Top of mind...");
            let mut body = serde_json::json!({"concept": concept, "scale": scale});
            if let Some(ref l) = layers {
                body["layers"] = serde_json::json!(l);
            }
            let resp = post_json(client, &format!("{}/api/run/top_of_mind", server), &body)?;
            let r = &resp["result"];
            println!("  {}", r["generated_text"].as_str().unwrap_or(""));
            println!();

            println!("=== Pipeline complete for '{}' ===", concept);
        }
    }
    Ok(())
}

use std::collections::HashSet;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    response::{
        sse::{Event, KeepAlive, Sse},
        Json,
    },
};
use candle_core::IndexOp;
use tokio::sync::mpsc;

use crate::db;
use crate::experiments;
use crate::state::*;
use crate::steering;

type AppState = State<Arc<SharedState>>;
type ApiError = (axum::http::StatusCode, Json<serde_json::Value>);

fn err500(e: impl std::fmt::Display) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": e.to_string()})),
    )
}

fn err400(e: impl std::fmt::Display) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    (
        axum::http::StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": e.to_string()})),
    )
}

fn err404(e: impl std::fmt::Display) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    (
        axum::http::StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error": e.to_string()})),
    )
}

fn sse_json_event(event_name: &str, payload: serde_json::Value) -> Event {
    let data = serde_json::to_string(&payload)
        .unwrap_or_else(|_| "{\"error\":\"serialization\"}".to_string());
    Event::default().event(event_name).data(data)
}

fn truncate_for_label(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

// ── GET endpoints (kept from original) ──────────────────────────────

pub async fn api_experiments(State(state): AppState) -> Result<Json<serde_json::Value>, ApiError> {
    let experiments = state.experiments.read().map_err(|e| err500(e))?;
    let payload =
        serde_json::to_value(&experiments.values().collect::<Vec<_>>()).map_err(|e| err500(e))?;
    Ok(Json(payload))
}

pub async fn api_experiment(
    State(state): AppState,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let experiments = state.experiments.read().map_err(|e| err500(e))?;
    match experiments.get(&id) {
        Some(exp) => {
            let payload = serde_json::to_value(exp).map_err(|e| err500(e))?;
            Ok(Json(payload))
        }
        None => Err(err404("experiment not found")),
    }
}

pub async fn api_model_info(State(state): AppState) -> Result<Json<serde_json::Value>, ApiError> {
    let payload = serde_json::to_value(&state.model_info).map_err(|e| err500(e))?;
    Ok(Json(payload))
}

pub async fn api_steering_vectors(
    State(state): AppState,
) -> Result<Json<serde_json::Value>, ApiError> {
    let vecs = state.steering_vectors.read().map_err(|e| err500(e))?;
    let payload =
        serde_json::to_value(&vecs.values().collect::<Vec<_>>()).map_err(|e| err500(e))?;
    Ok(Json(payload))
}

// ── POST /api/tokenize ──────────────────────────────────────────────

pub async fn api_tokenize(
    State(state): AppState,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let model = state.model.lock().map_err(|e| err500(e))?;
        let (ids, tokens) = model.tokenize(&req.text).map_err(|e| err500(e))?;
        Ok(Json(serde_json::json!({
            "count": ids.len(),
            "token_ids": ids,
            "tokens": tokens,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/forward ───────────────────────────────────────────────

pub async fn api_forward(
    State(state): AppState,
    Json(req): Json<ForwardRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let top_k = req.top_k.unwrap_or(10);
        let model = state.model.lock().map_err(|e| err500(e))?;

        let (token_ids, tokens) = model.tokenize(&req.text).map_err(|e| err500(e))?;
        let result = model.forward_introspect(&req.text).map_err(|e| err500(e))?;
        let lens = model
            .logit_lens(&result.hidden_states)
            .map_err(|e| err500(e))?;

        let num_layers = lens.layer_probs.len();
        let mut layers_data = Vec::with_capacity(num_layers);

        for (layer_idx, probs_tensor) in lens.layer_probs.iter().enumerate() {
            let layer_type = if layer_idx == 0 {
                "embedding"
            } else if layer_idx % state.model_info.full_attention_interval == 0 {
                "full_attention"
            } else {
                "gdn"
            };

            let probs_vec: Vec<f32> = probs_tensor
                .squeeze(0)
                .map_err(|e| err500(e))?
                .to_vec1()
                .map_err(|e| err500(e))?;

            let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top_tokens: Vec<TokenProb> = indexed
                .iter()
                .take(top_k)
                .map(|&(id, prob)| TokenProb {
                    token: model.decode_token(id as u32),
                    token_id: id as u32,
                    probability: prob,
                })
                .collect();

            layers_data.push(LayerData {
                layer_idx,
                layer_type: layer_type.to_string(),
                top1_prob: indexed.first().map(|x| x.1).unwrap_or(0.0),
                top_tokens,
            });
        }

        // Save experiment
        let exp_id = uuid::Uuid::new_v4().to_string();
        let experiment = Experiment {
            id: exp_id.clone(),
            name: format!("logit_lens: {}", truncate_for_label(&req.text, 50)),
            status: "completed".to_string(),
            created_at: now_timestamp(),
            config: ExperimentConfig {
                experiment_type: "logit_lens".to_string(),
                prompt: req.text,
                steering_layers: None,
                steering_scale: None,
            },
            results: Some(ExperimentResults {
                logit_lens: Some(LogitLensData {
                    tokens: tokens.clone(),
                    token_ids: token_ids.clone(),
                    layers: layers_data.clone(),
                }),
                ..empty_results()
            }),
        };

        {
            let mut experiments = state.experiments.write().map_err(|e| err500(e))?;
            db::save_experiment(&state.db_path, &experiment).ok();
            experiments.insert(exp_id.clone(), experiment);
        }

        Ok(Json(serde_json::json!({
            "experiment_id": exp_id,
            "tokens": tokens,
            "token_ids": token_ids,
            "layers": layers_data,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/set_steering_vector ───────────────────────────────────

pub async fn api_set_steering_vector(
    State(state): AppState,
    Json(req): Json<SetSteeringVectorRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let model = state.model.lock().map_err(|e| err500(e))?;
        let device = model.device();
        let hidden_size = state.model_info.hidden_size;

        if req.vector.len() != hidden_size {
            return Err(err400(format!(
                "vector length {} != hidden_size {}",
                req.vector.len(),
                hidden_size
            )));
        }

        let vector =
            candle_core::Tensor::new(req.vector.as_slice(), device).map_err(|e| err500(e))?;
        let scaled = (&vector * req.scale).map_err(|e| err500(e))?;

        for &layer in &req.layers {
            if layer >= state.model_info.num_layers {
                return Err(err400(format!(
                    "layer {} out of range (model has {} layers)",
                    layer, state.model_info.num_layers
                )));
            }
            model.set_steering_vector(layer, scaled.clone());
        }

        Ok(Json(serde_json::json!({
            "status": "ok",
            "layers": req.layers,
            "scale": req.scale,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/clear_steering_vectors ────────────────────────────────

pub async fn api_clear_steering_vectors(
    State(state): AppState,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let model = state.model.lock().map_err(|e| err500(e))?;
        model.clear_steering_vectors();
        Ok(Json(serde_json::json!({"status": "ok"})))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/train (SSE streaming) ─────────────────────────────────

pub async fn api_train(
    State(state): AppState,
    Json(req): Json<TrainRequest>,
) -> Sse<impl futures_core::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, mut rx) = mpsc::channel::<Event>(64);
    let state = state.clone();

    tokio::task::spawn_blocking(move || {
        let model = match state.model.lock() {
            Ok(model) => model,
            Err(e) => {
                let _ = tx.blocking_send(sse_json_event(
                    "error",
                    serde_json::json!({"error": e.to_string()}),
                ));
                return;
            }
        };
        let target_layers = req.layers.as_deref();

        let tx_progress = tx.clone();
        let progress_cb: Option<steering::ProgressCallback> = Some(Box::new(move |done, total| {
            let _ = tx_progress.blocking_send(sse_json_event(
                "progress",
                serde_json::json!({"done": done, "total": total}),
            ));
        }));

        let result = steering::train_concept_vector(
            &model,
            &req.concept,
            req.num_suffixes,
            target_layers,
            progress_cb,
        );

        match result {
            Ok(trained) => {
                let num_layers = trained.vectors.len();
                let num_pairs = trained.num_pairs;

                let svec = SteeringVectorSet {
                    name: req.concept.clone(),
                    concept: req.concept.clone(),
                    vectors: trained.vectors,
                    num_training_pairs: num_pairs,
                    created_at: now_timestamp(),
                };

                {
                    let mut vecs = match state.steering_vectors.write() {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = tx.blocking_send(sse_json_event(
                                "error",
                                serde_json::json!({"error": e.to_string()}),
                            ));
                            return;
                        }
                    };
                    // Persist to SQLite
                    if let Err(e) = db::save_steering_vector(&state.db_path, &svec) {
                        tracing::warn!("Failed to persist steering vector: {}", e);
                    }
                    vecs.insert(req.concept.clone(), svec);
                }

                let _ = tx.blocking_send(sse_json_event(
                    "complete",
                    serde_json::json!({
                        "concept": req.concept,
                        "num_pairs": num_pairs,
                        "num_layers": num_layers,
                    }),
                ));
            }
            Err(e) => {
                let _ = tx.blocking_send(sse_json_event(
                    "error",
                    serde_json::json!({"error": e.to_string()}),
                ));
            }
        }
    });

    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            yield Ok(event);
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

// ── POST /api/apply_steering_vector ─────────────────────────────────

pub async fn api_apply_steering_vector(
    State(state): AppState,
    Json(req): Json<ApplySteeringVectorRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let scale = req.scale.unwrap_or(8.0);
        let num_layers = state.model_info.num_layers;
        let layers = req
            .layers
            .unwrap_or_else(|| default_steering_layers(num_layers));

        let svec_vectors = {
            let vecs = state.steering_vectors.read().map_err(|e| err500(e))?;
            let svec = vecs
                .get(&req.name)
                .ok_or_else(|| err404(format!("steering vector '{}' not found", req.name)))?;
            svec.vectors.clone()
        };

        let model = state.model.lock().map_err(|e| err500(e))?;
        steering::apply_steering_vectors(&model, &svec_vectors, &layers, scale)
            .map_err(|e| err500(e))?;

        Ok(Json(serde_json::json!({
            "status": "ok",
            "name": req.name,
            "scale": scale,
            "layers": layers,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/run/logit_diff ────────────────────────────────────────

pub async fn api_run_logit_diff(
    State(state): AppState,
    Json(req): Json<RunLogitDiffRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let scale = req.scale.unwrap_or(8.0);
        let variant = req.user_turn1_variant.as_deref().unwrap_or("with_info");
        let num_layers = state.model_info.num_layers;
        let layers = req
            .layers
            .unwrap_or_else(|| default_steering_layers(num_layers));
        let top_k = req.top_k.unwrap_or(10);

        let result =
            experiments::run_logit_diff(&state, &req.concept, scale, variant, &layers, top_k)
                .map_err(|e| err500(e))?;

        let exp_id = save_experiment(
            &state,
            format!("logit_diff: {} ({})", req.concept, variant),
            "logit_diff",
            &format!("concept={}, variant={}", req.concept, variant),
            Some(layers),
            Some(scale),
            ExperimentResults {
                logit_diff: Some(result.clone()),
                ..empty_results()
            },
        );

        Ok(Json(serde_json::json!({
            "experiment_id": exp_id,
            "result": result,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/run/control_questions ─────────────────────────────────

pub async fn api_run_control_questions(
    State(state): AppState,
    Json(req): Json<RunControlQuestionsRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let scale = req.scale.unwrap_or(8.0);
        let variant = req.user_turn1_variant.as_deref().unwrap_or("with_info");
        let num_layers = state.model_info.num_layers;
        let layers = req
            .layers
            .unwrap_or_else(|| default_steering_layers(num_layers));

        let result =
            experiments::run_control_questions(&state, &req.concept, scale, variant, &layers)
                .map_err(|e| err500(e))?;

        let exp_id = save_experiment(
            &state,
            format!("control_questions: {}", req.concept),
            "control_questions",
            &format!("concept={}, variant={}", req.concept, variant),
            Some(layers),
            Some(scale),
            ExperimentResults {
                control_questions: Some(result.clone()),
                ..empty_results()
            },
        );

        Ok(Json(serde_json::json!({
            "experiment_id": exp_id,
            "result": result,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/run/logit_lens_comparison ─────────────────────────────

pub async fn api_run_logit_lens_comparison(
    State(state): AppState,
    Json(req): Json<RunLogitLensComparisonRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let scale = req.scale.unwrap_or(8.0);
        let variant = req.user_turn1_variant.as_deref().unwrap_or("with_info");
        let num_layers = state.model_info.num_layers;
        let layers = req
            .layers
            .unwrap_or_else(|| default_steering_layers(num_layers));

        let result = experiments::run_logit_lens_comparison(
            &state,
            &req.concept,
            scale,
            variant,
            &layers,
            &req.tracked_tokens,
        )
        .map_err(|e| err500(e))?;

        let exp_id = save_experiment(
            &state,
            format!("logit_lens_cmp: {} ({})", req.concept, variant),
            "logit_lens_comparison",
            &format!("concept={}, variant={}", req.concept, variant),
            Some(layers),
            Some(scale),
            ExperimentResults {
                logit_lens_comparison: Some(result.clone()),
                ..empty_results()
            },
        );

        Ok(Json(serde_json::json!({
            "experiment_id": exp_id,
            "result": result,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/run/top_of_mind ───────────────────────────────────────

pub async fn api_run_top_of_mind(
    State(state): AppState,
    Json(req): Json<RunTopOfMindRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let scale = req.scale.unwrap_or(8.0);
        let max_tokens = req.max_tokens.unwrap_or(128);
        let temperature = req.temperature.unwrap_or(0.7);
        let top_p = req.top_p.unwrap_or(0.9);
        let num_layers = state.model_info.num_layers;
        let layers = req
            .layers
            .unwrap_or_else(|| default_steering_layers(num_layers));

        let svec_vectors = {
            let vecs = state.steering_vectors.read().map_err(|e| err500(e))?;
            let svec = vecs.get(&req.concept).ok_or_else(|| {
                err404(format!("steering vector '{}' not found", req.concept))
            })?;
            svec.vectors.clone()
        };

        let prompt = "<|im_start|>user\nWhat are you thinking about?\n<|im_end|>\n<|im_start|>assistant\nI'm thinking about";

        let model = state.model.lock().map_err(|e| err500(e))?;
        let gen_result = {
            let run = (|| -> anyhow::Result<_> {
                steering::apply_steering_vectors(&model, &svec_vectors, &layers, scale)?;
                model.generate(prompt, max_tokens, temperature, Some(top_p))
            })();
            model.clear_steering_vectors();
            run.map_err(|e| err500(e))?
        };

        let result = TopOfMindResult {
            concept: req.concept.clone(),
            scale,
            layers: layers.clone(),
            prompt: prompt.to_string(),
            generated_text: format!("I'm thinking about{}", gen_result.text),
            num_tokens: gen_result.token_ids.len(),
            temperature,
            stop_reason: format!("{:?}", gen_result.stop_reason),
        };

        let exp_id = save_experiment(
            &state,
            format!("top_of_mind: {}", req.concept),
            "top_of_mind",
            prompt,
            Some(layers),
            Some(scale),
            ExperimentResults {
                top_of_mind: Some(result.clone()),
                ..empty_results()
            },
        );

        Ok(Json(serde_json::json!({
            "experiment_id": exp_id,
            "result": result,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/run/concept_activation ────────────────────────────────

pub async fn api_run_concept_activation(
    State(state): AppState,
    Json(req): Json<MeasureConceptActivationRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let svec_vectors = {
            let vecs = state.steering_vectors.read().map_err(|e| err500(e))?;
            let svec = vecs
                .get(&req.concept)
                .ok_or_else(|| err404(format!("steering vector '{}' not found", req.concept)))?;
            svec.vectors.clone()
        };

        let measure_layers: HashSet<usize> = req
            .layers
            .map(|l| l.into_iter().collect())
            .unwrap_or_else(|| svec_vectors.keys().copied().collect());

        let model = state.model.lock().map_err(|e| err500(e))?;
        let result = model
            .forward_introspect_layers(&req.text, Some(measure_layers.clone()))
            .map_err(|e| err500(e))?;

        let mut sorted_layers: Vec<usize> = measure_layers.iter().copied().collect();
        sorted_layers.sort();

        let mut activation_list = Vec::new();
        let mut total = 0.0f32;
        let mut max_act = f32::NEG_INFINITY;
        let mut max_layer = 0usize;

        for (vec_idx, &layer_idx) in sorted_layers.iter().enumerate() {
            if let Some(direction) = svec_vectors.get(&layer_idx) {
                if vec_idx < result.hidden_states.len() {
                    let hs = &result.hidden_states[vec_idx];
                    let seq_len = hs.dim(1).map_err(|e| err500(e))?;
                    let last = hs
                        .i((0, seq_len - 1))
                        .map_err(|e| err500(e))?
                        .to_dtype(candle_core::DType::F32)
                        .map_err(|e| err500(e))?;
                    let cv = candle_core::Tensor::new(direction.as_slice(), model.device())
                        .map_err(|e| err500(e))?
                        .to_dtype(candle_core::DType::F32)
                        .map_err(|e| err500(e))?;
                    let dot = (&last * &cv)
                        .map_err(|e| err500(e))?
                        .sum_all()
                        .map_err(|e| err500(e))?
                        .to_scalar::<f32>()
                        .map_err(|e| err500(e))?;

                    activation_list.push(ConceptLayerActivation {
                        layer_idx,
                        activation: dot,
                    });
                    total += dot;
                    if dot > max_act {
                        max_act = dot;
                        max_layer = layer_idx;
                    }
                }
            }
        }

        let n = activation_list.len().max(1) as f32;
        let mean_activation = total / n;
        if max_act == f32::NEG_INFINITY {
            max_act = 0.0;
        }

        let ca_result = ConceptActivationResult {
            concept: req.concept.clone(),
            text: req.text.clone(),
            activations: activation_list,
            mean_activation,
            max_activation: max_act,
            max_layer,
        };

        let exp_id = save_experiment(
            &state,
            format!("activation: {}", req.concept),
            "concept_activation",
            &req.text,
            None,
            None,
            ExperimentResults {
                concept_activation: Some(ca_result.clone()),
                ..empty_results()
            },
        );

        Ok(Json(serde_json::json!({
            "experiment_id": exp_id,
            "result": ca_result,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/run/layer_type_lens ───────────────────────────────────

pub async fn api_run_layer_type_lens(
    State(state): AppState,
    Json(req): Json<RunLayerTypeLensRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let result = experiments::run_layer_type_lens(&state, &req.text).map_err(|e| err500(e))?;

        let exp_id = save_experiment(
            &state,
            format!("layer_type_lens: {}", truncate_for_label(&req.text, 40)),
            "layer_type_lens",
            &req.text,
            None,
            None,
            ExperimentResults {
                layer_type_lens: Some(result.clone()),
                ..empty_results()
            },
        );

        Ok(Json(serde_json::json!({
            "experiment_id": exp_id,
            "result": result,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/run/steering_survival ─────────────────────────────────

pub async fn api_run_steering_survival(
    State(state): AppState,
    Json(req): Json<RunSteeringSurvivalRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let scale = req.scale.unwrap_or(8.0);
        let probe = req
            .probe_text
            .as_deref()
            .unwrap_or("The meaning of life is");

        let result = experiments::run_steering_survival(
            &state,
            &req.concept,
            &req.injection_layers,
            scale,
            probe,
        )
        .map_err(|e| err500(e))?;

        let exp_id = save_experiment(
            &state,
            format!("steering_survival: {}", req.concept),
            "steering_survival",
            &format!(
                "concept={}, injection_layers={:?}",
                req.concept, req.injection_layers
            ),
            Some(req.injection_layers),
            Some(scale),
            ExperimentResults {
                steering_survival: Some(result.clone()),
                ..empty_results()
            },
        );

        Ok(Json(serde_json::json!({
            "experiment_id": exp_id,
            "result": result,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/run/cka ───────────────────────────────────────────────

pub async fn api_run_cka(
    State(state): AppState,
    Json(req): Json<RunCkaRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let default_texts: Vec<String> = vec![
            "The capital of France is",
            "def fibonacci(n):",
            "Once upon a time in a",
            "The chemical formula for water is",
            "In quantum mechanics, the",
            "The stock market crashed because",
            "She walked into the room and",
            "According to Einstein's theory of",
            "The recipe calls for two cups of",
            "In the year 2050, humanity will",
            "The function returns a pointer to",
            "Love is a many-splendored thing",
            "The president announced that",
            "To solve this equation, first",
            "The cat sat on the mat and",
            "In machine learning, overfitting",
            "The old house at the end of",
            "Rust's ownership system ensures",
            "The patient presented with symptoms of",
            "During the Renaissance, artists began",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let texts = req.texts.unwrap_or(default_texts);
        let num_texts = texts.len();
        let result = experiments::run_cka(&state, &texts).map_err(|e| err500(e))?;

        let exp_id = save_experiment(
            &state,
            format!("cka: {} texts", num_texts),
            "cka",
            &format!("{} texts", num_texts),
            None,
            None,
            ExperimentResults {
                cka: Some(result.clone()),
                ..empty_results()
            },
        );

        Ok(Json(serde_json::json!({
            "experiment_id": exp_id,
            "result": result,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/run/routing_analysis ──────────────────────────────────

pub async fn api_run_routing_analysis(
    State(state): AppState,
    Json(req): Json<RunRoutingAnalysisRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let scale = req.scale.unwrap_or(8.0);
        let layers = req.layers.unwrap_or_else(|| (5..26).collect());

        let result = experiments::run_routing_analysis(
            &state,
            &req.text,
            req.concept.as_deref(),
            scale,
            &layers,
        )
        .map_err(|e| err500(e))?;

        let exp_id = save_experiment(
            &state,
            format!("routing: {}", truncate_for_label(&req.text, 30)),
            "routing_analysis",
            &req.text,
            Some(layers),
            Some(scale),
            ExperimentResults {
                routing_analysis: Some(result.clone()),
                ..empty_results()
            },
        );

        Ok(Json(serde_json::json!({
            "experiment_id": exp_id,
            "result": result,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/run/causal_tracing ────────────────────────────────────

pub async fn api_run_causal_tracing(
    State(state): AppState,
    Json(req): Json<RunCausalTracingRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let result = experiments::run_causal_tracing(&state, &req.clean_text, &req.corrupted_text)
            .map_err(|e| err500(e))?;

        let exp_id = save_experiment(
            &state,
            format!("causal_trace: {}", truncate_for_label(&req.clean_text, 30)),
            "causal_tracing",
            &format!(
                "clean='{}', corrupted='{}'",
                req.clean_text, req.corrupted_text
            ),
            None,
            None,
            ExperimentResults {
                causal_tracing: Some(result.clone()),
                ..empty_results()
            },
        );

        Ok(Json(serde_json::json!({
            "experiment_id": exp_id,
            "result": result,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/run/gdn_state_stats ───────────────────────────────────

pub async fn api_run_gdn_state_stats(
    State(state): AppState,
    Json(req): Json<RunGdnStateStatsRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let result = experiments::run_gdn_state_stats(&state, &req.text).map_err(|e| err500(e))?;

        let exp_id = save_experiment(
            &state,
            format!("gdn_stats: {}", truncate_for_label(&req.text, 30)),
            "gdn_state_stats",
            &req.text,
            None,
            None,
            ExperimentResults {
                gdn_state_stats: Some(result.clone()),
                ..empty_results()
            },
        );

        Ok(Json(serde_json::json!({
            "experiment_id": exp_id,
            "result": result,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/run/code_logit_diff ────────────────────────────────────

pub async fn api_run_code_logit_diff(
    State(state): AppState,
    Json(req): Json<RunCodeLogitDiffRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let scale = req.scale.unwrap_or(8.0);
        let num_layers = state.model_info.num_layers;
        let layers = req
            .layers
            .unwrap_or_else(|| default_steering_layers(num_layers));
        let top_k = req.top_k.unwrap_or(10);

        let result =
            experiments::run_code_logit_diff(&state, &req.concept, scale, &layers, top_k)
                .map_err(|e| err500(e))?;

        let exp_id = save_experiment(
            &state,
            format!("code_logit_diff: {}", req.concept),
            "code_logit_diff",
            &format!("concept={}", req.concept),
            Some(layers),
            Some(scale),
            ExperimentResults {
                code_logit_diff: Some(result.clone()),
                ..empty_results()
            },
        );

        Ok(Json(serde_json::json!({
            "experiment_id": exp_id,
            "result": result,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/run/code_gen_detection ────────────────────────────────

pub async fn api_run_code_gen_detection(
    State(state): AppState,
    Json(req): Json<RunCodeGenDetectionRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let scale = req.scale.unwrap_or(8.0);
        let num_layers = state.model_info.num_layers;
        let layers = req
            .layers
            .unwrap_or_else(|| default_steering_layers(num_layers));
        let max_tokens = req.max_tokens.unwrap_or(32);
        let temperatures = req.temperatures.unwrap_or_else(|| vec![0.0, 0.7]);

        let result = experiments::run_code_gen_detection(
            &state,
            &req.concept,
            scale,
            &layers,
            max_tokens,
            &temperatures,
        )
        .map_err(|e| err500(e))?;

        let exp_id = save_experiment(
            &state,
            format!("code_gen_detection: {}", req.concept),
            "code_gen_detection",
            &format!("concept={}", req.concept),
            Some(layers),
            Some(scale),
            ExperimentResults {
                code_gen_detection: Some(result.clone()),
                ..empty_results()
            },
        );

        Ok(Json(serde_json::json!({
            "experiment_id": exp_id,
            "result": result,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/run/concept_identification ────────────────────────────

pub async fn api_run_concept_identification(
    State(state): AppState,
    Json(req): Json<RunConceptIdentificationRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let scale = req.scale.unwrap_or(8.0);
        let num_layers = state.model_info.num_layers;
        let layers = req
            .layers
            .unwrap_or_else(|| default_steering_layers(num_layers));

        let result = experiments::run_concept_identification(
            &state,
            &req.concepts,
            scale,
            &layers,
        )
        .map_err(|e| err500(e))?;

        let exp_id = save_experiment(
            &state,
            format!("concept_id: {:?}", req.concepts),
            "concept_identification",
            &format!("concepts={:?}", req.concepts),
            Some(layers),
            Some(scale),
            ExperimentResults {
                concept_identification: Some(result.clone()),
                ..empty_results()
            },
        );

        Ok(Json(serde_json::json!({
            "experiment_id": exp_id,
            "result": result,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── POST /api/run/discrimination_matrix ─────────────────────────────

pub async fn api_run_discrimination_matrix(
    State(state): AppState,
    Json(req): Json<RunDiscriminationMatrixRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let scale = req.scale.unwrap_or(8.0);
        let num_layers = state.model_info.num_layers;
        let layers = req
            .layers
            .unwrap_or_else(|| default_steering_layers(num_layers));

        let result = experiments::run_discrimination_matrix(
            &state,
            &req.concepts,
            scale,
            &layers,
        )
        .map_err(|e| err500(e))?;

        let exp_id = save_experiment(
            &state,
            format!("discrimination: {:?}", req.concepts),
            "discrimination_matrix",
            &format!("concepts={:?}", req.concepts),
            Some(layers),
            Some(scale),
            ExperimentResults {
                discrimination_matrix: Some(result.clone()),
                ..empty_results()
            },
        );

        Ok(Json(serde_json::json!({
            "experiment_id": exp_id,
            "result": result,
        })))
    })
    .await
    .map_err(|e| err500(e))?
}

// ── Helpers ─────────────────────────────────────────────────────────

fn empty_results() -> ExperimentResults {
    ExperimentResults {
        logit_lens: None,
        logit_diff: None,
        control_questions: None,
        logit_lens_comparison: None,
        top_of_mind: None,
        concept_activation: None,
        cka: None,
        routing_analysis: None,
        causal_tracing: None,
        steering_survival: None,
        gdn_state_stats: None,
        layer_type_lens: None,
        code_logit_diff: None,
        code_gen_detection: None,
        concept_identification: None,
        discrimination_matrix: None,
    }
}

fn save_experiment(
    state: &SharedState,
    name: String,
    experiment_type: &str,
    prompt: &str,
    steering_layers: Option<Vec<usize>>,
    steering_scale: Option<f64>,
    results: ExperimentResults,
) -> String {
    let exp_id = uuid::Uuid::new_v4().to_string();
    let experiment = Experiment {
        id: exp_id.clone(),
        name,
        status: "completed".to_string(),
        created_at: now_timestamp(),
        config: ExperimentConfig {
            experiment_type: experiment_type.to_string(),
            prompt: prompt.to_string(),
            steering_layers,
            steering_scale,
        },
        results: Some(results),
    };

    if let Ok(mut experiments) = state.experiments.write() {
        db::save_experiment(&state.db_path, &experiment).ok();
        experiments.insert(exp_id.clone(), experiment);
    }
    exp_id
}

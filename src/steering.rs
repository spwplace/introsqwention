//! Steering vector training via contrastive PCA.
//!
//! Implements the "pca_center" method from repeng: collect hidden-state
//! differences between positive/negative prompt pairs, then extract
//! the top principal component per layer via power iteration.

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Tensor};
use mistralrs_core::introspection::IntrospectionModel;

use crate::prompts::TRUNCATED_OUTPUTS_JSON;

// ── ChatML Formatting ───────────────────────────────────────────────

/// A single message in a chat conversation.
pub struct ChatMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

/// Render a conversation as ChatML text.
///
/// If `continue_final` is true, the last message is left open (no `<|im_end|>`)
/// so the model continues generating from it.
/// If `add_generation_prompt` is true, an `<|im_start|>assistant\n` is appended.
pub fn format_chatml(
    messages: &[ChatMessage<'_>],
    continue_final: bool,
    add_generation_prompt: bool,
) -> String {
    let mut out = String::new();
    for (i, msg) in messages.iter().enumerate() {
        let is_last = i == messages.len() - 1;
        out.push_str(&format!("<|im_start|>{}\n{}", msg.role, msg.content));
        if is_last && continue_final {
            // Leave open — no im_end
        } else {
            out.push_str("<|im_end|>\n");
        }
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

/// Build the generation prompt for concept vector training.
///
/// Equivalent to Thebes `generation_prompt(persona)`:
/// ```text
/// <|im_start|>system\n<|im_end|>\n<|im_start|>user\nPlease talk about {concept}.<|im_end|>\n<|im_start|>assistant\n
/// ```
fn training_prompt(concept: &str) -> String {
    format_chatml(
        &[
            ChatMessage {
                role: "system",
                content: "",
            },
            ChatMessage {
                role: "user",
                content: &format!("Please talk about {}.", concept),
            },
        ],
        false,
        true,
    )
}

// ── Power Iteration PCA ─────────────────────────────────────────────

/// Extract the top principal component from a matrix via power iteration.
///
/// `diff_matrix` has shape `(N, D)` where N = number of samples, D = hidden_size.
/// Returns a unit vector of shape `(D,)` — the top eigenvector of the covariance matrix.
///
/// Uses the "pca_center" method: center the data, then find the direction of
/// maximum variance via `v <- A^T (A v) / ||A^T (A v)||` iterated.
pub fn power_iteration_pca(
    diff_matrix: &Tensor,
    num_iters: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let diff_f32 = diff_matrix.to_dtype(DType::F32)?;
    let (_n, d) = (diff_f32.dim(0)?, diff_f32.dim(1)?);

    // Center: subtract mean across samples
    let mean = diff_f32.mean(0)?; // (D,)
    let centered = diff_f32.broadcast_sub(&mean)?; // (N, D)

    // Initialize v randomly — use a deterministic pattern for reproducibility
    let init_data: Vec<f32> = (0..d)
        .map(|i| ((i * 7 + 13) % 101) as f32 / 101.0 - 0.5)
        .collect();
    let mut v = Tensor::new(init_data.as_slice(), device)?.to_dtype(DType::F32)?; // (D,)
    let norm = v.sqr()?.sum_all()?.sqrt()?;
    v = v.broadcast_div(&norm)?;

    // Power iteration: v <- A^T @ (A @ v) / ||...||
    // A = centered, shape (N, D)
    // A @ v = (N, D) @ (D,) = (N,)     — project samples onto v
    // A^T @ (A@v) = (D, N) @ (N,) = (D,) — weighted sum of samples
    let centered_t = centered.t()?; // (D, N)

    for _ in 0..num_iters {
        let av = centered.matmul(&v.unsqueeze(1)?)?.squeeze(1)?; // (N,)
        let atav = centered_t.matmul(&av.unsqueeze(1)?)?.squeeze(1)?; // (D,)
        let norm = atav.sqr()?.sum_all()?.sqrt()?;
        v = atav.broadcast_div(&norm)?;
    }

    // Sign convention: make the largest-magnitude component positive
    let v_vec: Vec<f32> = v.to_vec1()?;
    let max_abs_idx = v_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    if v_vec[max_abs_idx] < 0.0 {
        v = v.neg()?;
    }

    Ok(v)
}

// ── Steering Vector Training ────────────────────────────────────────

/// Result of training a steering vector.
pub struct TrainedSteeringVectors {
    /// Per-layer direction vectors (layer_idx -> unit direction of shape (hidden_size,)).
    /// layer_idx follows the introspection convention: 0 = embedding, 1..N = decoder layers.
    pub vectors: HashMap<usize, Vec<f32>>,
    /// Number of prompt pairs used for training.
    pub num_pairs: usize,
}

/// Progress callback signature: (completed_pairs, total_pairs).
pub type ProgressCallback = Box<dyn Fn(usize, usize) + Send>;

/// Train concept steering vectors using the pca_center method.
///
/// For each suffix, creates positive ("Please talk about {concept}") and negative
/// ("Please talk about anything") prompt pairs, runs forward passes to collect
/// hidden states, computes per-layer differences, and extracts the top PCA component.
///
/// `num_suffixes` limits the number of suffixes to use (None = all non-empty suffixes).
/// `target_layers` specifies which introspection layers to train on (None = all decoder layers).
pub fn train_concept_vector(
    model: &IntrospectionModel,
    concept: &str,
    num_suffixes: Option<usize>,
    target_layers: Option<&[usize]>,
    progress: Option<ProgressCallback>,
) -> anyhow::Result<TrainedSteeringVectors> {
    // Parse suffixes
    let mut suffixes: Vec<String> = serde_json::from_str(TRUNCATED_OUTPUTS_JSON)?;
    suffixes.retain(|s| !s.trim().is_empty());
    let suffixes = if let Some(n) = num_suffixes {
        &suffixes[..n.min(suffixes.len())]
    } else {
        &suffixes
    };
    let total = suffixes.len();
    if total == 0 {
        anyhow::bail!("No training suffixes available after filtering.");
    }

    tracing::info!(
        "Training steering vector for '{}' with {} prompt pairs",
        concept,
        total
    );

    // Build prompt prefixes
    let positive_prefix = training_prompt(concept);
    let negative_prefix = training_prompt("anything");

    // Determine which layers to capture
    let info = model.model_info();
    let num_introspection_layers = info.num_layers + 1; // +1 for embedding at index 0
    let layer_indices: Vec<usize> = if let Some(layers) = target_layers {
        let mut ls = layers.to_vec();
        ls.sort_unstable();
        ls.dedup();
        for &layer_idx in &ls {
            if layer_idx == 0 {
                anyhow::bail!(
                    "Layer 0 (embedding) is unsupported for steering vectors. \
                     Use introspection decoder layers in 1..={}.",
                    info.num_layers
                );
            }
            if layer_idx >= num_introspection_layers {
                anyhow::bail!(
                    "Layer {} out of range for model with {} decoder layers \
                     (valid introspection layers: 1..={}).",
                    layer_idx,
                    info.num_layers,
                    info.num_layers
                );
            }
        }
        ls
    } else {
        // All decoder layers (skip embedding at index 0)
        (1..num_introspection_layers).collect()
    };
    if layer_indices.is_empty() {
        anyhow::bail!("No valid layers selected for steering-vector training.");
    }

    let hidden_size = info.hidden_size;
    let device = model.device();

    // Accumulate per-layer difference vectors: layer_idx -> Vec of (hidden_size,) vecs
    let mut layer_diffs: HashMap<usize, Vec<Vec<f32>>> = HashMap::new();
    for &li in &layer_indices {
        layer_diffs.insert(li, Vec::with_capacity(total));
    }

    // Process each pair
    let capture_set: std::collections::HashSet<usize> = layer_indices.iter().copied().collect();

    for (pair_idx, suffix) in suffixes.iter().enumerate() {
        let positive_text = format!("{}{}", positive_prefix, suffix);
        let negative_text = format!("{}{}", negative_prefix, suffix);

        // Forward passes with selective layer capture
        let pos_result =
            model.forward_introspect_layers(&positive_text, Some(capture_set.clone()))?;
        let neg_result =
            model.forward_introspect_layers(&negative_text, Some(capture_set.clone()))?;

        // Extract last-token hidden state per layer and compute difference
        for (vec_idx, &layer_idx) in layer_indices.iter().enumerate() {
            let pos_hs = &pos_result.hidden_states[vec_idx]; // (1, seq_len, hidden_size)
            let neg_hs = &neg_result.hidden_states[vec_idx];

            let pos_last = pos_hs
                .i((0, pos_hs.dim(1)? - 1))?
                .contiguous()?
                .to_dtype(DType::F32)?; // (hidden_size,)
            let neg_last = neg_hs
                .i((0, neg_hs.dim(1)? - 1))?
                .contiguous()?
                .to_dtype(DType::F32)?;

            let diff = pos_last.sub(&neg_last)?;
            let diff_vec: Vec<f32> = diff.to_vec1()?;
            let bucket = layer_diffs.get_mut(&layer_idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "internal error: missing accumulation bucket for layer {}",
                    layer_idx
                )
            })?;
            bucket.push(diff_vec);
        }

        if let Some(ref cb) = progress {
            cb(pair_idx + 1, total);
        }

        if (pair_idx + 1) % 50 == 0 || pair_idx + 1 == total {
            tracing::info!(
                "  Training progress: {}/{} pairs processed",
                pair_idx + 1,
                total
            );
        }
    }

    // PCA per layer
    tracing::info!("Computing PCA for {} layers...", layer_indices.len());
    let mut vectors: HashMap<usize, Vec<f32>> = HashMap::new();

    for &layer_idx in &layer_indices {
        let diffs = layer_diffs.remove(&layer_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "internal error: missing accumulated diffs for layer {}",
                layer_idx
            )
        })?;
        let n = diffs.len();

        // Build (N, hidden_size) tensor
        let flat: Vec<f32> = diffs.into_iter().flatten().collect();
        let diff_matrix = Tensor::new(flat.as_slice(), device)?.reshape((n, hidden_size))?;

        let direction = power_iteration_pca(&diff_matrix, 20, device)?;
        let direction_vec: Vec<f32> = direction.to_vec1()?;
        vectors.insert(layer_idx, direction_vec);
    }

    tracing::info!("Steering vector training complete for '{}'", concept);

    Ok(TrainedSteeringVectors {
        vectors,
        num_pairs: total,
    })
}

/// Apply a stored steering vector set to the model.
pub fn apply_steering_vectors(
    model: &IntrospectionModel,
    vectors: &HashMap<usize, Vec<f32>>,
    layers: &[usize],
    scale: f64,
) -> anyhow::Result<()> {
    let device = model.device();
    let dtype = model.dtype();
    let info = model.model_info();
    let max_introspection_layer = info.num_layers;

    let mut unique_layers = layers.to_vec();
    unique_layers.sort_unstable();
    unique_layers.dedup();

    for &layer_idx in &unique_layers {
        if layer_idx == 0 {
            anyhow::bail!(
                "Layer 0 (embedding) cannot be steered with set_steering_vector. \
                 Use decoder introspection layers in 1..={}.",
                max_introspection_layer
            );
        }
        if layer_idx > max_introspection_layer {
            anyhow::bail!(
                "Layer {} out of range (valid introspection layers: 1..={}).",
                layer_idx,
                max_introspection_layer
            );
        }

        let vec_data = vectors.get(&layer_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "No trained steering vector for introspection layer {}.",
                layer_idx
            )
        })?;
        let vector = Tensor::new(vec_data.as_slice(), device)?.to_dtype(dtype)?;
        let scaled = (&vector * scale)?;

        // The model API uses 0-based decoder layer indices.
        let decoder_layer_idx = layer_idx - 1;
        model.set_steering_vector(decoder_layer_idx, scaled);
    }
    Ok(())
}

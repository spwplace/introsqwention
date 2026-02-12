//! Experiment execution logic for introspection detection, control questions,
//! and comparative logit lens analysis.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use candle_core::{DType, IndexOp};
use mistralrs_core::introspection::IntrospectionModel;

use crate::prompts;
use crate::state::*;
use crate::steering::{apply_steering_vectors, format_chatml, ChatMessage};

/// Build the standard multi-turn detection conversation as ChatML text.
///
/// The conversation structure matches Thebes exactly:
/// 1. system: ""
/// 2. user: {user_turn1} (selected by variant)
/// 3. assistant: "{ }"
/// 4. user: {final_user_content} (detection question or control question)
/// 5. assistant: "The answer is" (continue_final=true, left open for measurement)
fn build_detection_conversation(
    user_turn1_variant: &str,
    final_user_content: &str,
) -> anyhow::Result<String> {
    let turn1_text = prompts::user_turn1_by_variant(user_turn1_variant).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown turn1 variant '{}'. Use: no_info, with_info, inaccurate_info",
            user_turn1_variant
        )
    })?;

    Ok(format_chatml(
        &[
            ChatMessage {
                role: "system",
                content: "",
            },
            ChatMessage {
                role: "user",
                content: turn1_text,
            },
            ChatMessage {
                role: "assistant",
                content: prompts::ASST_TURN_1,
            },
            ChatMessage {
                role: "user",
                content: final_user_content,
            },
            ChatMessage {
                role: "assistant",
                content: prompts::ASST_ANSWER_PREFIX,
            },
        ],
        true,  // continue_final — leave "The answer is" open
        false, // no generation prompt
    ))
}

/// Ensure steering state is cleared before and after a model operation.
/// This avoids experiment contamination if the operation fails.
fn with_clean_steering<R, F>(model: &IntrospectionModel, f: F) -> anyhow::Result<R>
where
    F: FnOnce(&IntrospectionModel) -> anyhow::Result<R>,
{
    model.clear_steering_vectors();
    let out = f(model);
    model.clear_steering_vectors();
    out
}

/// Extract softmax probabilities at the last token position from logits tensor.
fn last_token_probs(logits: &candle_core::Tensor) -> anyhow::Result<Vec<f32>> {
    let seq_len = logits.dim(1)?;
    let last_logits = logits.i((0, seq_len - 1))?.to_dtype(DType::F32)?; // (vocab_size,)
                                                                         // Softmax
    let max_val = last_logits.max(0)?;
    let shifted = last_logits.broadcast_sub(&max_val)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_all()?;
    let probs = exp.broadcast_div(&sum)?;
    Ok(probs.to_vec1()?)
}

fn mean_and_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

fn logprob_for_token(logits_vec: &[f32], token_id: u32) -> f32 {
    let idx = token_id as usize;
    if idx >= logits_vec.len() {
        return f32::NEG_INFINITY;
    }
    let max_logit = logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp = logits_vec
        .iter()
        .map(|&x| (x - max_logit).exp())
        .sum::<f32>();
    logits_vec[idx] - max_logit - sum_exp.ln()
}

/// Compute log P(completion | prompt) exactly by teacher forcing all completion tokens.
#[allow(dead_code)]
fn completion_logprob(
    model: &IntrospectionModel,
    prompt: &str,
    completion: &str,
) -> anyhow::Result<(Vec<u32>, f32)> {
    let (prompt_ids, _) = model.tokenize(prompt)?;
    let full_text = format!("{}{}", prompt, completion);
    let (full_ids, _) = model.tokenize(&full_text)?;
    if !full_ids.starts_with(prompt_ids.as_slice()) {
        anyhow::bail!(
            "Completion tokenization changed prompt prefix; cannot align sequence logprob safely."
        );
    }
    if full_ids.len() <= prompt_ids.len() {
        anyhow::bail!("Completion {:?} produced zero tokens", completion);
    }
    let completion_ids = full_ids[prompt_ids.len()..].to_vec();

    let result = model.forward_introspect(&full_text)?;
    let logits = result.logits.to_dtype(DType::F32)?;

    let mut logprob = 0.0f32;
    for (i, &tok) in completion_ids.iter().enumerate() {
        let pos = prompt_ids.len() + i - 1;
        let logits_vec: Vec<f32> = logits.i((0, pos))?.to_vec1()?;
        logprob += logprob_for_token(&logits_vec, tok);
    }

    Ok((completion_ids, logprob))
}

/// Sum sequence-level probabilities across unique completion tokenizations.
#[allow(dead_code)]
fn completion_set_prob(
    model: &IntrospectionModel,
    prompt: &str,
    candidates: &[&str],
) -> anyhow::Result<f32> {
    let mut seen = HashSet::<Vec<u32>>::new();
    let mut total = 0.0f32;

    for &candidate in candidates {
        let (completion_ids, logprob) = completion_logprob(model, prompt, candidate)?;
        if !seen.insert(completion_ids) {
            continue;
        }
        total += logprob.exp();
    }

    Ok(total)
}

#[allow(dead_code)]
fn yes_no_completion_probs(model: &IntrospectionModel, prompt: &str) -> anyhow::Result<(f32, f32)> {
    let p_yes = completion_set_prob(model, prompt, prompts::YES_COMPLETION_CANDIDATES)?;
    let p_no = completion_set_prob(model, prompt, prompts::NO_COMPLETION_CANDIDATES)?;
    Ok((p_yes, p_no))
}

/// Extract P(yes) and P(no) from a pre-computed probability vector at the last token position.
///
/// For single-token completions (" yes", " no"), this is equivalent to running
/// `yes_no_completion_probs` but reuses an already-computed forward pass — saving
/// 2 forward passes per call. The causal mask ensures logits at the last prompt
/// position are identical regardless of what tokens follow.
fn yes_no_probs_from_last_token(
    model: &IntrospectionModel,
    probs_vec: &[f32],
) -> anyhow::Result<(f32, f32)> {
    let p_yes = completion_candidates_prob(model, probs_vec, prompts::YES_COMPLETION_CANDIDATES)?;
    let p_no = completion_candidates_prob(model, probs_vec, prompts::NO_COMPLETION_CANDIDATES)?;
    Ok((p_yes, p_no))
}

/// Sum probabilities for completion candidates directly from a softmax distribution.
/// Only works for single-token completions — falls back to error for multi-token.
fn completion_candidates_prob(
    model: &IntrospectionModel,
    probs_vec: &[f32],
    candidates: &[&str],
) -> anyhow::Result<f32> {
    let mut seen = HashSet::<u32>::new();
    let mut total = 0.0f32;

    for &candidate in candidates {
        let (ids, _) = model.tokenize(candidate)?;
        if ids.len() != 1 {
            anyhow::bail!(
                "completion_candidates_prob requires single-token completions, \
                 but {:?} tokenized to {} tokens",
                candidate,
                ids.len()
            );
        }
        let tok_id = ids[0];
        if !seen.insert(tok_id) {
            continue;
        }
        total += probs_vec.get(tok_id as usize).copied().unwrap_or(0.0);
    }
    Ok(total)
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn deterministic_random_unit_vector(dim: usize, mut seed: u64) -> Vec<f32> {
    let mut vec = Vec::with_capacity(dim);
    for _ in 0..dim {
        let r = splitmix64(&mut seed);
        let u = ((r >> 11) as f64) / ((1u64 << 53) as f64);
        vec.push((u as f32) * 2.0 - 1.0);
    }
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in &mut vec {
        *x /= norm;
    }
    vec
}

fn random_control_vectors(
    concept: &str,
    layers: &[usize],
    reference_vectors: &HashMap<usize, Vec<f32>>,
) -> HashMap<usize, Vec<f32>> {
    let mut out = HashMap::new();
    for &layer_idx in layers {
        if let Some(reference) = reference_vectors.get(&layer_idx) {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            concept.hash(&mut hasher);
            layer_idx.hash(&mut hasher);
            let seed = hasher.finish();
            out.insert(
                layer_idx,
                deterministic_random_unit_vector(reference.len(), seed),
            );
        }
    }
    out
}

// ── Logit Diff Experiment ───────────────────────────────────────────

/// Run the introspection detection experiment (logit diff).
///
/// Compares P(yes) vs P(no) at "The answer is" position between
/// a base (no steering) and steered forward pass.
pub fn run_logit_diff(
    state: &SharedState,
    concept: &str,
    scale: f64,
    user_turn1_variant: &str,
    layers: &[usize],
    top_k: usize,
) -> anyhow::Result<LogitDiffResult> {
    // Look up the named steering vector — clone and drop the read lock
    // before acquiring the model mutex (same pattern as run_control_questions)
    let svec_vectors = {
        let steering_vecs = state
            .steering_vectors
            .read()
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let svec = steering_vecs.get(concept).ok_or_else(|| {
            anyhow::anyhow!(
                "Steering vector '{}' not found. Train one first with train_steering_vector.",
                concept
            )
        })?;
        svec.vectors.clone()
    };
    let random_vectors = random_control_vectors(concept, layers, &svec_vectors);

    let mut trials = Vec::new();
    let mut first_trial_base_probs: Option<Vec<f32>> = None;
    let mut first_trial_steered_probs: Option<Vec<f32>> = None;

    for &question in prompts::DETECTION_QUESTION_VARIANTS {
        let conversation = build_detection_conversation(user_turn1_variant, question)?;

        // Base forward pass (no steering) — single forward, extract yes/no from logits
        let (base_probs, base_p_yes, base_p_no) = {
            let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
            with_clean_steering(&model, |m| {
                let result = m.forward_introspect(&conversation)?;
                let probs = last_token_probs(&result.logits)?;
                let (p_yes, p_no) = yes_no_probs_from_last_token(m, &probs)?;
                Ok((probs, p_yes, p_no))
            })?
        };

        // Steered forward pass — single forward
        let (steered_probs, steered_p_yes, steered_p_no) = {
            let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
            with_clean_steering(&model, |m| {
                apply_steering_vectors(m, &svec_vectors, layers, scale)?;
                let result = m.forward_introspect(&conversation)?;
                let probs = last_token_probs(&result.logits)?;
                let (p_yes, p_no) = yes_no_probs_from_last_token(m, &probs)?;
                Ok((probs, p_yes, p_no))
            })?
        };

        // Random-direction steering control — single forward
        let (random_p_yes, random_p_no) = {
            let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
            with_clean_steering(&model, |m| {
                apply_steering_vectors(m, &random_vectors, layers, scale)?;
                let result = m.forward_introspect(&conversation)?;
                let probs = last_token_probs(&result.logits)?;
                yes_no_probs_from_last_token(m, &probs)
            })?
        };

        if first_trial_base_probs.is_none() {
            first_trial_base_probs = Some(base_probs.clone());
            first_trial_steered_probs = Some(steered_probs.clone());
        }

        trials.push(LogitDiffTrialResult {
            prompt_variant: question.to_string(),
            base_p_yes,
            base_p_no,
            steered_p_yes,
            steered_p_no,
            random_control_p_yes: random_p_yes,
            random_control_p_no: random_p_no,
        });
    }

    let n = trials.len() as f32;
    let base_p_yes = trials.iter().map(|t| t.base_p_yes).sum::<f32>() / n;
    let base_p_no = trials.iter().map(|t| t.base_p_no).sum::<f32>() / n;
    let steered_p_yes = trials.iter().map(|t| t.steered_p_yes).sum::<f32>() / n;
    let steered_p_no = trials.iter().map(|t| t.steered_p_no).sum::<f32>() / n;
    let random_control_p_yes = trials.iter().map(|t| t.random_control_p_yes).sum::<f32>() / n;
    let random_control_p_no = trials.iter().map(|t| t.random_control_p_no).sum::<f32>() / n;

    let yes_shifts: Vec<f32> = trials
        .iter()
        .map(|t| t.steered_p_yes - t.base_p_yes)
        .collect();
    let (mean_yes_shift, std_yes_shift) = mean_and_std(&yes_shifts);
    let stderr_yes_shift = if yes_shifts.is_empty() {
        0.0
    } else {
        std_yes_shift / (yes_shifts.len() as f32).sqrt()
    };
    let ci_radius = 1.96 * stderr_yes_shift;
    let yes_shift_ci95_low = mean_yes_shift - ci_radius;
    let yes_shift_ci95_high = mean_yes_shift + ci_radius;

    let base_probs = first_trial_base_probs
        .ok_or_else(|| anyhow::anyhow!("No logit-diff trials were executed"))?;
    let steered_probs = first_trial_steered_probs
        .ok_or_else(|| anyhow::anyhow!("No logit-diff trials were executed"))?;

    // Top-k tokens by steered probability
    let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
    let mut indexed: Vec<(usize, f32)> = steered_probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_tokens: Vec<LogitDiffToken> = indexed
        .iter()
        .take(top_k)
        .map(|&(id, steered_p)| {
            let base_p = base_probs.get(id).copied().unwrap_or(0.0);
            LogitDiffToken {
                token: model.decode_token(id as u32),
                token_id: id as u32,
                base_prob: base_p,
                steered_prob: steered_p,
                diff: steered_p - base_p,
            }
        })
        .collect();

    Ok(LogitDiffResult {
        variant: user_turn1_variant.to_string(),
        concept: concept.to_string(),
        scale,
        layers: layers.to_vec(),
        num_trials: trials.len(),
        top_tokens,
        base_p_yes,
        base_p_no,
        steered_p_yes,
        steered_p_no,
        random_control_p_yes,
        random_control_p_no,
        mean_yes_shift,
        std_yes_shift,
        stderr_yes_shift,
        yes_shift_ci95_low,
        yes_shift_ci95_high,
        trials,
    })
}

// ── Control Questions Experiment ────────────────────────────────────

/// Run the control questions experiment.
///
/// Tests whether a steering vector corrupts general factual knowledge
/// by measuring P(yes)/P(no) shifts across 16 factual yes/no questions.
pub fn run_control_questions(
    state: &SharedState,
    concept: &str,
    scale: f64,
    user_turn1_variant: &str,
    layers: &[usize],
) -> anyhow::Result<ControlQuestionsResult> {
    let steering_vecs = state
        .steering_vectors
        .read()
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let svec = steering_vecs
        .get(concept)
        .ok_or_else(|| anyhow::anyhow!("Steering vector '{}' not found.", concept))?;
    // Clone the vectors so we can drop the read lock before locking the model
    let svec_vectors = svec.vectors.clone();
    drop(steering_vecs);

    let mut questions = Vec::new();

    for (q_idx, q) in prompts::CONTROL_QUESTIONS.iter().enumerate() {
        let final_content = format!(
            "{} Answer with \"The answer is yes\" or \"The answer is no\"",
            q.text
        );
        let conversation = build_detection_conversation(user_turn1_variant, &final_content)?;

        // Base forward pass — single forward, extract yes/no from logits
        let (base_p_yes, base_p_no) = {
            let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
            with_clean_steering(&model, |m| {
                let result = m.forward_introspect(&conversation)?;
                let probs = last_token_probs(&result.logits)?;
                yes_no_probs_from_last_token(m, &probs)
            })?
        };

        // Steered forward pass — single forward
        let (steered_p_yes, steered_p_no) = {
            let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
            with_clean_steering(&model, |m| {
                apply_steering_vectors(m, &svec_vectors, layers, scale)?;
                let result = m.forward_introspect(&conversation)?;
                let probs = last_token_probs(&result.logits)?;
                yes_no_probs_from_last_token(m, &probs)
            })?
        };

        let expected_answer = if q.expected_yes { "yes" } else { "no" }.to_string();
        let base_correct_prob = if q.expected_yes {
            base_p_yes
        } else {
            base_p_no
        };
        let steered_correct_prob = if q.expected_yes {
            steered_p_yes
        } else {
            steered_p_no
        };
        let base_pred_yes = base_p_yes >= base_p_no;
        let steered_pred_yes = steered_p_yes >= steered_p_no;

        questions.push(ControlQuestionResult {
            question: q.text.to_string(),
            expected_answer,
            base_p_yes,
            base_p_no,
            steered_p_yes,
            steered_p_no,
            base_correct_prob,
            steered_correct_prob,
            base_predicted_answer: if base_pred_yes { "yes" } else { "no" }.to_string(),
            steered_predicted_answer: if steered_pred_yes { "yes" } else { "no" }.to_string(),
            base_is_correct: base_pred_yes == q.expected_yes,
            steered_is_correct: steered_pred_yes == q.expected_yes,
        });

        tracing::info!(
            "  Control question {}/{}: {}",
            q_idx + 1,
            prompts::CONTROL_QUESTIONS.len(),
            q.text
        );
    }

    // Compute summary statistics
    let n = questions.len() as f32;
    let mean_base_p_yes = questions.iter().map(|q| q.base_p_yes).sum::<f32>() / n;
    let mean_base_p_no = questions.iter().map(|q| q.base_p_no).sum::<f32>() / n;
    let mean_steered_p_yes = questions.iter().map(|q| q.steered_p_yes).sum::<f32>() / n;
    let mean_steered_p_no = questions.iter().map(|q| q.steered_p_no).sum::<f32>() / n;

    let yes_shifts: Vec<f32> = questions
        .iter()
        .map(|q| q.steered_p_yes - q.base_p_yes)
        .collect();
    let no_shifts: Vec<f32> = questions
        .iter()
        .map(|q| q.steered_p_no - q.base_p_no)
        .collect();

    let mean_yes_shift = yes_shifts.iter().sum::<f32>() / n;
    let mean_no_shift = no_shifts.iter().sum::<f32>() / n;

    let std_yes_shift = (yes_shifts
        .iter()
        .map(|x| (x - mean_yes_shift).powi(2))
        .sum::<f32>()
        / n)
        .sqrt();
    let std_no_shift = (no_shifts
        .iter()
        .map(|x| (x - mean_no_shift).powi(2))
        .sum::<f32>()
        / n)
        .sqrt();
    let correct_shifts: Vec<f32> = questions
        .iter()
        .map(|q| q.steered_correct_prob - q.base_correct_prob)
        .collect();
    let (mean_correct_prob_shift, std_correct_prob_shift) = mean_and_std(&correct_shifts);
    let mean_base_correct_prob = questions.iter().map(|q| q.base_correct_prob).sum::<f32>() / n;
    let mean_steered_correct_prob = questions
        .iter()
        .map(|q| q.steered_correct_prob)
        .sum::<f32>()
        / n;
    let base_accuracy = questions.iter().filter(|q| q.base_is_correct).count() as f32 / n;
    let steered_accuracy = questions.iter().filter(|q| q.steered_is_correct).count() as f32 / n;
    let accuracy_shift = steered_accuracy - base_accuracy;

    Ok(ControlQuestionsResult {
        concept: concept.to_string(),
        scale,
        layers: layers.to_vec(),
        questions,
        summary: ControlQuestionsSummary {
            mean_base_p_yes,
            mean_base_p_no,
            mean_steered_p_yes,
            mean_steered_p_no,
            mean_yes_shift,
            mean_no_shift,
            std_yes_shift,
            std_no_shift,
            mean_base_correct_prob,
            mean_steered_correct_prob,
            mean_correct_prob_shift,
            std_correct_prob_shift,
            base_accuracy,
            steered_accuracy,
            accuracy_shift,
        },
    })
}

// ── Logit Lens Comparison Experiment ────────────────────────────────

/// Run comparative logit lens: per-layer probability for tracked tokens,
/// comparing base (no steering) vs steered forward passes.
pub fn run_logit_lens_comparison(
    state: &SharedState,
    concept: &str,
    scale: f64,
    user_turn1_variant: &str,
    layers: &[usize],
    tracked_token_strings: &[String],
) -> anyhow::Result<LogitLensComparisonResult> {
    let conversation = build_detection_conversation(user_turn1_variant, prompts::USER_TURN_2)?;

    // Resolve tracked token IDs
    let tracked_tokens: Vec<(String, u32)> = {
        let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
        tracked_token_strings
            .iter()
            .filter_map(|s| {
                let (ids, _) = model.tokenize(s).ok()?;
                ids.first().map(|&id| (s.clone(), id))
            })
            .collect()
    };

    let steering_vecs = state
        .steering_vectors
        .read()
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let svec = steering_vecs
        .get(concept)
        .ok_or_else(|| anyhow::anyhow!("Steering vector '{}' not found.", concept))?;
    let svec_vectors = svec.vectors.clone();
    drop(steering_vecs);

    let info = &state.model_info;

    // Base forward pass with full hidden state capture
    let (base_layers_data, _) = {
        let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
        with_clean_steering(&model, |m| {
            let result = m.forward_introspect(&conversation)?;
            let lens = m.logit_lens(&result.hidden_states)?;
            build_comparison_layer_data(m, &lens.layer_probs, &tracked_tokens, info)
        })?
    };

    // Steered forward pass
    let (steered_layers_data, _) = {
        let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
        with_clean_steering(&model, |m| {
            apply_steering_vectors(m, &svec_vectors, layers, scale)?;
            let result = m.forward_introspect(&conversation)?;
            let lens = m.logit_lens(&result.hidden_states)?;
            build_comparison_layer_data(m, &lens.layer_probs, &tracked_tokens, info)
        })?
    };

    Ok(LogitLensComparisonResult {
        variant: user_turn1_variant.to_string(),
        concept: concept.to_string(),
        scale,
        tracked_tokens: tracked_tokens.iter().map(|(s, _)| s.clone()).collect(),
        base_layers: base_layers_data,
        steered_layers: steered_layers_data,
    })
}

/// Extract per-layer comparison data from logit lens probability tensors.
fn build_comparison_layer_data(
    model: &mistralrs_core::introspection::IntrospectionModel,
    layer_probs: &[candle_core::Tensor],
    tracked_tokens: &[(String, u32)],
    info: &mistralrs_core::introspection::ModelInfo,
) -> anyhow::Result<(Vec<ComparisonLayerData>, Vec<Vec<f32>>)> {
    let mut layers = Vec::with_capacity(layer_probs.len());
    let mut all_tracked = Vec::new();

    for (layer_idx, probs_tensor) in layer_probs.iter().enumerate() {
        let probs_vec: Vec<f32> = probs_tensor.squeeze(0)?.to_dtype(DType::F32)?.to_vec1()?;

        let layer_type = if layer_idx == 0 {
            "embedding"
        } else if layer_idx % info.full_attention_interval == 0 {
            "full_attention"
        } else {
            "gdn"
        };

        // Track specific token probabilities
        let tracked_probs: Vec<f32> = tracked_tokens
            .iter()
            .map(|(_, id)| probs_vec.get(*id as usize).copied().unwrap_or(0.0))
            .collect();

        // Top-1 token
        let (top1_idx, &top1_prob) = probs_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        all_tracked.push(tracked_probs.clone());

        layers.push(ComparisonLayerData {
            layer_idx,
            layer_type: layer_type.to_string(),
            tracked_probs,
            top1_token: model.decode_token(top1_idx as u32),
            top1_prob,
        });
    }

    Ok((layers, all_tracked))
}

// ══════════════════════════════════════════════════════════════════════
// Hybrid Architecture Experiments (QWEN3-SPEC Steps 3A–3F)
// ══════════════════════════════════════════════════════════════════════

// ── 3F. Logit Lens by Layer Type ────────────────────────────────────

/// Logit lens annotated with layer types (GDN vs full attention).
///
/// No new hooks needed — post-processes existing logit_lens_all() output.
/// Quick win: shows where "understanding jumps" happen in the hybrid stack.
pub fn run_layer_type_lens(state: &SharedState, text: &str) -> anyhow::Result<LayerTypeLensResult> {
    let info = &state.model_info;

    let (_hidden_states, layer_probs) = {
        let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
        let result = model.forward_introspect(text)?;
        let lens = model.logit_lens(&result.hidden_states)?;
        (result.hidden_states, lens.layer_probs)
    };

    let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut layers = Vec::with_capacity(layer_probs.len());
    let mut gdn_confs = Vec::new();
    let mut attn_confs = Vec::new();

    for (layer_idx, probs_tensor) in layer_probs.iter().enumerate() {
        let probs_vec: Vec<f32> = probs_tensor.squeeze(0)?.to_dtype(DType::F32)?.to_vec1()?;

        let layer_type = if layer_idx == 0 {
            "embedding".to_string()
        } else {
            info.layer_types
                .get(layer_idx - 1)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string())
        };

        // Top-1 token
        let (top1_idx, &top1_prob) = probs_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        // Shannon entropy
        let entropy: f32 = probs_vec
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();

        match layer_type.as_str() {
            "gdn" => gdn_confs.push(top1_prob),
            "full_attention" => attn_confs.push(top1_prob),
            _ => {}
        }

        layers.push(LayerTypeLensEntry {
            layer_idx,
            layer_type,
            top1_prob,
            top1_token: model.decode_token(top1_idx as u32),
            entropy,
        });
    }

    let mean_gdn_confidence = if gdn_confs.is_empty() {
        0.0
    } else {
        gdn_confs.iter().sum::<f32>() / gdn_confs.len() as f32
    };
    let mean_attn_confidence = if attn_confs.is_empty() {
        0.0
    } else {
        attn_confs.iter().sum::<f32>() / attn_confs.len() as f32
    };

    // Compute confidence jumps at full attention layers
    let mut attn_jumps = Vec::new();
    for i in 1..layers.len() {
        if layers[i].layer_type == "full_attention" {
            let delta = layers[i].top1_prob - layers[i - 1].top1_prob;
            attn_jumps.push(LayerFloat {
                layer_idx: layers[i].layer_idx,
                value: delta,
            });
        }
    }

    Ok(LayerTypeLensResult {
        layers,
        mean_gdn_confidence,
        mean_attn_confidence,
        attn_jumps,
    })
}

// ── 3D. Steering Survival Through GDN ───────────────────────────────

/// Measure how a steering vector injection at a single layer propagates
/// through subsequent GDN vs attention layers.
///
/// For each injection layer, injects the steering vector at ONLY that layer,
/// then measures concept activation at all subsequent layers.
pub fn run_steering_survival(
    state: &SharedState,
    concept: &str,
    injection_layers: &[usize],
    scale: f64,
    probe_text: &str,
) -> anyhow::Result<SteeringSurvivalResult> {
    let info = &state.model_info;

    // Get the concept vectors
    let svec_vectors = {
        let steering_vecs = state
            .steering_vectors
            .read()
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let svec = steering_vecs
            .get(concept)
            .ok_or_else(|| anyhow::anyhow!("Steering vector '{}' not found.", concept))?;
        svec.vectors.clone()
    };

    let mut normalized_layers = injection_layers.to_vec();
    normalized_layers.sort_unstable();
    normalized_layers.dedup();
    if normalized_layers.is_empty() {
        anyhow::bail!("injection_layers cannot be empty");
    }

    let mut traces = Vec::new();

    for &inj_layer in &normalized_layers {
        if inj_layer == 0 || inj_layer > info.num_layers {
            anyhow::bail!(
                "Invalid injection layer {}. Use introspection layer indices in 1..={}.",
                inj_layer,
                info.num_layers
            );
        }
        let inj_decoder_idx = inj_layer - 1;
        let inj_type = info
            .layer_types
            .get(inj_decoder_idx)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        // Set steering at ONLY this one layer
        let hidden_states = {
            let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
            with_clean_steering(&model, |m| {
                if let Some(direction) = svec_vectors.get(&inj_layer) {
                    let dir_tensor = candle_core::Tensor::new(direction.as_slice(), m.device())?
                        .to_dtype(m.dtype())?;
                    let scaled = (&dir_tensor * scale)?;
                    m.set_steering_vector(inj_decoder_idx, scaled);
                } else {
                    anyhow::bail!(
                        "No trained steering vector for injection layer {}. \
                         Train vectors for this introspection layer first.",
                        inj_layer
                    );
                }
                let result = m.forward_introspect(probe_text)?;
                Ok(result.hidden_states)
            })?
        };

        // Measure concept activation at each layer after injection
        let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
        let mut activations = Vec::new();

        // Layer indices are introspection indices:
        // hidden_states[0] = embedding, hidden_states[1] = decoder L0, ..., [N] = decoder L(N-1)
        for layer_idx in (inj_layer + 1)..=info.num_layers {
            if layer_idx >= hidden_states.len() {
                break;
            }
            let decoder_layer_idx = layer_idx - 1;

            let layer_type = info
                .layer_types
                .get(decoder_layer_idx)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());

            // Compute concept activation: cosine(hidden_state, direction)
            // Direction vectors from contrastive PCA are unit-normalized,
            // so cosine = dot / hs_norm.
            let activation = if let Some(direction) = svec_vectors.get(&layer_idx) {
                let hs = &hidden_states[layer_idx];
                let seq_len = hs.dim(1)?;
                let last = hs.i((0, seq_len - 1))?.to_dtype(DType::F32)?;
                let dir = candle_core::Tensor::new(direction.as_slice(), model.device())?
                    .to_dtype(DType::F32)?;
                let dot: f32 = (&last * &dir)?.sum_all()?.to_scalar()?;
                let hs_norm: f32 = last.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                if hs_norm > 0.0 {
                    dot / hs_norm
                } else {
                    0.0
                }
            } else {
                tracing::warn!(
                    "No direction vector for layer {} — steering survival activation \
                     will be NaN. Train vectors for all layers.",
                    layer_idx
                );
                f32::NAN
            };

            activations.push(SurvivalActivation {
                layer_idx,
                layer_type,
                activation,
            });
        }

        traces.push(SteeringSurvivalTrace {
            injection_layer: inj_layer,
            injection_layer_type: inj_type,
            activations,
        });
    }

    Ok(SteeringSurvivalResult { traces })
}

// ── 3A. CKA Cross-Layer Similarity ──────────────────────────────────

/// Linear CKA (Centered Kernel Alignment) between all pairs of layers.
///
/// For each text, extracts the last-token hidden state at each layer,
/// builds per-layer matrices (n_texts × hidden_size), then computes
/// linear CKA: ||Y^T X||_F^2 / (||X^T X||_F · ||Y^T Y||_F)
pub fn run_cka(state: &SharedState, texts: &[String]) -> anyhow::Result<CkaResult> {
    let info = &state.model_info;
    let num_capture = info.num_layers + 1; // +1 for embedding

    // Collect last-token hidden states: per_layer[layer_idx] = Vec of (hidden_size,) tensors
    let mut per_layer: Vec<Vec<Vec<f32>>> = vec![Vec::new(); num_capture];

    for text in texts {
        let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
        let result = model.forward_introspect(text)?;

        for (layer_idx, hs) in result.hidden_states.iter().enumerate() {
            if layer_idx >= num_capture {
                break;
            }
            let seq_len = hs.dim(1)?;
            let last: Vec<f32> = hs.i((0, seq_len - 1))?.to_dtype(DType::F32)?.to_vec1()?;
            per_layer[layer_idx].push(last);
        }
    }

    // Build layer labels
    let mut layer_labels = Vec::with_capacity(num_capture);
    layer_labels.push("emb".to_string());
    for i in 0..info.num_layers {
        let lt = info
            .layer_types
            .get(i)
            .map(|s| if s == "full_attention" { "attn" } else { &s })
            .unwrap_or("?");
        layer_labels.push(format!("L{} ({})", i, lt));
    }

    // Compute Gram matrices (X^T X for centered X) and CKA
    let n = texts.len();
    let centered_grams: Vec<Vec<f64>> = per_layer
        .iter()
        .map(|vecs| centered_gram(vecs, n))
        .collect();

    let gram_norms: Vec<f64> = centered_grams
        .iter()
        .map(|g| frobenius_norm_sq(g, n))
        .collect();

    let mut similarity_matrix = vec![vec![0.0f32; num_capture]; num_capture];

    for i in 0..num_capture {
        for j in i..num_capture {
            let cross = cross_frobenius(&centered_grams[i], &centered_grams[j], n);
            let denom = (gram_norms[i] * gram_norms[j]).sqrt();
            let cka = if denom > 0.0 {
                (cross / denom) as f32
            } else {
                0.0
            };
            similarity_matrix[i][j] = cka;
            similarity_matrix[j][i] = cka;
        }
    }

    // CKA between consecutive layers (off-diagonal of similarity matrix)
    let mut consecutive_cka = Vec::with_capacity(num_capture - 1);
    for i in 0..num_capture - 1 {
        consecutive_cka.push(similarity_matrix[i][i + 1]);
    }

    Ok(CkaResult {
        similarity_matrix,
        layer_labels,
        consecutive_cka,
    })
}

/// Compute centered Gram matrix K = X X^T where X is centered (mean-subtracted).
/// Input: vecs[sample][feature], n = number of samples.
/// Returns flattened n×n Gram matrix.
fn centered_gram(vecs: &[Vec<f32>], n: usize) -> Vec<f64> {
    if vecs.is_empty() || n == 0 {
        return vec![0.0; n * n];
    }
    let d = vecs[0].len();
    // Compute mean
    let mut mean = vec![0.0f64; d];
    for v in vecs {
        for (j, &val) in v.iter().enumerate() {
            mean[j] += val as f64;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    // Centered Gram: K[i][j] = sum_d (x_i[d] - mean[d]) * (x_j[d] - mean[d])
    let mut gram = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let mut dot = 0.0f64;
            for k in 0..d {
                let a = vecs[i][k] as f64 - mean[k];
                let b = vecs[j][k] as f64 - mean[k];
                dot += a * b;
            }
            gram[i * n + j] = dot;
            gram[j * n + i] = dot;
        }
    }
    gram
}

/// Frobenius norm squared of an n×n matrix (flattened).
fn frobenius_norm_sq(g: &[f64], n: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            let v = g[i * n + j];
            sum += v * v;
        }
    }
    sum
}

/// Cross-Frobenius: tr(A^T B) = sum_ij A[i][j]*B[i][j]
fn cross_frobenius(a: &[f64], b: &[f64], n: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..n * n {
        sum += a[i] * b[i];
    }
    sum
}

// ── 3B. MoE Routing Analysis ────────────────────────────────────────

/// Analyze MoE routing patterns: entropy, expert frequency, shared gate values.
/// If a concept is specified, also computes routing divergence under steering.
pub fn run_routing_analysis(
    state: &SharedState,
    text: &str,
    concept: Option<&str>,
    scale: f64,
    layers: &[usize],
) -> anyhow::Result<RoutingAnalysisResult> {
    use crate::steering::apply_steering_vectors;

    // Base forward with routing capture
    let base_routing = {
        let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
        with_clean_steering(&model, |m| {
            let (_logits, _hs, routing) = m.forward_introspect_with_routing(text)?;
            Ok(routing)
        })?
    };

    // Steered forward (optional)
    let steered_routing = if let Some(concept_name) = concept {
        let svec_vectors = {
            let steering_vecs = state
                .steering_vectors
                .read()
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let svec = steering_vecs
                .get(concept_name)
                .ok_or_else(|| anyhow::anyhow!("Steering vector '{}' not found.", concept_name))?;
            svec.vectors.clone()
        };
        let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
        let routing = with_clean_steering(&model, |m| {
            apply_steering_vectors(m, &svec_vectors, layers, scale)?;
            let (_logits, _hs, routing) = m.forward_introspect_with_routing(text)?;
            Ok(routing)
        })?;
        Some(routing)
    } else {
        None
    };

    let info = &state.model_info;

    // Compute per-layer metrics from base routing
    let mut layer_entropy = Vec::new();
    let mut expert_frequency = Vec::new();
    let mut shared_gate_values = Vec::new();

    for (i, rd) in base_routing.iter().enumerate() {
        let layer_idx = i; // routing_data is collected in layer order

        // Entropy of softmaxed router logits
        let router_probs: Vec<f32> =
            candle_nn::ops::softmax_last_dim(&rd.router_logits.to_dtype(DType::F32)?)?
                .mean(0)? // average across tokens
                .to_vec1()?;

        let entropy: f32 = router_probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();

        layer_entropy.push(LayerFloat {
            layer_idx,
            value: entropy,
        });

        // Expert selection frequency
        let topk: Vec<u32> = rd.topk_indices.flatten_all()?.to_vec1()?;
        let num_experts = info.num_experts;
        let mut freq = vec![0.0f32; num_experts];
        let total = topk.len() as f32;
        for &idx in &topk {
            if (idx as usize) < num_experts {
                freq[idx as usize] += 1.0 / total;
            }
        }
        expert_frequency.push(LayerFrequency {
            layer_idx,
            frequencies: freq,
        });

        // Shared gate mean
        let gate_mean: f32 = rd
            .shared_gate
            .to_dtype(DType::F32)?
            .mean_all()?
            .to_scalar()?;
        shared_gate_values.push(LayerFloat {
            layer_idx,
            value: gate_mean,
        });
    }

    // Compute routing divergence if we have steered data
    let routing_divergence = if let Some(steered) = steered_routing.as_ref() {
        let mut divergences = Vec::new();
        for (i, (base_rd, steer_rd)) in base_routing.iter().zip(steered.iter()).enumerate() {
            // KL(steered || base) on mean routing weights
            let base_probs: Vec<f32> =
                candle_nn::ops::softmax_last_dim(&base_rd.router_logits.to_dtype(DType::F32)?)?
                    .mean(0)?
                    .to_vec1()?;

            let steer_probs: Vec<f32> =
                candle_nn::ops::softmax_last_dim(&steer_rd.router_logits.to_dtype(DType::F32)?)?
                    .mean(0)?
                    .to_vec1()?;

            let kl: f32 = steer_probs
                .iter()
                .zip(base_probs.iter())
                .filter(|(q, p)| **q > 1e-10 && **p > 1e-10)
                .map(|(q, p)| *q * (*q / *p).ln())
                .sum();

            divergences.push(LayerFloat {
                layer_idx: i,
                value: kl,
            });
        }
        Some(divergences)
    } else {
        None
    };

    let shared_gate_delta = if let Some(steered) = steered_routing.as_ref() {
        let mut deltas = Vec::new();
        for (i, (base_rd, steer_rd)) in base_routing.iter().zip(steered.iter()).enumerate() {
            let base_gate: f32 = base_rd
                .shared_gate
                .to_dtype(DType::F32)?
                .mean_all()?
                .to_scalar()?;
            let steer_gate: f32 = steer_rd
                .shared_gate
                .to_dtype(DType::F32)?
                .mean_all()?
                .to_scalar()?;
            deltas.push(LayerFloat {
                layer_idx: i,
                value: steer_gate - base_gate,
            });
        }
        Some(deltas)
    } else {
        None
    };

    Ok(RoutingAnalysisResult {
        layer_entropy,
        expert_frequency,
        routing_divergence,
        shared_gate_values,
        shared_gate_delta,
    })
}

// ── 3C. Causal Tracing ──────────────────────────────────────────────

/// Causal tracing: patch clean hidden states into corrupted runs one layer
/// at a time to identify which layers are critical for the prediction.
///
/// NOTE: This is whole-layer patching — the entire hidden state tensor
/// (batch, seq_len, hidden_dim) is replaced at each layer. This is coarser
/// than per-token causal tracing (Meng et al. 2022) which patches at specific
/// (layer, token) intersections. Whole-layer patching identifies critical
/// layers but not critical token positions within those layers.
///
/// IMPORTANT: clean_text and corrupted_text must tokenize to the same length,
/// since patching replaces the full hidden state tensor.
pub fn run_causal_tracing(
    state: &SharedState,
    clean_text: &str,
    corrupted_text: &str,
) -> anyhow::Result<CausalTracingResult> {
    let info = &state.model_info;

    // Validate: both texts must tokenize to same length
    {
        let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
        let (clean_ids, _) = model.tokenize(clean_text)?;
        let (corrupt_ids, _) = model.tokenize(corrupted_text)?;
        if clean_ids.len() != corrupt_ids.len() {
            anyhow::bail!(
                "Token count mismatch: clean='{}' has {} tokens, corrupted='{}' has {} tokens. \
                 Both must have the same number of tokens for whole-layer patching.",
                clean_text,
                clean_ids.len(),
                corrupted_text,
                corrupt_ids.len()
            );
        }
    }

    // Clean forward: capture all hidden states, record top-1 probability
    let (clean_top_token, clean_top_id, clean_prob, clean_hidden) = {
        let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
        model.clear_steering_vectors();
        model.clear_patches();
        let result = model.forward_introspect(clean_text)?;
        let probs = last_token_probs(&result.logits)?;
        let (top_id, &top_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        let top_token = model.decode_token(top_id as u32);
        (top_token, top_id as u32, top_prob, result.hidden_states)
    };

    // Corrupted forward: measure P(clean_top_token)
    let (corrupted_top_token, corrupted_prob) = {
        let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
        let result = model.forward_introspect(corrupted_text)?;
        let probs = last_token_probs(&result.logits)?;
        let corr_prob = probs.get(clean_top_id as usize).copied().unwrap_or(0.0);
        let (corr_top_id, _) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        let corr_top_token = model.decode_token(corr_top_id as u32);
        (corr_top_token, corr_prob)
    };

    // For each layer: patch clean hidden state, run corrupted, measure recovery
    let mut layer_recovery = Vec::new();
    let recovery_range = clean_prob - corrupted_prob;

    for layer_idx in 0..info.num_layers {
        let capture_idx = layer_idx + 1; // hidden_states[0] = embedding
        if capture_idx >= clean_hidden.len() {
            break;
        }

        let layer_type = info
            .layer_types
            .get(layer_idx)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        let patched_prob = {
            let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
            model.clear_patches();
            model.set_patch(layer_idx, clean_hidden[capture_idx].clone());
            let result = model.forward_introspect(corrupted_text)?;
            model.clear_patches();
            let probs = last_token_probs(&result.logits)?;
            probs.get(clean_top_id as usize).copied().unwrap_or(0.0)
        };

        let recovery = if recovery_range.abs() > 1e-10 {
            (patched_prob - corrupted_prob) / recovery_range
        } else {
            0.0
        };

        layer_recovery.push(CausalTracingLayer {
            layer_idx,
            layer_type,
            recovery,
        });

        tracing::info!(
            "  Causal trace layer {}/{}: recovery={:.3}",
            layer_idx,
            info.num_layers,
            recovery
        );
    }

    Ok(CausalTracingResult {
        layer_recovery,
        clean_top_token,
        corrupted_top_token,
        clean_prob,
        corrupted_prob,
    })
}

// ── 3E. GDN Recurrent State Stats ───────────────────────────────────

/// Compute statistics about GDN recurrent states after a forward pass:
/// Frobenius norm, effective rank (exp(spectral entropy)), top singular value.
pub fn run_gdn_state_stats(state: &SharedState, text: &str) -> anyhow::Result<GdnStateStatsResult> {
    // Forward pass to populate GDN caches
    {
        let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
        model.clear_steering_vectors();
        model.clear_patches();
        let _ = model.forward_introspect(text)?;
    }

    // Extract recurrent states
    let gdn_states = {
        let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
        model.gdn_recurrent_states()?
    };

    let mut layers = Vec::new();

    for (layer_idx, state_tensor) in gdn_states {
        // state shape: (1, num_v_heads, head_k_dim, head_v_dim)
        // Flatten to 2D: (num_v_heads, head_k_dim * head_v_dim) for SVD-like analysis
        let dims = state_tensor.dims();
        let state_2d = if dims.len() == 4 {
            let (_, n_heads, k_dim, v_dim) = (dims[0], dims[1], dims[2], dims[3]);
            state_tensor
                .squeeze(0)?
                .reshape((n_heads, k_dim * v_dim))?
                .to_dtype(DType::F32)?
        } else {
            state_tensor
                .flatten_all()?
                .unsqueeze(0)?
                .to_dtype(DType::F32)?
        };

        // Frobenius norm
        let frob_norm: f32 = state_2d.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

        // Compute singular values via eigenvalues of S^T S
        // S^T S has shape (k*v, k*v) — but that could be huge.
        // Instead compute S S^T which is (n_heads, n_heads) — much smaller.
        let gram = state_2d.matmul(&state_2d.t()?)?; // (n_heads, n_heads)
        let eigenvalues = symmetric_eigenvalues(&gram)?;

        // Singular values = sqrt(eigenvalues of S S^T)
        let singular_values: Vec<f32> = eigenvalues
            .iter()
            .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
            .collect();

        let top_sv = singular_values.first().copied().unwrap_or(0.0);

        // Spectral entropy and effective rank
        let sv_sum: f32 = singular_values.iter().sum();
        let spectral_entropy = if sv_sum > 0.0 {
            let normalized: Vec<f32> = singular_values.iter().map(|&s| s / sv_sum).collect();
            -normalized
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p * p.ln())
                .sum::<f32>()
        } else {
            0.0
        };

        let effective_rank = spectral_entropy.exp();

        layers.push(GdnStateStats {
            layer_idx,
            frobenius_norm: frob_norm,
            effective_rank,
            top_singular_value: top_sv,
            spectral_entropy,
        });
    }

    Ok(GdnStateStatsResult { layers })
}

// ══════════════════════════════════════════════════════════════════════
// Code-Native Introspection Detection Experiments
// ══════════════════════════════════════════════════════════════════════

/// Build a code-native detection conversation as ChatML text.
///
/// Same injection mechanism as the vgel protocol (user turn 1 + "{ }" prefill),
/// but with a code-oriented system prompt, question, and measurement prefix.
fn build_code_detection_conversation(
    system_prompt: &str,
    user_turn1_variant: &str,
    code_question: &str,
    assistant_prefix: &str,
) -> anyhow::Result<String> {
    let turn1_text = prompts::user_turn1_by_variant(user_turn1_variant).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown turn1 variant '{}'. Use: no_info, with_info, inaccurate_info",
            user_turn1_variant
        )
    })?;

    Ok(format_chatml(
        &[
            ChatMessage {
                role: "system",
                content: system_prompt,
            },
            ChatMessage {
                role: "user",
                content: turn1_text,
            },
            ChatMessage {
                role: "assistant",
                content: prompts::ASST_TURN_1,
            },
            ChatMessage {
                role: "user",
                content: code_question,
            },
            ChatMessage {
                role: "assistant",
                content: assistant_prefix,
            },
        ],
        true,  // continue_final — leave prefix open for measurement
        false, // no generation prompt
    ))
}

/// Extract P(True) and P(False) from a pre-computed probability vector using code candidates.
fn true_false_probs_from_last_token(
    model: &IntrospectionModel,
    probs_vec: &[f32],
    true_candidates: &[&str],
    false_candidates: &[&str],
) -> anyhow::Result<(f32, f32)> {
    let p_true = completion_candidates_prob(model, probs_vec, true_candidates)?;
    let p_false = completion_candidates_prob(model, probs_vec, false_candidates)?;
    Ok((p_true, p_false))
}

/// Parse a generated text for a boolean result.
///
/// Looks for True/False/true/false/1/0/None at the start of the generated text.
fn parse_boolean_generation(text: &str) -> Option<bool> {
    let trimmed = text.trim();
    let first_word = trimmed
        .split(|c: char| c.is_whitespace() || c == ',' || c == '}' || c == '\n' || c == ';')
        .next()
        .unwrap_or("");
    match first_word {
        "True" | "true" | "1" => Some(true),
        "False" | "false" | "0" | "None" => Some(false),
        _ => None,
    }
}

/// Parse a generated text for a concept name (looks for quoted string literals).
fn parse_concept_name(text: &str) -> Option<String> {
    // Try double quotes first, then single quotes
    for quote in ['"', '\''] {
        if let Some(start) = text.find(quote) {
            if let Some(end) = text[start + 1..].find(quote) {
                let name = text[start + 1..start + 1 + end].trim().to_lowercase();
                if !name.is_empty() {
                    return Some(name);
                }
            }
        }
    }
    // Fallback: take first word if it looks like a concept name (alphabetic)
    let trimmed = text.trim().to_lowercase();
    let first_word = trimmed
        .split(|c: char| !c.is_alphabetic())
        .next()
        .unwrap_or("");
    if first_word.len() >= 3 {
        Some(first_word.to_string())
    } else {
        None
    }
}

/// Score how well a parsed concept matches the expected concept.
fn score_concept_match(parsed: &str, expected: &str) -> &'static str {
    let parsed_lower = parsed.to_lowercase();
    let expected_lower = expected.to_lowercase();

    if parsed_lower == expected_lower {
        return "exact";
    }

    let synonyms = prompts::concept_synonyms(expected);
    for &syn in synonyms {
        if parsed_lower == syn || parsed_lower.contains(syn) || syn.contains(&*parsed_lower) {
            return "synonym";
        }
    }

    "miss"
}

// ── Phase 1: Code Logit Diff ────────────────────────────────────────

/// Run code-native logit diff: measures P(True)+P(False) at code completion
/// positions instead of P(yes)+P(no) at "The answer is" position.
///
/// Tests 3 templates × 3 turn1 variants = 9 configurations.
/// Each: base + steered + random = 27 forward passes (~14 min).
pub fn run_code_logit_diff(
    state: &SharedState,
    concept: &str,
    scale: f64,
    layers: &[usize],
    top_k: usize,
) -> anyhow::Result<CodeLogitDiffResult> {
    let svec_vectors = {
        let steering_vecs = state
            .steering_vectors
            .read()
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let svec = steering_vecs.get(concept).ok_or_else(|| {
            anyhow::anyhow!("Steering vector '{}' not found. Train one first.", concept)
        })?;
        svec.vectors.clone()
    };
    let random_vectors = random_control_vectors(concept, layers, &svec_vectors);

    let variants = [
        prompts::TURN1_VARIANT_NO_INFO,
        prompts::TURN1_VARIANT_WITH_INFO,
        prompts::TURN1_VARIANT_INACCURATE,
    ];

    let mut trials = Vec::new();
    let mut first_base_probs: Option<Vec<f32>> = None;
    let mut first_steered_probs: Option<Vec<f32>> = None;

    for template in prompts::CODE_TEMPLATES {
        for &variant in &variants {
            let conversation = build_code_detection_conversation(
                template.system_prompt,
                variant,
                template.question,
                template.assistant_prefix,
            )?;

            tracing::info!(
                "Code logit diff: template={}, variant={}",
                template.name,
                variant
            );

            // Base forward pass
            let (base_probs, base_p_true, base_p_false) = {
                let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
                with_clean_steering(&model, |m| {
                    let result = m.forward_introspect(&conversation)?;
                    let probs = last_token_probs(&result.logits)?;
                    let (pt, pf) = true_false_probs_from_last_token(
                        m,
                        &probs,
                        template.true_candidates,
                        template.false_candidates,
                    )?;
                    Ok((probs, pt, pf))
                })?
            };

            // Steered forward pass
            let (steered_probs, steered_p_true, steered_p_false) = {
                let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
                with_clean_steering(&model, |m| {
                    apply_steering_vectors(m, &svec_vectors, layers, scale)?;
                    let result = m.forward_introspect(&conversation)?;
                    let probs = last_token_probs(&result.logits)?;
                    let (pt, pf) = true_false_probs_from_last_token(
                        m,
                        &probs,
                        template.true_candidates,
                        template.false_candidates,
                    )?;
                    Ok((probs, pt, pf))
                })?
            };

            // Random-direction control
            let (random_p_true, random_p_false) = {
                let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
                with_clean_steering(&model, |m| {
                    apply_steering_vectors(m, &random_vectors, layers, scale)?;
                    let result = m.forward_introspect(&conversation)?;
                    let probs = last_token_probs(&result.logits)?;
                    true_false_probs_from_last_token(
                        m,
                        &probs,
                        template.true_candidates,
                        template.false_candidates,
                    )
                })?
            };

            if first_base_probs.is_none() {
                first_base_probs = Some(base_probs);
                first_steered_probs = Some(steered_probs);
            }

            trials.push(CodeLogitDiffTrial {
                template: template.name.to_string(),
                variant: variant.to_string(),
                base_p_true,
                base_p_false,
                base_coverage: base_p_true + base_p_false,
                steered_p_true,
                steered_p_false,
                steered_coverage: steered_p_true + steered_p_false,
                random_p_true,
                random_p_false,
                random_coverage: random_p_true + random_p_false,
            });

            tracing::info!(
                "  Coverage: base={:.4}%, steered={:.4}%, random={:.4}%",
                (base_p_true + base_p_false) * 100.0,
                (steered_p_true + steered_p_false) * 100.0,
                (random_p_true + random_p_false) * 100.0,
            );
        }
    }

    let n = trials.len() as f32;
    let mean_base_coverage = trials.iter().map(|t| t.base_coverage).sum::<f32>() / n;
    let mean_steered_coverage = trials.iter().map(|t| t.steered_coverage).sum::<f32>() / n;
    let mean_random_coverage = trials.iter().map(|t| t.random_coverage).sum::<f32>() / n;
    let base_p_true = trials.iter().map(|t| t.base_p_true).sum::<f32>() / n;
    let steered_p_true = trials.iter().map(|t| t.steered_p_true).sum::<f32>() / n;
    let random_p_true = trials.iter().map(|t| t.random_p_true).sum::<f32>() / n;
    let base_p_false = trials.iter().map(|t| t.base_p_false).sum::<f32>() / n;
    let steered_p_false = trials.iter().map(|t| t.steered_p_false).sum::<f32>() / n;
    let random_p_false = trials.iter().map(|t| t.random_p_false).sum::<f32>() / n;

    let true_shifts: Vec<f32> = trials
        .iter()
        .map(|t| t.steered_p_true - t.base_p_true)
        .collect();
    let (mean_true_shift, std_true_shift) = mean_and_std(&true_shifts);

    // Top-k tokens from first trial
    let base_probs = first_base_probs
        .ok_or_else(|| anyhow::anyhow!("No code logit-diff trials were executed"))?;
    let steered_probs = first_steered_probs
        .ok_or_else(|| anyhow::anyhow!("No code logit-diff trials were executed"))?;

    let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
    let mut indexed: Vec<(usize, f32)> = steered_probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_tokens: Vec<LogitDiffToken> = indexed
        .iter()
        .take(top_k)
        .map(|&(id, steered_p)| {
            let base_p = base_probs.get(id).copied().unwrap_or(0.0);
            LogitDiffToken {
                token: model.decode_token(id as u32),
                token_id: id as u32,
                base_prob: base_p,
                steered_prob: steered_p,
                diff: steered_p - base_p,
            }
        })
        .collect();

    Ok(CodeLogitDiffResult {
        concept: concept.to_string(),
        scale,
        layers: layers.to_vec(),
        trials,
        mean_base_coverage,
        mean_steered_coverage,
        mean_random_coverage,
        base_p_true,
        steered_p_true,
        random_p_true,
        base_p_false,
        steered_p_false,
        random_p_false,
        mean_true_shift,
        std_true_shift,
        top_tokens,
    })
}

// ── Phase 2: Code Gen Detection ─────────────────────────────────────

/// Run code-native generation detection: generates short code under
/// base/steered/random conditions, parses output for True/False.
///
/// Tests 3 templates × 3 conditions × (1 greedy + N sampled) trials.
pub fn run_code_gen_detection(
    state: &SharedState,
    concept: &str,
    scale: f64,
    layers: &[usize],
    max_tokens: usize,
    temperatures: &[f64],
) -> anyhow::Result<CodeGenDetectionResult> {
    let svec_vectors = {
        let steering_vecs = state
            .steering_vectors
            .read()
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let svec = steering_vecs.get(concept).ok_or_else(|| {
            anyhow::anyhow!("Steering vector '{}' not found. Train one first.", concept)
        })?;
        svec.vectors.clone()
    };
    let random_vectors = random_control_vectors(concept, layers, &svec_vectors);

    let mut conditions = Vec::new();

    for template in prompts::CODE_TEMPLATES {
        // Build conversation for generation (using with_info variant)
        let conversation = build_code_detection_conversation(
            template.system_prompt,
            prompts::TURN1_VARIANT_WITH_INFO,
            template.question,
            template.assistant_prefix,
        )?;

        for &temperature in temperatures {
            let num_trials = if temperature == 0.0 { 1 } else { 5 };

            for (condition_name, vectors) in [
                ("base", None),
                ("steered", Some(&svec_vectors)),
                ("random", Some(&random_vectors)),
            ] {
                tracing::info!(
                    "Code gen: template={}, condition={}, temp={}, trials={}",
                    template.name,
                    condition_name,
                    temperature,
                    num_trials,
                );

                let mut generations = Vec::new();
                let mut true_count = 0usize;
                let mut parseable_count = 0usize;

                for trial_idx in 0..num_trials {
                    let gen_result = {
                        let model =
                            state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
                        let run = (|| -> anyhow::Result<_> {
                            model.clear_steering_vectors();
                            if let Some(vecs) = vectors {
                                apply_steering_vectors(&model, vecs, layers, scale)?;
                            }
                            let top_p = if temperature > 0.0 { Some(0.9) } else { None };
                            model.generate(&conversation, max_tokens, temperature, top_p)
                        })();
                        model.clear_steering_vectors();
                        run.map_err(|e| anyhow::anyhow!("{}", e))?
                    };

                    let parsed = parse_boolean_generation(&gen_result.text);
                    if let Some(val) = parsed {
                        parseable_count += 1;
                        if val {
                            true_count += 1;
                        }
                    }

                    tracing::info!(
                        "  Trial {}/{}: {:?} -> {:?}",
                        trial_idx + 1,
                        num_trials,
                        gen_result.text.chars().take(80).collect::<String>(),
                        parsed,
                    );

                    generations.push(CodeGenTrial {
                        generated_text: gen_result.text,
                        parsed_result: parsed,
                        num_tokens: gen_result.token_ids.len(),
                    });
                }

                let detection_rate = if parseable_count > 0 {
                    true_count as f32 / parseable_count as f32
                } else {
                    0.0
                };

                conditions.push(CodeGenCondition {
                    template: template.name.to_string(),
                    condition: condition_name.to_string(),
                    temperature,
                    num_trials,
                    generations,
                    detection_rate,
                });
            }
        }
    }

    Ok(CodeGenDetectionResult {
        concept: concept.to_string(),
        scale,
        layers: layers.to_vec(),
        conditions,
    })
}

// ── Phase 3: Concept Identification ─────────────────────────────────

/// Run concept identification: generates `concept = "..."` under steering
/// for each trained concept, parses string literal, scores against synonym map.
pub fn run_concept_identification(
    state: &SharedState,
    concepts: &[String],
    scale: f64,
    layers: &[usize],
) -> anyhow::Result<ConceptIdentificationResult> {
    let mut identifications = Vec::new();

    for concept in concepts {
        let svec_vectors = {
            let steering_vecs = state
                .steering_vectors
                .read()
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let svec = steering_vecs.get(concept.as_str()).ok_or_else(|| {
                anyhow::anyhow!("Steering vector '{}' not found. Train one first.", concept)
            })?;
            svec.vectors.clone()
        };

        // Build identification conversation
        let conversation = build_code_detection_conversation(
            "You are a Python runtime. Execute code mentally and return only the result.",
            prompts::TURN1_VARIANT_WITH_INFO,
            "What concept was injected into your previous message? Assign a string.",
            "concept = \"",
        )?;

        tracing::info!("Concept identification: injecting '{}'", concept);

        let gen_result = {
            let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
            let run = (|| -> anyhow::Result<_> {
                model.clear_steering_vectors();
                apply_steering_vectors(&model, &svec_vectors, layers, scale)?;
                model.generate(&conversation, 16, 0.0, None)
            })();
            model.clear_steering_vectors();
            run.map_err(|e| anyhow::anyhow!("{}", e))?
        };

        // The generated text comes after `concept = "`, so prepend the quote for parsing
        let full_text = format!("\"{}", gen_result.text);
        let parsed = parse_concept_name(&full_text);
        let match_type = match &parsed {
            Some(name) => score_concept_match(name, concept).to_string(),
            None => "unparseable".to_string(),
        };

        tracing::info!(
            "  Generated: {:?} -> parsed: {:?} ({})",
            gen_result.text.chars().take(60).collect::<String>(),
            parsed,
            match_type,
        );

        identifications.push(ConceptIdEntry {
            injected_concept: concept.clone(),
            generated_text: gen_result.text,
            parsed_concept: parsed,
            match_type,
        });
    }

    Ok(ConceptIdentificationResult {
        concepts: concepts.to_vec(),
        scale,
        layers: layers.to_vec(),
        identifications,
    })
}

// ── Phase 4: Discrimination Matrix ──────────────────────────────────

/// Run discrimination matrix: for each concept pair (inject X, identify as Y),
/// builds an N×N confusion matrix with per-concept accuracy.
///
/// Uses multiple trials per concept for statistical robustness.
pub fn run_discrimination_matrix(
    state: &SharedState,
    concepts: &[String],
    scale: f64,
    layers: &[usize],
) -> anyhow::Result<DiscriminationMatrixResult> {
    let n = concepts.len();
    let num_trials = 5;
    // matrix[i][j] = count of times injecting concept i identified concept j
    let mut counts: Vec<Vec<f32>> = vec![vec![0.0; n]; n];

    for (i, inject_concept) in concepts.iter().enumerate() {
        let svec_vectors = {
            let steering_vecs = state
                .steering_vectors
                .read()
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let svec = steering_vecs.get(inject_concept.as_str()).ok_or_else(|| {
                anyhow::anyhow!(
                    "Steering vector '{}' not found. Train one first.",
                    inject_concept
                )
            })?;
            svec.vectors.clone()
        };

        let conversation = build_code_detection_conversation(
            "You are a Python runtime. Execute code mentally and return only the result.",
            prompts::TURN1_VARIANT_WITH_INFO,
            "What concept was injected into your previous message? Assign a string.",
            "concept = \"",
        )?;

        tracing::info!(
            "Discrimination matrix: injecting '{}' ({}/{})",
            inject_concept,
            i + 1,
            n
        );

        for trial in 0..num_trials {
            let gen_result = {
                let model = state.model.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
                let run = (|| -> anyhow::Result<_> {
                    model.clear_steering_vectors();
                    apply_steering_vectors(&model, &svec_vectors, layers, scale)?;
                    let temp = if trial == 0 { 0.0 } else { 0.7 };
                    let top_p = if trial == 0 { None } else { Some(0.9) };
                    model.generate(&conversation, 16, temp, top_p)
                })();
                model.clear_steering_vectors();
                run.map_err(|e| anyhow::anyhow!("{}", e))?
            };

            let full_text = format!("\"{}", gen_result.text);
            if let Some(parsed) = parse_concept_name(&full_text) {
                // Find which concept (if any) this matches
                for (j, check_concept) in concepts.iter().enumerate() {
                    let match_type = score_concept_match(&parsed, check_concept);
                    if match_type == "exact" || match_type == "synonym" {
                        counts[i][j] += 1.0;
                        break;
                    }
                }
            }
        }
    }

    // Normalize to fractions
    let matrix: Vec<Vec<f32>> = counts
        .iter()
        .map(|row| {
            let total: f32 = row.iter().sum();
            if total > 0.0 {
                row.iter().map(|&c| c / total).collect()
            } else {
                row.clone()
            }
        })
        .collect();

    let per_concept_accuracy: Vec<f32> = (0..n).map(|i| matrix[i][i]).collect();
    let overall_accuracy =
        per_concept_accuracy.iter().sum::<f32>() / per_concept_accuracy.len().max(1) as f32;

    Ok(DiscriminationMatrixResult {
        concepts: concepts.to_vec(),
        scale,
        layers: layers.to_vec(),
        matrix,
        per_concept_accuracy,
        overall_accuracy,
    })
}

/// Compute eigenvalues of a small symmetric matrix via power iteration.
/// Returns eigenvalues sorted descending.
fn symmetric_eigenvalues(gram: &candle_core::Tensor) -> anyhow::Result<Vec<f32>> {
    let n = gram.dim(0)?;
    let gram_vec: Vec<Vec<f32>> = (0..n)
        .map(|i| -> anyhow::Result<Vec<f32>> { Ok(gram.i(i)?.to_vec1::<f32>()?) })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut eigenvalues = Vec::new();
    let mut deflated = gram_vec.clone();

    for _ in 0..n.min(32) {
        // Power iteration for top eigenvalue
        let mut v = vec![1.0f32; n];
        let norm = (n as f32).sqrt();
        for x in &mut v {
            *x /= norm;
        }

        for _ in 0..100 {
            // w = A * v
            let mut w = vec![0.0f32; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += deflated[i][j] * v[j];
                }
            }
            let w_norm: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt();
            if w_norm < 1e-12 {
                break;
            }
            for x in &mut w {
                *x /= w_norm;
            }
            v = w;
        }

        // eigenvalue = v^T A v
        let mut ev = 0.0f32;
        for i in 0..n {
            let mut row_dot = 0.0f32;
            for j in 0..n {
                row_dot += deflated[i][j] * v[j];
            }
            ev += v[i] * row_dot;
        }

        if ev.abs() < 1e-10 {
            break;
        }

        eigenvalues.push(ev);

        // Deflate: A = A - ev * v v^T
        for i in 0..n {
            for j in 0..n {
                deflated[i][j] -= ev * v[i] * v[j];
            }
        }
    }

    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(eigenvalues)
}

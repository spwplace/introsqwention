use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

use mistralrs_core::introspection::{IntrospectionModel, ModelInfo};
use serde::{Deserialize, Serialize};

/// Shared state between API handlers and the dashboard.
pub struct SharedState {
    pub model: Mutex<IntrospectionModel>,
    pub model_info: ModelInfo,
    pub experiments: RwLock<HashMap<String, Experiment>>,
    pub steering_vectors: RwLock<HashMap<String, SteeringVectorSet>>,
    pub db_path: std::path::PathBuf,
}

// ── Steering Vector Storage ─────────────────────────────────────────

/// A trained set of per-layer steering direction vectors.
#[derive(Clone, Serialize, Deserialize)]
pub struct SteeringVectorSet {
    pub name: String,
    pub concept: String,
    /// Per-layer direction vectors: layer_idx -> unit direction vector (hidden_size floats).
    pub vectors: HashMap<usize, Vec<f32>>,
    pub num_training_pairs: usize,
    pub created_at: String,
}

// ── Experiment Data Types ───────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub status: String,
    pub created_at: String,
    pub config: ExperimentConfig,
    pub results: Option<ExperimentResults>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub experiment_type: String,
    pub prompt: String,
    pub steering_layers: Option<Vec<usize>>,
    pub steering_scale: Option<f64>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ExperimentResults {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_lens: Option<LogitLensData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_diff: Option<LogitDiffResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_questions: Option<ControlQuestionsResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_lens_comparison: Option<LogitLensComparisonResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_of_mind: Option<TopOfMindResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concept_activation: Option<ConceptActivationResult>,
    // ── Hybrid architecture experiments ──
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cka: Option<CkaResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routing_analysis: Option<RoutingAnalysisResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub causal_tracing: Option<CausalTracingResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steering_survival: Option<SteeringSurvivalResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gdn_state_stats: Option<GdnStateStatsResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_type_lens: Option<LayerTypeLensResult>,
    // ── Code-native detection experiments ──
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_logit_diff: Option<CodeLogitDiffResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_gen_detection: Option<CodeGenDetectionResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concept_identification: Option<ConceptIdentificationResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub discrimination_matrix: Option<DiscriminationMatrixResult>,
}

// ── Logit Lens (existing) ───────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct LogitLensData {
    pub tokens: Vec<String>,
    pub token_ids: Vec<u32>,
    pub layers: Vec<LayerData>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LayerData {
    pub layer_idx: usize,
    pub layer_type: String,
    pub top1_prob: f32,
    pub top_tokens: Vec<TokenProb>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TokenProb {
    pub token: String,
    pub token_id: u32,
    pub probability: f32,
}

// ── Logit Diff ──────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct LogitDiffResult {
    /// User turn 1 variant used (no_info, with_info, inaccurate_info).
    pub variant: String,
    /// Concept steering vector name.
    pub concept: String,
    pub scale: f64,
    pub layers: Vec<usize>,
    /// Number of detection prompt variants (trials) aggregated.
    pub num_trials: usize,
    /// Top tokens by steered probability, with base comparison.
    pub top_tokens: Vec<LogitDiffToken>,
    /// Mean P(yes) in base condition across trials.
    pub base_p_yes: f32,
    /// Mean P(no) in base condition across trials.
    pub base_p_no: f32,
    /// Mean P(yes) in steered condition across trials.
    pub steered_p_yes: f32,
    /// Mean P(no) in steered condition across trials.
    pub steered_p_no: f32,
    /// Mean P(yes) in random-direction control across trials.
    pub random_control_p_yes: f32,
    /// Mean P(no) in random-direction control across trials.
    pub random_control_p_no: f32,
    /// Mean yes-probability shift (steered - base).
    pub mean_yes_shift: f32,
    /// Std-dev of yes-probability shift across trials.
    pub std_yes_shift: f32,
    /// Standard error of yes-probability shift across trials.
    pub stderr_yes_shift: f32,
    /// 95% confidence interval bounds for yes-probability shift.
    pub yes_shift_ci95_low: f32,
    pub yes_shift_ci95_high: f32,
    /// Trial-level measurements.
    pub trials: Vec<LogitDiffTrialResult>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LogitDiffToken {
    pub token: String,
    pub token_id: u32,
    pub base_prob: f32,
    pub steered_prob: f32,
    pub diff: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LogitDiffTrialResult {
    pub prompt_variant: String,
    pub base_p_yes: f32,
    pub base_p_no: f32,
    pub steered_p_yes: f32,
    pub steered_p_no: f32,
    pub random_control_p_yes: f32,
    pub random_control_p_no: f32,
}

// ── Control Questions ───────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct ControlQuestionsResult {
    pub concept: String,
    pub scale: f64,
    pub layers: Vec<usize>,
    pub questions: Vec<ControlQuestionResult>,
    pub summary: ControlQuestionsSummary,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ControlQuestionResult {
    pub question: String,
    pub expected_answer: String,
    pub base_p_yes: f32,
    pub base_p_no: f32,
    pub steered_p_yes: f32,
    pub steered_p_no: f32,
    pub base_correct_prob: f32,
    pub steered_correct_prob: f32,
    pub base_predicted_answer: String,
    pub steered_predicted_answer: String,
    pub base_is_correct: bool,
    pub steered_is_correct: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ControlQuestionsSummary {
    pub mean_base_p_yes: f32,
    pub mean_base_p_no: f32,
    pub mean_steered_p_yes: f32,
    pub mean_steered_p_no: f32,
    pub mean_yes_shift: f32,
    pub mean_no_shift: f32,
    pub std_yes_shift: f32,
    pub std_no_shift: f32,
    pub mean_base_correct_prob: f32,
    pub mean_steered_correct_prob: f32,
    pub mean_correct_prob_shift: f32,
    pub std_correct_prob_shift: f32,
    pub base_accuracy: f32,
    pub steered_accuracy: f32,
    pub accuracy_shift: f32,
}

// ── Logit Lens Comparison ───────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct LogitLensComparisonResult {
    pub variant: String,
    pub concept: String,
    pub scale: f64,
    /// Token strings being tracked across layers.
    pub tracked_tokens: Vec<String>,
    /// Per-layer data for base (no steering) forward pass.
    pub base_layers: Vec<ComparisonLayerData>,
    /// Per-layer data for steered forward pass.
    pub steered_layers: Vec<ComparisonLayerData>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ComparisonLayerData {
    pub layer_idx: usize,
    pub layer_type: String,
    /// Probabilities for each tracked token at this layer (same order as tracked_tokens).
    pub tracked_probs: Vec<f32>,
    /// Top-1 token and its probability at this layer.
    pub top1_token: String,
    pub top1_prob: f32,
}

// ── Top of Mind (Generation) ────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct TopOfMindResult {
    pub concept: String,
    pub scale: f64,
    pub layers: Vec<usize>,
    pub prompt: String,
    pub generated_text: String,
    pub num_tokens: usize,
    pub temperature: f64,
    pub stop_reason: String,
}

// ── Concept Activation ──────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct ConceptActivationResult {
    pub concept: String,
    pub text: String,
    /// Per-layer activation score (dot product with concept direction).
    pub activations: Vec<ConceptLayerActivation>,
    /// Mean activation across measured layers.
    pub mean_activation: f32,
    /// Max activation across measured layers.
    pub max_activation: f32,
    /// Layer index with max activation.
    pub max_layer: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ConceptLayerActivation {
    pub layer_idx: usize,
    pub activation: f32,
}

// ══════════════════════════════════════════════════════════════════════
// Hybrid Architecture Experiment Results (QWEN3-SPEC)
// ══════════════════════════════════════════════════════════════════════

// ── CKA Cross-Layer Similarity (3A) ─────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct CkaResult {
    /// (N+1) x (N+1) similarity matrix (N = num_layers, +1 for embedding).
    pub similarity_matrix: Vec<Vec<f32>>,
    /// Labels for each row/column: "emb", "L0 (gdn)", "L3 (attn)", etc.
    pub layer_labels: Vec<String>,
    /// CKA similarity between consecutive layers (diagonal+1 of similarity_matrix).
    pub consecutive_cka: Vec<f32>,
}

// ── MoE Routing Analysis (3B) ───────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct RoutingAnalysisResult {
    /// Per-layer routing entropy (how decisive is the router?)
    pub layer_entropy: Vec<LayerFloat>,
    /// Per-expert selection frequency: (layer_idx, [freq_per_expert; 64])
    pub expert_frequency: Vec<LayerFrequency>,
    /// KL(steered || base) per layer — how much steering changes routing
    pub routing_divergence: Option<Vec<LayerFloat>>,
    /// Shared expert gate values per layer
    pub shared_gate_values: Vec<LayerFloat>,
    /// Shared gate delta under steering
    pub shared_gate_delta: Option<Vec<LayerFloat>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LayerFloat {
    pub layer_idx: usize,
    pub value: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LayerFrequency {
    pub layer_idx: usize,
    pub frequencies: Vec<f32>,
}

// ── Causal Tracing (3C) ─────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct CausalTracingResult {
    /// Per-layer recovery: (layer_idx, layer_type, recovery fraction 0-1)
    pub layer_recovery: Vec<CausalTracingLayer>,
    pub clean_top_token: String,
    pub corrupted_top_token: String,
    pub clean_prob: f32,
    pub corrupted_prob: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CausalTracingLayer {
    pub layer_idx: usize,
    pub layer_type: String,
    pub recovery: f32,
}

// ── Steering Survival Through GDN (3D) ──────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct SteeringSurvivalResult {
    /// Per-injection-point decay traces.
    pub traces: Vec<SteeringSurvivalTrace>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SteeringSurvivalTrace {
    pub injection_layer: usize,
    pub injection_layer_type: String,
    /// (layer_idx, layer_type, concept_activation) for layers after injection.
    pub activations: Vec<SurvivalActivation>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SurvivalActivation {
    pub layer_idx: usize,
    pub layer_type: String,
    pub activation: f32,
}

// ── GDN Recurrent State Stats (3E) ──────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct GdnStateStatsResult {
    pub layers: Vec<GdnStateStats>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GdnStateStats {
    pub layer_idx: usize,
    pub frobenius_norm: f32,
    pub effective_rank: f32,
    pub top_singular_value: f32,
    pub spectral_entropy: f32,
}

// ── Logit Lens by Layer Type (3F) ───────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct LayerTypeLensResult {
    pub layers: Vec<LayerTypeLensEntry>,
    pub mean_gdn_confidence: f32,
    pub mean_attn_confidence: f32,
    /// Confidence jumps at full attention layers: (layer_idx, delta from previous layer)
    pub attn_jumps: Vec<LayerFloat>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LayerTypeLensEntry {
    pub layer_idx: usize,
    pub layer_type: String,
    pub top1_prob: f32,
    pub top1_token: String,
    pub entropy: f32,
}

// ══════════════════════════════════════════════════════════════════════
// Code-Native Detection Experiment Results
// ══════════════════════════════════════════════════════════════════════

// ── Code Logit Diff (Phase 1) ───────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct CodeLogitDiffResult {
    pub concept: String,
    pub scale: f64,
    pub layers: Vec<usize>,
    /// Per-template × per-variant trial results.
    pub trials: Vec<CodeLogitDiffTrial>,
    /// Mean coverage P(True)+P(False) across all base trials — the key diagnostic.
    pub mean_base_coverage: f32,
    pub mean_steered_coverage: f32,
    pub mean_random_coverage: f32,
    /// Mean P(True) across conditions.
    pub base_p_true: f32,
    pub steered_p_true: f32,
    pub random_p_true: f32,
    /// Mean P(False) across conditions.
    pub base_p_false: f32,
    pub steered_p_false: f32,
    pub random_p_false: f32,
    /// Mean true-probability shift (steered - base).
    pub mean_true_shift: f32,
    pub std_true_shift: f32,
    /// Top tokens by steered probability (from first trial).
    pub top_tokens: Vec<LogitDiffToken>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CodeLogitDiffTrial {
    pub template: String,
    pub variant: String,
    pub base_p_true: f32,
    pub base_p_false: f32,
    pub base_coverage: f32,
    pub steered_p_true: f32,
    pub steered_p_false: f32,
    pub steered_coverage: f32,
    pub random_p_true: f32,
    pub random_p_false: f32,
    pub random_coverage: f32,
}

// ── Code Gen Detection (Phase 2) ────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct CodeGenDetectionResult {
    pub concept: String,
    pub scale: f64,
    pub layers: Vec<usize>,
    pub conditions: Vec<CodeGenCondition>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CodeGenCondition {
    pub template: String,
    /// "base", "steered", or "random"
    pub condition: String,
    pub temperature: f64,
    pub num_trials: usize,
    pub generations: Vec<CodeGenTrial>,
    /// Fraction of generations that parsed as True.
    pub detection_rate: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CodeGenTrial {
    pub generated_text: String,
    pub parsed_result: Option<bool>,
    pub num_tokens: usize,
}

// ── Concept Identification (Phase 3) ────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct ConceptIdentificationResult {
    pub concepts: Vec<String>,
    pub scale: f64,
    pub layers: Vec<usize>,
    pub identifications: Vec<ConceptIdEntry>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ConceptIdEntry {
    pub injected_concept: String,
    pub generated_text: String,
    pub parsed_concept: Option<String>,
    /// "exact", "synonym", "miss", or "unparseable"
    pub match_type: String,
}

// ── Discrimination Matrix (Phase 4) ─────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct DiscriminationMatrixResult {
    pub concepts: Vec<String>,
    pub scale: f64,
    pub layers: Vec<usize>,
    /// N×N matrix: matrix[i][j] = fraction of times injecting concept i identified concept j.
    pub matrix: Vec<Vec<f32>>,
    pub per_concept_accuracy: Vec<f32>,
    pub overall_accuracy: f32,
}

// ══════════════════════════════════════════════════════════════════════
// API Request Types
// ══════════════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize)]
pub struct TokenizeRequest {
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct ForwardRequest {
    pub text: String,
    pub top_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct SetSteeringVectorRequest {
    pub layers: Vec<usize>,
    pub scale: f64,
    pub vector: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct TrainRequest {
    pub concept: String,
    pub num_suffixes: Option<usize>,
    pub layers: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
pub struct ApplySteeringVectorRequest {
    pub name: String,
    pub layers: Option<Vec<usize>>,
    pub scale: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct RunLogitDiffRequest {
    pub concept: String,
    pub scale: Option<f64>,
    pub user_turn1_variant: Option<String>,
    pub layers: Option<Vec<usize>>,
    pub top_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct RunControlQuestionsRequest {
    pub concept: String,
    pub scale: Option<f64>,
    pub user_turn1_variant: Option<String>,
    pub layers: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
pub struct RunLogitLensComparisonRequest {
    pub concept: String,
    pub scale: Option<f64>,
    pub tracked_tokens: Vec<String>,
    pub user_turn1_variant: Option<String>,
    pub layers: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
pub struct RunTopOfMindRequest {
    pub concept: String,
    pub scale: Option<f64>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub layers: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
pub struct RunLayerTypeLensRequest {
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct RunSteeringSurvivalRequest {
    pub concept: String,
    pub injection_layers: Vec<usize>,
    pub scale: Option<f64>,
    pub probe_text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RunCkaRequest {
    pub texts: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct RunRoutingAnalysisRequest {
    pub text: String,
    pub concept: Option<String>,
    pub scale: Option<f64>,
    pub layers: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
pub struct RunCausalTracingRequest {
    pub clean_text: String,
    pub corrupted_text: String,
}

#[derive(Debug, Deserialize)]
pub struct RunGdnStateStatsRequest {
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct RunCodeLogitDiffRequest {
    pub concept: String,
    pub scale: Option<f64>,
    pub layers: Option<Vec<usize>>,
    pub top_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct RunCodeGenDetectionRequest {
    pub concept: String,
    pub scale: Option<f64>,
    pub layers: Option<Vec<usize>>,
    pub max_tokens: Option<usize>,
    pub temperatures: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
pub struct RunConceptIdentificationRequest {
    pub concepts: Vec<String>,
    pub scale: Option<f64>,
    pub layers: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
pub struct RunDiscriminationMatrixRequest {
    pub concepts: Vec<String>,
    pub scale: Option<f64>,
    pub layers: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
pub struct MeasureConceptActivationRequest {
    pub concept: String,
    pub text: String,
    pub layers: Option<Vec<usize>>,
}

/// Default steering layers: middle third of the model (introspection indices).
pub fn default_steering_layers(num_layers: usize) -> Vec<usize> {
    let start = num_layers / 3;
    let end = 2 * num_layers / 3;
    (start..end).map(|i| i + 1).collect()
}

pub fn now_timestamp() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}

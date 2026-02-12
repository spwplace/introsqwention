# QWEN3-SPEC: Hybrid Architecture Experiments

Focused spec for the Qwen3-Next-specific interpretability work — the novel stuff that
nobody has done because this architecture class (GDN + full attention + sparse MoE) didn't
exist until recently.

Parallel track: another agent handles Qwen2.5 replication on :3131. We handle the hybrid
architecture analysis. Server is running Qwen3-Coder-Next-MLX-4bit.

## Known constraints (from handoff-notes.md)

- **4-bit quantization wall**: Scale >20 corrupts residual stream. Sweet spot: 5–12.
- **PCA sign inversion**: `measure_concept_activation` returned negative for relevant text.
  May need negative scale or sign-flip heuristic.
- **Metal contiguity**: Must `.contiguous()` before F32 casts on Metal to avoid kernel panics.
- **KV cache ghosting**: Fixed — `LocalHybridCache::reset()` called at start of forward if
  `seqlen_offsets[0] == 0`.

---

## Step 1: Model hooks in `qwen3_next.rs`

These are prerequisite infrastructure. Each is a small, surgical change.

### 1A. MoE Routing Capture

**What changes**: `SparseMoeBlock` gains a capture buffer.

```
File: mistral-introspect/mistralrs-core/src/models/qwen3_next.rs
```

Add to `SparseMoeBlock`:
```rust
struct SparseMoeBlock {
    // ... existing fields ...
    /// Captured routing data from last forward pass (when capture is enabled).
    routing_capture: Arc<Mutex<Option<MoeRoutingData>>>,
}
```

New struct (at top of file, near IntrospectionState):
```rust
pub struct MoeRoutingData {
    pub router_logits: Tensor,    // (num_tokens, num_experts) — raw, pre-softmax
    pub topk_indices: Tensor,     // (num_tokens, num_experts_per_tok)
    pub topk_weights: Tensor,     // (num_tokens, num_experts_per_tok) — post-softmax, post-norm
    pub shared_gate: Tensor,      // (num_tokens,) — sigmoid output
}
```

In `SparseMoeBlock::forward()` (line 1248), after line 1267 (topk_weights computed):
```rust
// Capture routing decisions
{
    let mut cap = self.routing_capture.lock().unwrap();
    if cap.is_some() {
        *cap = Some(MoeRoutingData {
            router_logits: router_logits.clone(),
            topk_indices: topk_ids.clone(),
            topk_weights: topk_weights.clone(),
            shared_gate: Tensor::zeros(1, DType::F32, router_logits.device())?, // filled below
        });
    }
}
```

And after line 1280 (shared_gate computed), update the capture with the real gate value:
```rust
{
    let mut cap = self.routing_capture.lock().unwrap();
    if let Some(ref mut data) = *cap {
        data.shared_gate = shared_gate.squeeze(D::Minus1)?.clone();
    }
}
```

Add to `IntrospectionState`:
```rust
pub routing_data: Vec<MoeRoutingData>,
pub capture_routing: bool,
```

In Model::forward() loop body (line ~1741), after the layer forward call but before
introspection capture: if `intro.capture_routing`, take the routing data from
`layer.moe.routing_capture` and push it.

**Enable/disable**: add `set_capture_routing(bool)` and `take_routing_data()` on Model.

**Cost**: Cloning 3 tensors per MoE layer per forward pass when enabled. The tensors are
small: `(seq_len, 64)` for router_logits, `(seq_len, 8)` for topk. Negligible.

**Lines changed**: ~50 in qwen3_next.rs, ~20 in introspection.rs for the API.

---

### 1B. Activation Patching / Replacement

**What changes**: `IntrospectionState` gains two new HashMaps.

```rust
pub struct IntrospectionState {
    pub hidden_states: Vec<Tensor>,
    pub steering_vectors: HashMap<usize, Tensor>,       // existing — additive
    pub patch_vectors: HashMap<usize, Tensor>,           // NEW — full replacement
    pub activation_caps: HashMap<usize, (Tensor, f32)>,  // NEW — (direction, threshold)
    pub capture: bool,
    pub capture_layers: Option<HashSet<usize>>,
    pub routing_data: Vec<MoeRoutingData>,               // from 1A
    pub capture_routing: bool,                            // from 1A
}
```

In forward loop (line ~1749), after steering vector addition:
```rust
// Activation patching: REPLACE hidden state
if let Some(patch) = intro.patch_vectors.get(&layer_idx) {
    x = patch.to_dtype(x.dtype())?.clone();
}

// Activation capping: clamp projection onto direction
if let Some((direction, threshold)) = intro.activation_caps.get(&layer_idx) {
    let dir = direction.to_dtype(x.dtype())?;
    // Project last-token hidden state onto direction
    let last = x.i((.., x.dim(1)? - 1, ..))?.squeeze(1)?;
    let proj = last.mul(&dir)?.sum(D::Minus1)?.unsqueeze(D::Minus1)?;
    // Clamp: if |proj| > threshold, subtract excess
    let excess = (proj.abs()? - *threshold)?.relu()?;
    let sign = proj.signum()?;
    let correction = (sign * excess)?.unsqueeze(1)?.broadcast_mul(&dir.unsqueeze(0)?)?;
    x = (x - correction)?;
}
```

New Model methods:
```rust
pub fn set_patch(&self, layer_idx: usize, hidden_state: Tensor);
pub fn clear_patches(&self);
pub fn set_activation_cap(&self, layer_idx: usize, direction: Tensor, threshold: f32);
pub fn clear_activation_caps(&self);
```

**Lines changed**: ~40 in qwen3_next.rs, ~30 in introspection.rs.

---

### 1C. GDN Recurrent State Read

**What changes**: A method on Model to extract recurrent states from the local cache.

```rust
impl Model {
    /// Extract GDN recurrent states from cache after a forward pass.
    /// Returns (layer_idx, recurrent_state) for each GDN layer.
    /// Shape: (1, num_v_heads, head_k_dim, head_v_dim)
    pub fn gdn_recurrent_states(&self) -> Result<Vec<(usize, Tensor)>> {
        let cache = self.local_cache.lock().unwrap();
        let mut states = Vec::new();
        for (i, lc) in cache.caches.iter().enumerate() {
            if let LocalLayerCache::LinearAttention(gdn_cache) = lc {
                states.push((i, gdn_cache.recurrent_state.clone()));
            }
        }
        Ok(states)
    }
}
```

Then in `IntrospectionModel` (introspection.rs), expose it:
```rust
pub fn gdn_recurrent_states(&self) -> anyhow::Result<Vec<(usize, Tensor)>>
```

**Lines changed**: ~15 in qwen3_next.rs, ~15 in introspection.rs.

---

## Step 2: Expose in IntrospectionModel API

`mistral-introspect/mistralrs-core/src/introspection.rs` needs wrapper methods for
the new model capabilities:

```rust
impl IntrospectionModel {
    // Routing
    pub fn set_capture_routing(&self, capture: bool);
    pub fn take_routing_data(&self) -> Vec<MoeRoutingData>;

    // Patching
    pub fn set_patch(&self, layer_idx: usize, hidden_state: Tensor);
    pub fn clear_patches(&self);
    pub fn set_activation_cap(&self, layer_idx: usize, direction: Tensor, threshold: f32);
    pub fn clear_activation_caps(&self);

    // GDN state
    pub fn gdn_recurrent_states(&self) -> anyhow::Result<Vec<(usize, Tensor)>>;

    // Convenience: forward with routing capture
    pub fn forward_introspect_with_routing(
        &self, input_ids: &Tensor, seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
    ) -> anyhow::Result<(Tensor, Vec<Tensor>, Vec<MoeRoutingData>)>;
}
```

Each of these just delegates to the inner `Model` through the existing match on
`ModelBackend::Qwen3Next { model, .. }` / `ModelBackend::Qwen2 { .. }`.
The Qwen2 branch returns errors/empty for GDN-specific calls.

---

## Step 3: Experiments (in introsqwention/src/)

### 3A. CKA Cross-Layer Similarity

**File**: `src/experiments.rs` (add to existing)

```rust
pub struct CkaResult {
    pub similarity_matrix: Vec<Vec<f32>>,  // (N+1) x (N+1), N = num_layers
    pub layer_labels: Vec<String>,          // "emb", "L0 (gdn)", "L1 (gdn)", "L3 (attn)", ...
    pub consecutive_cosine: Vec<f32>,       // cosine sim between adjacent layers
}

pub fn run_cka(
    state: &SharedState,
    texts: &[String],       // 50-100 diverse prompts
) -> anyhow::Result<CkaResult>
```

Algorithm:
1. For each text, `forward_introspect()`, extract last-token hidden at each layer
2. Stack → per-layer matrix (N_texts, hidden_size)
3. Linear CKA: `||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)`
4. Also compute cosine sim between consecutive layers (simple, blog-friendly)

**MCP tool**: `run_cka` with optional `texts` param (default: built-in diverse set)

**Dashboard**: 48x48 heatmap with GDN/Attn annotations on axes. Consecutive cosine as a
line chart with layer-type background coloring.

**Effort**: ~120 lines experiment, ~40 lines MCP, ~60 lines chart.

---

### 3B. MoE Routing Analysis

**File**: `src/experiments.rs`

```rust
pub struct RoutingAnalysisResult {
    /// Per-layer routing entropy (how decisive is the router?)
    pub layer_entropy: Vec<(usize, f32)>,
    /// Per-expert selection frequency: (layer_idx, [freq_per_expert; 64])
    pub expert_frequency: Vec<(usize, Vec<f32>)>,
    /// KL(steered || base) per layer — how much steering changes routing
    pub routing_divergence: Option<Vec<(usize, f32)>>,
    /// Per-layer expert activation shift under steering: (layer, [delta_freq; 64])
    pub expert_shift: Option<Vec<(usize, Vec<f32>)>>,
    /// Shared expert gate values per layer
    pub shared_gate_values: Vec<(usize, f32)>,
    /// Shared gate delta under steering
    pub shared_gate_delta: Option<Vec<(usize, f32)>>,
}

pub fn run_routing_analysis(
    state: &SharedState,
    text: &str,
    concept: Option<&str>,  // if Some, compare base vs steered
    scale: f64,
    layers: &[usize],
) -> anyhow::Result<RoutingAnalysisResult>
```

Algorithm:
1. Base forward with routing capture → base routing data
2. If concept: steered forward with routing capture → steered routing data
3. Compute per-layer entropy: `H = -sum(p * log(p))` on softmaxed router logits
4. Expert frequency: count topk selections across tokens
5. KL divergence: between base and steered routing weight distributions
6. Shared gate: mean sigmoid value per layer

**MCP tool**: `run_routing_analysis`

**Dashboard**: Expert heatmap (experts × layers), entropy line, routing divergence bars.

**Effort**: ~150 lines experiment, ~50 lines MCP, ~80 lines chart.

---

### 3C. Causal Tracing

**File**: `src/experiments.rs`

Depends on: Step 1B (activation patching).

```rust
pub struct CausalTracingResult {
    pub layer_recovery: Vec<(usize, String, f32)>,  // (layer, type, recovery %)
    pub clean_top_token: String,
    pub corrupted_top_token: String,
    pub clean_prob: f32,
    pub corrupted_prob: f32,
}

pub fn run_causal_tracing(
    state: &SharedState,
    clean_text: &str,        // "The Eiffel Tower is in"
    corrupted_text: &str,    // "The Jfksl Qwert is in"
) -> anyhow::Result<CausalTracingResult>
```

Algorithm:
1. Clean forward → capture all hidden states, record P(top-1 token)
2. Corrupted forward → record P(clean_top_token) — should be ~0
3. For each layer i:
   - Set patch at layer i = clean hidden state i
   - Corrupted forward → measure P(clean_top_token)
   - Recovery = P(patched) / P(clean)
   - Clear patch

**Dashboard**: Bar chart colored by layer type (GDN=grey, Attn=teal).

**Effort**: ~120 lines experiment, ~40 lines MCP, ~40 lines chart.

---

### 3D. Steering Survival Through GDN

**File**: `src/experiments.rs`

Tests the handoff-notes hypothesis: "GDN might be more resistant to instant residual steering."

```rust
pub struct SteeringSurvivalResult {
    /// Per-layer concept activation after injecting at a single layer
    pub traces: Vec<SteeringSurvivalTrace>,
}

pub struct SteeringSurvivalTrace {
    pub injection_layer: usize,
    pub injection_layer_type: String,
    /// (layer_idx, layer_type, concept_activation) for all layers after injection
    pub activations: Vec<(usize, String, f32)>,
}

pub fn run_steering_survival(
    state: &SharedState,
    concept: &str,
    injection_layers: &[usize],  // e.g., [15, 16, 17, 18] — mix of GDN and attn
    scale: f64,
) -> anyhow::Result<SteeringSurvivalResult>
```

Algorithm:
For each injection layer L:
1. Set steering at layer L only
2. Forward introspect → capture all hidden states
3. For each subsequent layer, compute dot(hidden_state, concept_direction) / norms
4. Clear steering

This produces a "decay curve" for each injection point. Compare decay rates when
injecting at GDN vs full attention layers.

**Dashboard**: Multi-line chart — one line per injection layer, x = layer depth,
y = concept activation. GDN injection lines in grey, Attn injection in teal.

**Effort**: ~80 lines experiment, ~30 lines MCP, ~40 lines chart.

---

### 3E. GDN Recurrent State Logit Lens

**File**: `src/experiments.rs`

Depends on: Step 1C (recurrent state read).

```rust
pub struct GdnStateLensResult {
    /// Per-GDN-layer: top-k token predictions from the recurrent state
    pub layers: Vec<GdnStateLensLayer>,
}

pub struct GdnStateLensLayer {
    pub layer_idx: usize,
    pub top_tokens: Vec<TokenProb>,   // reuse existing TokenProb
}

pub fn run_gdn_state_lens(
    state: &SharedState,
    text: &str,
) -> anyhow::Result<GdnStateLensResult>
```

Algorithm:
1. Forward pass (text)
2. Extract GDN recurrent states → `Vec<(layer_idx, state)>`
3. For each state (shape: `(1, num_v_heads, head_k_dim, head_v_dim)`):
   a. Reshape to (1, num_v_heads * head_v_dim) = (1, value_dim)
   b. This isn't hidden_size — need to project through GDN's `out_proj` first
   c. Then project through norm + lm_head (logit lens)
   d. Extract top-k predictions

**Challenge**: The recurrent state lives in (head_k_dim, head_v_dim) space, not hidden_size.
To get a "logit lens" we need: state → matmul with query → value output → out_proj → norm → lm_head.
This is more complex than a simple projection. Alternative: flatten the recurrent state
and train a linear probe to predict tokens from it (more principled).

**Simpler version**: Just report statistics about the recurrent state — Frobenius norm,
rank (via SVD), spectral entropy. These tell us how much information each GDN layer is
storing, without needing the full projection pipeline.

```rust
pub struct GdnStateStatsResult {
    pub layers: Vec<GdnStateStats>,
}

pub struct GdnStateStats {
    pub layer_idx: usize,
    pub frobenius_norm: f32,
    pub effective_rank: f32,     // exp(spectral_entropy)
    pub top_singular_value: f32,
    pub condition_number: f32,   // sigma_max / sigma_min
}
```

**Effort**: Stats version: ~60 lines. Full logit lens version: ~120 lines + requires
exposing GDN out_proj through the API (more invasive).

---

### 3F. Logit Lens by Layer Type

**File**: `src/experiments.rs`

No new hooks needed — pure post-processing of existing `logit_lens_all()` output.

```rust
pub struct LayerTypeLensResult {
    pub layers: Vec<LayerTypeLensEntry>,
    /// Average confidence at GDN layers vs Attn layers
    pub mean_gdn_confidence: f32,
    pub mean_attn_confidence: f32,
    /// Confidence "jumps" at full attention layers
    pub attn_jumps: Vec<(usize, f32)>,  // (layer_idx, delta from previous layer)
}

pub struct LayerTypeLensEntry {
    pub layer_idx: usize,
    pub layer_type: String,
    pub top1_prob: f32,
    pub top1_token: String,
    pub entropy: f32,
}
```

Algorithm:
1. Forward introspect → hidden states
2. Logit lens all → per-layer probs
3. For each layer: top-1 prob, top-1 token, entropy
4. Compute jumps: prob[attn_layer] - prob[attn_layer - 1]
5. Averages by type

**Dashboard**: Line chart with x=layer, y=top-1 confidence. Background bands:
light grey for GDN, teal-tinted for Attn. Vertical lines at attention layer boundaries.

**Effort**: ~60 lines experiment, ~20 lines MCP, ~40 lines chart.

---

## Implementation Order

```
Phase A — Model hooks (qwen3_next.rs + introspection.rs):
  1A. MoE routing capture          ~70 lines    [unlocks 3B]
  1B. Activation patching          ~70 lines    [unlocks 3C]
  1C. GDN recurrent state read     ~30 lines    [unlocks 3E]

Phase B — Experiments (introsqwention/src/):
  3F. Logit lens by layer type     ~60 lines    [no deps, quick win]
  3D. Steering survival            ~80 lines    [no deps, tests key hypothesis]
  3A. CKA cross-layer similarity   ~120 lines   [no deps, core novel result]
  3B. MoE routing analysis         ~150 lines   [needs 1A]
  3C. Causal tracing               ~120 lines   [needs 1B]
  3E. GDN state stats              ~60 lines    [needs 1C]

Phase C — MCP + Dashboard:
  Wire each experiment as an MCP tool + add chart renderers to static/js/charts.js
```

Phase A is ~170 lines of model code. Phase B is ~590 lines of experiment code.
Phase C is plumbing.

Do Phase A first (all three hooks in one pass), then Phase B experiments can
be developed independently and in parallel.

---

## Files touched

| File | Changes |
|------|---------|
| `mistral-introspect/mistralrs-core/src/models/qwen3_next.rs` | +MoeRoutingData struct, +routing_capture on SparseMoeBlock, +patch/cap in forward loop, +gdn_recurrent_states() |
| `mistral-introspect/mistralrs-core/src/introspection.rs` | Expose new Model methods through IntrospectionModel API |
| `src/state.rs` | New result types for each experiment |
| `src/experiments.rs` | New experiment functions (3A-3F) |
| `src/mcp.rs` | New MCP tools |
| `static/js/charts.js` | New chart renderers |
| `static/js/experiment.js` | Wire new chart types |

---

## What this unlocks for the blog

| Blog Section | Experiment | Status |
|-------------|-----------|--------|
| §4 Hybrid Geometry | CKA (3A) + Logit lens by type (3F) | Planned |
| §5 Expert Routing | Routing analysis (3B) | Planned |
| §6 Optimal Steering | Steering survival (3D) | Planned |
| §8 Causal Tracing | Causal tracing (3C) | Planned |
| §4 (GDN deep-dive) | GDN state stats (3E) | Planned |

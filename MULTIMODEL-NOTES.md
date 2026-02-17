# Handoff: Implement New Model Architectures in mistral.rs

## Goal

We are replicating and extending [vgel's LLM introspection experiment](https://vgel.me/posts/qwen-introspection/) ("Small Models Can Introspect, Too"). This experiment tests whether LLMs can detect injected steering vectors in their own activations. We want to run it across a diverse set of model architectures — standard transformers, MoE, hybrid SSM/attention, and pure linear RNN — to discover where self-awareness lives across fundamentally different compute pathways.

Many of these models are **not currently supported** by mistral.rs. Your job is to implement them.

## What Exists

### Git Worktrees

All branched from `upstream/master` at `bdaf6eec9` ("Add new quant method: F8Q8 (#1883)"). The parent repo is `~/src/mistral.rs`.

| Worktree | Branch | Model | Org | Architecture Class |
|----------|--------|-------|-----|-------------------|
| `~/dev/mistral-models/olmo` | `olmo` | OLMo 3.1-32B | AllenAI | `Olmo3ForCausalLM` |
| `~/dev/mistral-models/afmoe` | `afmoe` | Trinity Mini 26B | Arcee AI | `AfmoeForCausalLM` |
| `~/dev/mistral-models/lfm` | `lfm` | LFM2.5 1.2B | Liquid AI | `Lfm2ForCausalLM` |
| `~/dev/mistral-models/falcon-h1` | `falcon-h1` | Falcon-H1 (0.5B-34B) | TII | `FalconH1ForCausalLM` |
| `~/dev/mistral-models/rwkv7` | `rwkv7` | RWKV-7 "Goose" (0.19B-2.9B) | RWKV | `Rwkv7ForCausalLM` |
| `~/dev/mistral-models/jamba` | `jamba` | Jamba 1.5 (52B/12B active) | AI21 | `JambaForCausalLM` |
| `~/dev/mistral-models/zamba2` | `zamba2` | Zamba2 (2.7B, 7B) | Zyphra | `Zamba2ForCausalLM` |
| `~/dev/mistral-models/nemotron-h` | `nemotron-h` | Nemotron-H (8B, 56B) | NVIDIA | `NemotronHForCausalLM` |
| `~/dev/mistral-models/phi4-flash` | `phi4-flash` | Phi-4-mini-flash (3.8B) | Microsoft | `Phi4FlashForCausalLM` |

There is also an `introspect` branch worktree at `~/dev/introsqwention/mistral-introspect/` which has the introspection infrastructure (steering vectors, logit lens, MCP server, dashboard) — but **you don't need to touch that**. Just implement the models as normal mistral.rs models. Introspection hooks will be added separately later.

### Existing Models as Reference

There are 22+ model implementations in `mistralrs-core/src/models/`. Key references by architecture type:

**Standard transformer:**
- **`smollm3.rs`** (802 lines) — Cleanest recent implementation. Best starting template.
- **`llama.rs`** (781 lines) — Classic Llama. OLMo3 is very close to this.

**MoE:**
- **`mixtral.rs`** — Standard MoE (8 experts, 2 active). Basic top-k routing.
- **`deepseek2.rs`** / **`deepseek3.rs`** — Complex MoE with shared experts, MLA attention.
- **`phi3_5_moe.rs`** — Another MoE reference.

**Hybrid / per-layer type switching:**
- **`qwen3_next.rs`** — Hybrid GDN/MoE. Has per-layer type dispatch (GDN layers vs full attention layers). Most complex existing model.
- **`granite.rs`** — IBM Granite MoE Hybrid (`GraniteMoeHybrid`). Another hybrid architecture with per-layer differences.

### Integration Checklist

For each new model, you need to touch these files:

1. **`mistralrs-core/src/models/<model>.rs`** — The model implementation itself
2. **`mistralrs-core/src/models/mod.rs`** — Add `pub(crate) mod <model>;`
3. **`mistralrs-core/src/pipeline/loaders/normal_loaders.rs`** — This is the big one:
   - Add variant to `NormalLoaderType` enum (~line 145)
   - Add `from_causal_lm_name` match arm (~line 193)
   - Add `from_str` match arm (~line 225)
   - Add `Display` impl arm (~line 253)
   - Add `to_loader` match arm (~line 315)
   - Implement `NormalModelLoader` (load, load_xlora, is_gptx, get_config_repr)
   - Implement `IsqModelLoader` (isq_layer_regexes, immediate_isq_predicates)
   - Implement `DeviceMappedModelLoader` (mapped_max_act_size_elems, non_mapped_max_act_size_elems, non_mapped_size_in_bytes, layer_sizes_in_bytes, num_layers, model_config)
4. **`mistralrs-core/src/pipeline/normal.rs`** — Add loader import (~line 14)

### The NormalModel Trait

Every model must implement `NormalModel` (defined in `normal_loaders.rs:42`):

```rust
pub trait NormalModel: IsqModel + AnyMoeBaseModelMixin {
    fn forward(&self, input_ids: &Tensor, seqlen_offsets: &[usize],
               context_lens: Vec<(usize, usize)>, position_ids: Vec<usize>,
               metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
               flash_params: &FlashParams) -> candle_core::Result<Tensor>;
    fn xlora_forward(...) -> candle_core::Result<Tensor>;
    fn is_xlora(&self) -> bool;
    fn device(&self) -> &Device;
    fn cache(&self) -> &EitherCache;
    fn cache_mut(&mut self) -> &mut EitherCache;
    fn max_seq_len(&self) -> usize;
    fn config(&self) -> &ModelConfigMetadata;
}
```

### Model Config Pattern

Each model defines a `#[derive(Debug, Clone, Deserialize)]` Config struct that maps directly to the HuggingFace `config.json`. Use `serde` defaults for optional fields. The Config is deserialized from `config.json` in the loader.

### VarBuilder Pattern

Weights are loaded via candle's `VarBuilder` with prefix navigation:
- `vb.pp("model")` — enters the `model.` prefix
- `vb.pp("layers").pp(i.to_string())` — enters `model.layers.0.` etc.
- `vb.get(shape, "weight")` — loads `<prefix>.weight` tensor

The weight key hierarchy must **exactly match** the PyTorch `state_dict` naming. Check `model.safetensors.index.json` from HuggingFace for the exact key names.

### KV Cache Pattern

All models use `EitherCache` (typically `NormalCache`). Each layer gets a cache entry. Reset when `seqlen_offsets[0] == 0`. Standard pattern:

```rust
let cache = &self.cache.normal().0;
// In layer forward:
let (k, v) = cache[layer_idx].lock().unwrap().update(...)?;
```

**Important for SSM/RNN models**: These don't use KV caches — they have fixed-size recurrent state. You'll need to manage their state separately (likely a `Vec<Tensor>` or similar per-layer state struct), but still conform to the `EitherCache` interface. Look at how `qwen3_next.rs` handles GDN layer state for a pattern.

---

## Tier 1: Standard Transformer & MoE (easiest)

### Model 1: OLMo 3.1-32B — Standard Transformer

**HuggingFace**: `allenai/OLMo-3.1-32B`
**Architecture class**: `Olmo3ForCausalLM`
**Model type**: `olmo3`
**Difficulty**: Easy — essentially a Llama variant

OLMo3 is a Llama-style transformer. Very close to `smollm3.rs` or `llama.rs`.

**Key things to check in `config.json`**:
- `num_hidden_layers`, `hidden_size`, `intermediate_size`
- `num_attention_heads`, `num_key_value_heads` (GQA)
- `hidden_act` (likely `silu`)
- `rms_norm_eps`
- `rope_theta`, `rope_scaling` (may have different RoPE config)
- `tie_word_embeddings`
- `max_position_embeddings`, `vocab_size`
- OLMo-specific: `clip_qkv` — OLMo sometimes clips QKV values after projection

**Weight key differences from Llama**: Check `model.safetensors.index.json`. OLMo may use slightly different naming (e.g., `model.transformer.` vs `model.`).

**Approach**: Copy `smollm3.rs`, rename structs, adjust Config fields and weight loading to match OLMo3's `config.json` and safetensors index.

---

### Model 2: Trinity Mini 26B / AfMoE — Custom MoE

**HuggingFace**: `arcee-ai/Trinity-Mini-26B`
**Architecture class**: `AfmoeForCausalLM`
**Model type**: `afmoe`
**Difficulty**: Medium — MoE patterns exist, but AfMoE has unique features

**Key architectural features**:
- **128 experts, 8 active per token** — far more than Mixtral (8 experts, 2 active)
- **1 shared expert** — always-active, added to routed expert output (like DeepSeek)
- **ReLU² activation** — `relu(x)^2` instead of SiLU for expert MLPs
- **Gated attention** — attention output goes through a gating mechanism

**Key things to check in `config.json`**:
- `num_experts`, `num_experts_per_tok`
- `shared_expert_intermediate_size` or similar
- `hidden_act` — may specify `relu2` or custom field
- Gate/router type — top-k routing, load balancing
- How gated attention is parameterized

**Reference implementations**:
- `mixtral.rs` for basic MoE routing
- `deepseek2.rs` for shared expert pattern
- For 128 experts, check if there's a packed/batched expert implementation vs naive loop

**Weight naming**: Likely `model.layers.N.mlp.experts.M.{gate_proj,up_proj,down_proj}` and `model.layers.N.mlp.shared_expert.{...}` — verify against safetensors index.

---

## Tier 2: Hybrid SSM/Transformer (most interesting for introspection)

These models mix Mamba/SSM layers with traditional attention layers. This is where introspection gets fascinating — steering vectors injected into SSM layers may propagate completely differently than in attention layers.

### Model 3: Falcon-H1 — Parallel SSM+Attention Hybrid

**HuggingFace**: `tiiuae/Falcon-H1-7B` (also 0.5B, 1.5B, 3B, 34B)
**Architecture class**: `FalconH1ForCausalLM` (check actual class name in config.json)
**Model type**: `falcon_h1`
**Difficulty**: Hard — novel parallel hybrid design
**Reference**: https://huggingface.co/blog/tiiuae/falcon-h1

**What makes it unique**: Unlike other hybrids that *interleave* SSM and attention layers, Falcon-H1 runs Mamba-2 heads and attention heads **in parallel within each block**. The outputs are then combined. This means every layer processes the input through both pathways simultaneously.

**Key architectural features**:
- Mamba-2 and attention heads in parallel within each "hybrid mixer block"
- 6 model sizes: 0.5B, 1.5B, 1.5B-Deep, 3B, 7B, 34B
- 256K context window
- Apache 2.0 license

**Key things to check**:
- How are the parallel pathways combined? (Concatenation? Addition? Gating?)
- Does each block have *both* Mamba-2 state and KV cache?
- What's the ratio of Mamba-2 heads to attention heads per block?
- HuggingFace transformers already has `FalconH1` support — check `modeling_falcon_h1.py`

**Why it's interesting for introspection**: Steering vectors hit both the SSM and attention pathways simultaneously. We can study whether introspection awareness comes from the attention pathway, the SSM pathway, or the combination.

---

### Model 4: Nemotron-H — Mostly-Mamba with Sparse Attention

**HuggingFace**: `nvidia/Nemotron-H-8B-Base-8K` (also 56B)
**Architecture class**: `NemotronHForCausalLM` (check actual class name)
**Model type**: `nemotron_h`
**Difficulty**: Medium-Hard — interleaved Mamba-2 with sparse attention
**Reference**: https://research.nvidia.com/labs/adlr/nemotronh/

**Architecture (8B)**:
- 24 Mamba-2 layers + 24 MLP layers + **only 4 self-attention layers**
- Mostly Mamba-2, with very sparse attention (4 out of ~48 layers)
- 3x throughput vs same-size pure transformers

**Architecture (56B)**:
- 54 Mamba-2 layers + 54 MLP layers + 10 self-attention layers

**Key things to check**:
- Layer ordering — which positions get the attention layers?
- Mamba-2 layer config (state dimension, conv dimension, expansion factor)
- Is attention full or sliding window?
- Paper: https://arxiv.org/abs/2504.03624

**Why it's interesting for introspection**: With only 4 attention layers out of ~48 total, steering vectors must propagate primarily through SSM layers. If introspection still works, that's strong evidence that attention isn't required for self-monitoring.

---

### Model 5: Jamba 1.5 — Triple Hybrid (SSM + Attention + MoE)

**HuggingFace**: `ai21labs/AI21-Jamba-Large-1.5` (52B total, 12B active)
**Architecture class**: `JambaForCausalLM`
**Model type**: `jamba`
**Difficulty**: Hard — three mechanisms interacting
**Reference**: https://www.ai21.com/research/jamba-1-5-hybrid-transformer-mamba-models-at-scale/

**Key architectural features**:
- **Triple hybrid**: Mamba layers + Transformer attention layers + MoE
- 1:8 ratio — only 1 out of every 8 layers uses attention
- 52B total params, 12B active per token (via MoE)
- 256K context window

**Key things to check**:
- Layer structure: which layers are Mamba, which are attention, which have MoE?
- MoE config: num experts, active per token
- How do Mamba and attention layers interact across depth?
- HuggingFace transformers has `Jamba` support — check `modeling_jamba.py`

**Why it's interesting for introspection**: Three different compute pathways. Can steering vectors injected into Mamba layers be detected when the detection question only passes through a few attention layers? This tests whether introspection requires the same mechanism that processed the injected signal.

---

### Model 6: Zamba2 — Shared Attention Backbone

**HuggingFace**: `Zyphra/Zamba2-7B` (also 2.7B)
**Architecture class**: `Zamba2ForCausalLM`
**Model type**: `zamba2`
**Difficulty**: Medium — Mamba-2 + cleverly shared attention
**Reference**: https://huggingface.co/Zyphra/Zamba2-7B

**Key architectural features**:
- Mamba-2 backbone with **only 2 shared attention layers**
- The 2 attention blocks are reused in an ABAB pattern throughout the network
- LoRA projectors applied to shared MLP blocks for depth specialization
- Very parameter-efficient design

**Key things to check**:
- How is the ABAB sharing implemented? (Same weight matrices reused at multiple depths?)
- How do LoRA projectors specialize the shared layers?
- Mamba-2 block details
- HuggingFace transformers has `Zamba2` support — check `modeling_zamba2.py`

**Why it's interesting for introspection**: The shared attention layers create a unique dynamic — steering at different depths feeds through the *same* attention weights. This could amplify or cancel out steering effects in ways that don't happen with independent layers.

---

### Model 7: Phi-4-mini-flash — SambaY (Mamba + SWA + GMU)

**HuggingFace**: `microsoft/Phi-4-mini-flash-reasoning`
**Architecture class**: `Phi4FlashForCausalLM` (check actual class name — may be `PhiMiniFlashForCausalLM` or similar)
**Model type**: `phi4_flash`
**Difficulty**: Hard — novel GMU mechanism
**Reference**: https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning

**Key architectural features**:
- **"SambaY" architecture**: decoder-hybrid-decoder
- Self-decoder: Mamba + Sliding Window Attention (SWA)
- Cross-decoder: interleaves cross-attention layers with Gated Memory Units (GMU)
- GMU: lightweight mechanism for sharing representations between layers
- 3.8B params, 200K vocab, 10x throughput improvement for long generation

**Key things to check**:
- What exactly is a GMU? (Check the paper and modeling code carefully)
- How does the decoder-hybrid-decoder structure work?
- Where is the single full attention layer positioned?
- This is Microsoft's most novel architecture — may require significant new code

---

## Tier 3: Pure Linear RNN (most architecturally radical)

### Model 8: RWKV-7 "Goose" — Pure Linear RNN, No Attention

**HuggingFace**: `RWKV/RWKV7-Goose-World3-1.5B-HF` (also 0.19B, 0.4B, 2.9B)
**Architecture class**: `Rwkv7ForCausalLM` (check actual HF class name)
**Model type**: `rwkv7`
**Difficulty**: Hard — completely different from transformers
**Reference**: https://arxiv.org/abs/2503.14456, https://github.com/BlinkDL/RWKV-LM

**Key architectural features**:
- **No attention at all** — 100% RNN architecture
- **Gated delta rule** with vector-valued gating and in-context learning rates
- Constant memory, linear time (no KV cache, fixed-size state per layer)
- Can recognize all regular languages
- 3x faster than RWKV-6 at 16K+ sequences
- Apache 2.0 license

**Key things to check**:
- The recurrent state structure — what tensors define the per-layer state?
- Gated delta rule forward pass (very different from attention)
- How does the HF implementation handle autoregressive generation? (State passing between steps)
- Time-mixing and channel-mixing blocks
- Check https://www.oxen.ai/blog/how-rwkv-7-goose-works-notes-from-the-author for architecture explanation
- HuggingFace transformers has RWKV support — check `modeling_rwkv7.py`

**Why it's the most interesting for introspection**: If a model with ZERO attention can detect injected steering vectors, that fundamentally challenges assumptions about what architectural mechanisms enable self-monitoring. RWKV-7's fixed-size recurrent state means information must be *compressed* through a bottleneck — can introspection survive that compression?

---

## Tier 4: Hybrid Conv/Attention

### Model 9: LFM2.5 1.2B — Hybrid Conv/Attention

**HuggingFace**: `LiquidAI/LFM2-1.2B` or `LiquidAI/LFM2.5-1.2B`
**Architecture class**: `Lfm2ForCausalLM`
**Model type**: `lfm2`
**Difficulty**: Hard — novel conv layers may be SSM-like

**Key architectural features**:
- **16 layers total**: 10 convolutional + 6 full attention
- **Layer pattern**: `[conv, conv, attn, conv, conv, attn, ...]` — attention every 3rd layer
- **Convolutional layers**: NOT standard convolutions — likely SSM or linear recurrence (similar to Mamba/Hyena). Check modeling code carefully.
- Small model (1.2B), fast to iterate on

**Key things to check**:
- How are conv layers defined? (Config will specify per-layer types)
- What is the conv layer's forward pass? (Causal conv1d? SSM? Mamba-style selective scan?)
- Does the conv layer have its own "cache" for autoregressive generation?
- RoPE: only applied to attention layers, or also conv?
- Are there `modeling_lfm2.py` files in the HuggingFace repo?

**References**:
- `qwen3_next.rs` for per-layer type switching
- `granite.rs` (GraniteMoeHybrid) for hybrid layer types

---

## Build & Test

```bash
# Check compilation (fast)
cargo check -p mistralrs-core --features metal

# Full build (if you want to test loading)
cargo build --release --features metal

# Run with a model
./target/release/mistralrs-cli run -m <path-to-model-weights>
```

Each worktree is independent — you can build in each one separately.

## Important Pitfalls (from CLAUDE.md)

1. **VarBuilder `.pp` calls must match PyTorch state_dict keys exactly.** Always check `model.safetensors.index.json`.
2. **Never use `Tensor::{from_vec,arange}` in hot loops** — causes CPU-GPU sync. Precompute at init.
3. **Feature flags**: Use `--features metal` on macOS, `--features "cuda flash-attn cudnn"` on Linux.
4. **Always `cargo check` before declaring done.** Code that doesn't compile is useless.
5. **Don't return TODOs.** Implement everything fully.
6. **For SSM/Mamba layers**: The recurrent state is fundamentally different from KV cache. Don't try to force it into the KV cache abstraction — manage it separately and adapt the cache interface minimally.
7. **For novel architectures**: Always start by reading the HuggingFace `modeling_*.py` file. The Python implementation IS the spec. Port it faithfully before optimizing.

## Priority Order

Priority is based on a combination of implementation difficulty and introspection experiment value:

1. **OLMo3** — Standard transformer, trivial to add, validates pipeline on a different model
2. **Falcon-H1** — Parallel SSM+attention is the most novel hybrid for introspection study
3. **RWKV-7** — Pure linear RNN, no attention at all — if introspection works here, it's a major finding
4. **Nemotron-H** — Mostly-Mamba, clean architecture, good size range
5. **Jamba 1.5** — Triple hybrid (SSM+attention+MoE), complex but very interesting
6. **Zamba2** — Shared attention layers, unique propagation dynamics
7. **AfMoE** — Custom MoE, interesting but MoE is less novel than SSM
8. **LFM2** — Needs most research, small model, conv layers may be hard to port
9. **Phi-4-mini-flash** — Novel GMU, but Microsoft's one-off architecture

## The Introspection Story

The end goal is a paper/blog post with this narrative:

> "We ran the introspection experiment across the full spectrum of modern LLM architectures:
> - Pure transformer (Qwen2.5-Coder-32B)
> - Hybrid GDN/MoE (Qwen3-Coder-Next)
> - Parallel SSM+attention (Falcon-H1)
> - Mostly-SSM with sparse attention (Nemotron-H)
> - Triple hybrid SSM+attention+MoE (Jamba)
> - Pure linear RNN with no attention (RWKV-7)
>
> Here's where self-awareness lives — and where it doesn't."

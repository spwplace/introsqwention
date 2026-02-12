# Handoff Notes: Introsqwention Research

## Current System State
- **Server**: Running on `localhost:3131`.
- **Model**: Qwen3-Coder-Next-MLX-4bit (Hybrid GDN/Attention).
- **Dashboard**: Fully functional for Logit Lens, Logit Diff, and Activation monitoring.
- **Stability**: Fixed three critical engine-level bugs; the system is now stable under heavy steering/training loads.

## Critical Bugs Fixed
1. **Causal Mask "Ghosting" (`past_kv_len` mismatch)**:
   - **Issue**: The `CausalMasker` was generating masks based on the KV cache length of the *previous* request. This happened because `kv_cache.reset()` was called inside the layer loop, but the mask was materialized before the loop.
   - **Fix**: Added `LocalHybridCache::reset()` and called it at the start of `forward` if `seqlen_offsets[0] == 0`.
2. **DType Inconsistency**:
   - **Issue**: PCA training produces `F32` vectors, but the model runs in `F16`. `candle` threw a shape/dtype mismatch on `broadcast_add`.
   - **Fix**: Implemented explicit casting to `model.dtype()` in `apply_steering_vectors` and added a safety check in the model's `forward` pass.
3. **Metal Kernel Panic**:
   - **Issue**: `called Result::unwrap() on an Err value` in `candle-metal-kernels`.
   - **Fix**: Identified that casting non-contiguous tensors to `F32` on Metal can trigger bitcode copy failures. Added `.contiguous()` calls before casting hidden states in the training loop.

## Observations & Hypotheses

### 1. The 4-Bit Quantization Wall
The model is highly sensitive to steering. At `scale=20`, we see conceptual "leakage" (e.g., "Cats" -> "ETS"), but by `scale=40`, the residual stream is completely corrupted into gibberish. 
- **Guess**: 4-bit quantization noise might be "smearing" the subtle directions found by PCA. A 16-bit or BF16 model would likely show much cleaner activation patterns.

### 2. GDN (Linear Attention) Dynamics
Qwen3-Next isn't a standard transformer; it interleaves GDN (Gated Delta Net) layers. 
- **Observation**: GDN layers maintain a recurrent state. Residual steering is applied *after* the GDN update.
- **Hypothesis**: The recurrent nature of GDN might make it more resistant to "instant" residual steering compared to the purely spatial attention mechanism. We may need to steer the GDN internal states (recurrent state) directly rather than just the residual stream.

### 3. PCA Inversion
- **Observation**: `measure_concept_activation` returned a strongly *negative* mean for relevant text.
- **Guess**: The PCA sign convention (making the max component positive) might be picking the "wrong" direction for the concept vs. the "anything" baseline. If activations are negative, try a negative scale (e.g., `-15.0`).

### 4. Training Data Sparsity
We are currently using the last-token hidden state for PCA. 
- **Guess**: For a 4-bit model, this might be too sparse. Averaging the difference across the last 3–5 tokens of the prompt might produce a vector that is more "center-of-mass" for the concept and less "noisy."

## Recommended Next Steps
- **Steer Middle Layers**: Stick to layers 16–32. Early layers are too close to embeddings; late layers are too close to the LM head.
- **Lower Scale**: Try scales in the 5–12 range to avoid the "gibberish" threshold.
- **Manual Logit Lens**: Use the `/` dashboard to look at the "I love you" pass. If you don't see the words "love" or "heart" appearing in any middle-layer logit lens, the steering vector hasn't "taken" yet.

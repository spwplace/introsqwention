# Handoff: Dual-Model Experiment on vast.ai 8x H200

## Machine Access
```
ssh -p 21036 root@208.64.254.74 -L 8080:localhost:8080 -i ~/.ssh/id_aws
```

## What's Running (tmux sessions)
- `qwen25` — Qwen2.5-Coder-32B server on port 8082, GPU 0, single device, WORKING
- `qwen3` — Qwen3-Coder-Next server on port 8081, GPUs 4-7, TP-4, BROKEN (see below)
- `track_a` — Qwen2.5 experiment script, RUNNING (training 8 concepts)
- `track_b` — Qwen3-Next experiment script, FINISHED (all failed due to TN layout error)

## Models
- Qwen2.5-Coder-32B: `/workspace/.hf_home/hub/models--Qwen--Qwen2.5-Coder-32B/snapshots/2e12b5f7bc878d424d222e224ed40aee564ec45f`
  - Architecture: `Qwen2ForCausalLM`, 64 layers, hidden_size=5120, dense
  - 62GB fp16, fits on single H200 (143GB)
- Qwen3-Coder-Next: `/workspace/.hf_home/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb`
  - Architecture: `Qwen3NextForCausalLM`, 48 layers, hidden_size=2048, 512 MoE experts (10 active)
  - 149GB fp16, needs at least 2-way TP

## BLOCKER: Qwen3-Coder-Next "TN layout" Error

Every forward pass on Qwen3-Next returns: `"This layer only supports TN layout"`

This is a CUBLAS GEMM error from candle — a matrix multiplication is receiving tensors with non-TN (transposed-normal) stride layout. This happens under TP-4 but we can't test single-GPU (149GB > 143GB H200).

**Likely cause**: The MoE `PackedExperts` layer or the GDN recurrent layers produce non-contiguous tensors after TP sharding. The RowParallelLayer's AllReduce output may not have the stride pattern CUBLAS expects.

**Where to look**:
1. `mistral-introspect/mistralrs-core/src/models/qwen3_next.rs` — the forward pass, especially `SparseMoeBlock` and `GDNBlock`
2. `mistral-introspect/mistralrs-quant/src/distributed/layers.rs` — `RowParallelLayer::forward()` and `PackedExperts` TP code
3. Search for `.contiguous()` calls — adding one before the failing matmul should fix it
4. The error comes from candle's `candle-core/src/cuda_backend/mod.rs` in the `gemm` function

**Debug approach**: Add `.contiguous()` before matmul operations in the MoE/GDN path, or trace which exact layer triggers the error by testing `forward` with individual layer capture (`"layers": [1]`, `"layers": [2]`, etc.).

**Alternative**: Try `--tp-size 2` with `CUDA_VISIBLE_DEVICES=0,1` to see if it's a TP-4 specific issue, though this requires ~75GB per GPU (fits in 143GB).

## Qwen2.5 Track A — In Progress

Track A is working. Current state as of handoff:
- Training: ~450/581 pairs for first concept (happiness), 7 more to go
- After training, will run: logit_diff → control_questions → top_of_mind → concept_identification → discrimination_matrix

Results go to `/tmp/results/qwen25/`. Script is `/tmp/track_a.sh`, log is `/tmp/track_a.log`.

Track A runs 8 concepts through the full vgel protocol:
```
happiness, sadness, anger, fear, honesty, deception, love, power
```

Each experiment:
- `logit_diff` — P(yes)/P(no) shift with steering (6 detection question variants, scale 8.0)
- `control_questions` — factual accuracy under steering (16 yes/no questions)
- `top_of_mind` — generation under steering (64 tokens)
- `concept_identification` — identify which concept is steered
- `discrimination_matrix` — 8x8 confusion matrix

**Default steering layers** for Qwen2.5 (64 layers): middle third = introspection layers 22-43.

## Database Setup
Each server has its own fresh SQLite:
- Qwen2.5: `data/qwen25.db`
- Qwen3-Next: `data/qwen3.db`
(The old shared `data/introsqwention.db` has stale Mac experiment data — don't use it)

## Server Binary
Built from the `dev` branch at commit `0cc2015`. Binary: `/workspace/introsqwention/target/release/introspect`
Built with `--features nccl` (CUDA + NCCL tensor parallelism).

Rebuild command: `cargo build -p introsqwention --features nccl --release`

## API Quick Reference

```bash
# Model info
curl localhost:8082/api/model_info

# Train steering vector (SSE stream)
curl -sN localhost:8082/api/train -H 'Content-Type: application/json' \
  -d '{"concept": "happiness"}'

# Logit diff detection
curl -s localhost:8082/api/run/logit_diff -H 'Content-Type: application/json' \
  -d '{"concept": "happiness", "scale": 8.0}'

# Control questions
curl -s localhost:8082/api/run/control_questions -H 'Content-Type: application/json' \
  -d '{"concept": "happiness", "scale": 8.0}'

# Top of mind generation
curl -s localhost:8082/api/run/top_of_mind -H 'Content-Type: application/json' \
  -d '{"concept": "happiness", "scale": 8.0, "max_tokens": 64}'

# Code logit diff (for code models)
curl -s localhost:8081/api/run/code_logit_diff -H 'Content-Type: application/json' \
  -d '{"concept": "happiness", "scale": 8.0}'

# Concept identification
curl -s localhost:8082/api/run/concept_identification -H 'Content-Type: application/json' \
  -d '{"concepts": ["happiness","sadness","anger","fear"], "scale": 8.0}'

# Discrimination matrix
curl -s localhost:8082/api/run/discrimination_matrix -H 'Content-Type: application/json' \
  -d '{"concepts": ["happiness","sadness","anger","fear"], "scale": 8.0}'

# Forward with hidden states
curl -s localhost:8082/api/forward -H 'Content-Type: application/json' \
  -d '{"text": "Hello world", "layers": [1, 32, 64]}'

# List trained vectors / experiments
curl -s localhost:8082/api/steering_vectors
curl -s localhost:8082/api/experiments
```

## Key Files on this Machine (Mac)
- Server code: `/Users/ember/dev/introsqwention/src/` (api.rs, experiments.rs, steering.rs, state.rs, main.rs)
- Model code: `/Users/ember/dev/introsqwention/mistral-introspect/mistralrs-core/src/` (introspection.rs, models/qwen3_next.rs)
- TP code: `src/distributed.rs`, `src/worker.rs`
- Previous findings: `data/qwen3-next-report.md`, `handoff-notes.md`
- MEMORY.md has project architecture notes

## What Needs Doing

1. **Fix Qwen3-Next TN layout error** — the main blocker. Debug the CUBLAS layout issue under TP.
2. **Monitor Track A** — Qwen2.5 experiments are running. Check `/tmp/track_a.log` and results in `/tmp/results/qwen25/`.
3. **Once Qwen3-Next is fixed**, re-run Track B: `/tmp/track_b.sh`
4. **Analyze results** — compare vgel detection on Qwen2.5-32B vs code-native experiments on Qwen3-Next.

## Success Metrics
| Experiment | Metric | Target |
|-----------|--------|--------|
| logit_diff | mean_yes_prob_shift | > 0.3 |
| control_questions | factual accuracy | ~1.0 |
| code_logit_diff | P(True)/P(False) shift | > 0.1 |
| concept_identification | correct concept rate | > 50% |
| discrimination_matrix | diagonal dominance | > 60% |

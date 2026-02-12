# Usage

## Building

```bash
# macOS (Metal)
cargo build --release --features metal

# Linux (CUDA)
cargo build --release --features cuda

# CPU only
cargo build --release
```

This produces two binaries in `target/release/`:
- `introspect` — the server (HTTP API + dashboard)
- `introspect-cli` — the CLI client

## Server

```bash
introspect -m <model-path-or-hf-id> [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | (required) | HuggingFace model ID or local path |
| `-p, --port` | 3131 | Port to serve on |
| `--host` | 127.0.0.1 | Host to bind to |
| `--dtype` | f16 | Weight dtype: `f16`, `bf16`, `f32` |
| `--static-dir` | static | Path to dashboard static files |

### Example

```bash
introspect -m Qwen/Qwen2.5-32B --dtype bf16
```

Once running:
- Dashboard: http://localhost:3131/
- API: http://localhost:3131/api/model_info

## CLI Client

```bash
introspect-cli [--server URL] <command>
```

The `--server` flag defaults to `http://127.0.0.1:3131`.

### Commands

#### Model inspection

```bash
# Model architecture info
introspect-cli info

# Tokenize text
introspect-cli tokenize "Hello world"

# Forward pass with logit lens (top-k predictions per layer)
introspect-cli forward "The meaning of life is" --top-k 10
```

#### Steering vectors

```bash
# Train a steering vector (streams progress)
introspect-cli train love
introspect-cli train love --suffixes 20              # quick test with 20 pairs
introspect-cli train love --layers 16,17,18,19,20    # specific layers only

# List trained vectors
introspect-cli vectors

# Apply to model (persists until cleared)
introspect-cli apply love --scale 8.0
introspect-cli apply love --scale 15 --layers 16,17,18,19,20

# Clear all steering vectors
introspect-cli clear
```

#### Experiments

```bash
# Introspection detection (logit diff)
introspect-cli run logit-diff love
introspect-cli run logit-diff love --variant no_info

# Control questions (factual accuracy under steering)
introspect-cli run control love --scale 8

# Logit lens comparison (base vs steered, per-layer)
introspect-cli run lens-compare love --tracked-tokens yes,no,love,cat

# Top-of-mind generation (what does the steered model say?)
introspect-cli run top-of-mind love --scale 12

# Concept activation measurement
introspect-cli run concept-activation love "I really enjoy cats"

# Full pipeline (train + all experiments)
introspect-cli run full love --suffixes 20 --scale 8
```

#### Hybrid architecture experiments

```bash
# Layer type lens (GDN vs attention confidence)
introspect-cli run layer-type-lens "The capital of France is"

# Steering survival through GDN layers
introspect-cli run steering-survival love --injection-layers 8,16,24,32

# CKA cross-layer similarity
introspect-cli run cka

# MoE routing analysis
introspect-cli run routing-analysis "The meaning of life is"
introspect-cli run routing-analysis "The meaning of life is" --concept love

# Causal tracing
introspect-cli run causal-tracing "The Eiffel Tower is in" "The Jfksl Qwert is in"

# GDN recurrent state statistics
introspect-cli run gdn-state-stats "The meaning of life is"
```

#### Experiment management

```bash
# List all experiments
introspect-cli experiments

# Get experiment details (JSON)
introspect-cli experiment <id>
```

### Full pipeline

`run full` executes the complete vgel introspection pipeline in sequence:

1. Train steering vector (with streaming progress bar)
2. Run logit diff
3. Run control questions
4. Run logit lens comparison
5. Run top-of-mind generation

```bash
introspect-cli run full love --suffixes 20 --scale 8
```

## HTTP API

All endpoints return JSON. Errors return `{"error": "..."}` with appropriate status codes.

### Read-only (GET)

| Endpoint | Description |
|----------|-------------|
| `GET /api/model_info` | Model architecture info |
| `GET /api/experiments` | List all experiments |
| `GET /api/experiments/{id}` | Get experiment by ID |
| `GET /api/steering_vectors` | List trained vectors |

### Core operations (POST)

| Endpoint | Body | Description |
|----------|------|-------------|
| `POST /api/tokenize` | `{text}` | Tokenize text |
| `POST /api/forward` | `{text, top_k?}` | Forward pass + logit lens |
| `POST /api/set_steering_vector` | `{layers, scale, vector}` | Set raw steering vector |
| `POST /api/clear_steering_vectors` | `{}` | Clear all steering vectors |
| `POST /api/train` | `{concept, num_suffixes?, layers?}` | Train steering vector (SSE) |
| `POST /api/apply_steering_vector` | `{name, layers?, scale?}` | Apply trained vector |

### Experiment runners (POST)

| Endpoint | Body |
|----------|------|
| `POST /api/run/logit_diff` | `{concept, scale?, user_turn1_variant?, layers?, top_k?}` |
| `POST /api/run/control_questions` | `{concept, scale?, user_turn1_variant?, layers?}` |
| `POST /api/run/logit_lens_comparison` | `{concept, scale?, tracked_tokens, user_turn1_variant?, layers?}` |
| `POST /api/run/top_of_mind` | `{concept, scale?, max_tokens?, temperature?, top_p?, layers?}` |
| `POST /api/run/concept_activation` | `{concept, text, layers?}` |
| `POST /api/run/layer_type_lens` | `{text}` |
| `POST /api/run/steering_survival` | `{concept, injection_layers, scale?, probe_text?}` |
| `POST /api/run/cka` | `{texts?}` |
| `POST /api/run/routing_analysis` | `{text, concept?, scale?, layers?}` |
| `POST /api/run/causal_tracing` | `{clean_text, corrupted_text}` |
| `POST /api/run/gdn_state_stats` | `{text}` |

All experiment endpoints save results and return `{experiment_id, result}`.

### Training SSE stream

`POST /api/train` returns `text/event-stream`:

```
event: progress
data: {"done": 5, "total": 20}

event: progress
data: {"done": 10, "total": 20}

event: complete
data: {"concept": "love", "num_pairs": 20, "num_layers": 48}
```

### curl examples

```bash
# Model info
curl localhost:3131/api/model_info

# Tokenize
curl -X POST localhost:3131/api/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello world"}'

# Forward pass
curl -X POST localhost:3131/api/forward \
  -H 'Content-Type: application/json' \
  -d '{"text": "The meaning of life is", "top_k": 5}'

# Train (SSE stream)
curl -N -X POST localhost:3131/api/train \
  -H 'Content-Type: application/json' \
  -d '{"concept": "love", "num_suffixes": 5}'

# Run logit diff
curl -X POST localhost:3131/api/run/logit_diff \
  -H 'Content-Type: application/json' \
  -d '{"concept": "love", "scale": 8.0}'
```

## Defaults

- **Steering layers**: middle third of the model (introspection indices). For a 48-layer model, layers 17..32.
- **Scale**: 8.0
- **Training suffixes**: all non-empty suffixes in the embedded dataset (currently 581 pairs). Use `--suffixes N` for faster iteration.
- **Logit diff variant**: `with_info` (gives the model context about introspection).
- **Top-of-mind**: temperature 0.7, top-p 0.9, max 128 tokens.

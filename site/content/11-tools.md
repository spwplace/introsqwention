+++
title = "Tools & Reproducibility"
weight = 11
description = "How to use the introspection toolkit, reproduce our experiments, and extend the work."

[extra]
num = "11"
+++

## The Introsqwention Toolkit

All experiments in this blog post were conducted using our open-source Rust-based introspection toolkit, built on top of [mistral.rs](https://github.com/EricLBuehler/mistral.rs).

### Installation

```bash
git clone https://github.com/spwplace/mistral-introspect
cd mistral-introspect

# macOS (Metal)
cargo build --release --features metal

# Linux (CUDA)
cargo build --release --features cuda
```

### Running the Server

```bash
# Start the introspection server with dashboard
cargo run --release --features metal -- \
    --model Qwen/Qwen3-Coder-Next-0.6B \
    --port 3131
```

This starts:
- **Dashboard** at `http://localhost:3131/` — live visualization of experiments
- **MCP endpoint** at `http://localhost:3131/mcp` — for programmatic control
- **JSON API** at `http://localhost:3131/api/` — raw data access

### MCP Tools

The server exposes these MCP tools:

| Tool | Description |
|------|-------------|
| `model_info` | Get model architecture details |
| `forward_introspect` | Run forward pass with hidden state capture |
| `train_steering_vector` | Train a contrastive PCA steering vector |
| `apply_steering` | Apply trained vectors to the model |
| `run_logit_diff` | Introspection detection experiment |
| `run_control_questions` | Knowledge corruption test |
| `run_logit_lens_comparison` | Per-layer token tracking |
| `run_top_of_mind` | Steered text generation |
| `run_concept_activation` | Per-layer concept activation measurement |

### Extending the Toolkit

The toolkit is designed to be extended. Key extension points:

- **New experiment types**: Add to `experiments.rs` and expose via `mcp.rs`
- **New model architectures**: Implement the `NormalModel` trait with introspection hooks
- **New visualization types**: Add chart builders to `static/js/charts.js`

## Reproducing Our Experiments

All experiments can be reproduced by connecting an MCP client to the server and running the tools in sequence. A typical workflow:

```
1. train_steering_vector(concept="love")
2. run_logit_diff(concept="love", variant="with_info")
3. run_control_questions(concept="love")
4. run_logit_lens_comparison(concept="love")
5. run_top_of_mind(concept="love")
6. run_concept_activation(concept="love", text="I really enjoy spending time with people I care about")
```

## Acknowledgments

- [vgel](https://vgel.me/) for the original introspection detection experiments
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) for the inference engine
- The mechanistic interpretability community for the theoretical foundations

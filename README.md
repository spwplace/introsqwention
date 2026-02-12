# Introsqwention

An activation-level introspection research platform for **Qwen-family** models. It provides an HTTP server, browser dashboard, and CLI for running introspection experiments inspired by work such as *"Small Models Can Introspect, Too"*.

## Features

- **Contrastive Steering Vector Training**: Train concept vectors (e.g., "honesty", "happiness", "i love you") using contrastive PCA on hidden state differences.
- **Logit Lens Dashboard**: Real-time visualization of what every layer in the model is "thinking" at the last token position.
- **Detection Experiments**: 
  - **Logit Diff**: Measure the shift in P(yes) vs P(no) when a concept is injected.
  - **Control Questions**: Verify that steering doesn't corrupt general factual knowledge.
  - **Comparative Logit Lens**: Track specific tokens (like "yes") across all layers to see *where* detection occurs.
- **Top of Mind**: Autoregressive generation under the influence of steering vectors.
- **Concept Activation**: Monitor how aligned a specific text input is with a trained concept across all layers.

## Getting Started

### Prerequisites

- A local or HuggingFace-hosted Qwen-family model supported by `mistral-introspect` (for example `Qwen/Qwen2.5-32B`).
- Rust toolchain.

### Running the Server

```bash
cargo run --release --features metal -- -m /path/to/model --port 3131
```

Access the dashboard at `http://localhost:3131/`.

## API + CLI Surface

The server exposes HTTP endpoints (see `USAGE.md`) and the `introspect-cli` client:

- Train steering vectors with streaming progress (`train` / `POST /api/train`).
- Run core experiments (`logit_diff`, `control_questions`, `logit_lens_comparison`, `top_of_mind`, etc.).
- Inspect model internals (`forward`, layer-type lens, CKA, routing, causal tracing, GDN state stats).

## Architecture

The model uses a hybrid architecture with **GDN (Gated Delta Net)** linear attention layers interleaved with standard full attention layers. This project includes a specialized fork of `mistral.rs` (in `mistral-introspect/`) to support introspection features for this specific architecture.

## License

MIT

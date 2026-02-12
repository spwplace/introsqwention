+++
title = "The Introspection Toolkit"
weight = 2
description = "Architecture of our Rust-based interpretability toolkit: hidden state capture, steering vectors, logit lens, and the MCP server."

[extra]
num = "02"
+++

## Overview

Our toolkit is built in Rust as a custom inference harness on top of [mistral.rs](https://github.com/EricLBuehler/mistral.rs). Rather than using Python hooks (like TransformerLens), we instrument the forward pass directly in compiled code, giving us:

- **Zero-copy hidden state capture** at any layer
- **Steering vector injection** via broadcast addition to the residual stream
- **Batched logit lens** that projects all captured layers through the LM head in one pass
- **An MCP (Model Context Protocol) server** for programmatic experiment control
- **A live dashboard** for real-time visualization

## Architecture

The toolkit consists of three main components:

### IntrospectionModel

A wrapper around the model backend that exposes `forward_introspect()` — a modified forward pass that clones the residual stream tensor after each decoder layer. This capture is controlled by an `IntrospectionState` struct that lives inside the model and can be toggled on/off with selective layer filtering.

### Steering Vector Training

Contrastive PCA using non-empty suffix pairs from the embedded dataset. For each pair, we run a forward pass through the full conversation template, extract the last-token hidden state at each layer, compute the difference between concept-positive and concept-negative activations, and extract the top principal component via power iteration. The result is a per-layer unit direction vector.

### MCP Server

All experiments are exposed as MCP tools, making the toolkit controllable from any MCP client (Claude, cursor, etc.). The server uses axum with rmcp's StreamableHTTP transport. Experiment results are stored in shared state and rendered on the dashboard.

<div class="chart-placeholder">Chart: Toolkit architecture diagram — model → IntrospectionState → capture → logit lens / steering / probes</div>

## Key Design Decisions

- **Rust over Python**: The forward pass instrumentation adds minimal overhead because it's compiled into the same binary as the model. No Python-Rust FFI boundary for each hook call.
- **Batched logit lens**: Instead of projecting one layer at a time, we stack all captured hidden states and project them through the final norm + lm_head in a single batched operation.
- **Per-layer steering**: Steering vectors are stored per-layer and injected in the residual stream *after* each targeted layer's output. This allows precise control over which layer types receive the steering signal.

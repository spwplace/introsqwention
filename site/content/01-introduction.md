+++
title = "Introduction"
weight = 1
description = "Why study a hybrid GDN + full attention + MoE architecture, and what we hope to learn."

[extra]
num = "01"
+++

## Motivation

Qwen3-Next represents a new breed of language model: a *hybrid* architecture that interleaves Gated Delta Net (GDN) linear attention layers with traditional full attention layers, and routes every layer's MLP through a Sparse Mixture of Experts. This combination is architecturally novel, and the interpretability community has not yet studied it.

We started from a replication of [vgel's "Small Models Can Introspect, Too"](https://vgel.me/posts/qwen-introspection/) and asked: what does this architecture look like on the *inside*?

## What This Blog Post Covers

This is a record of mechanistic interpretability experiments on **Qwen3-Coder-Next 0.6B** (48 layers, 1024d hidden, 64 experts with top-8 routing). We built a Rust-based introspection toolkit on top of [mistral.rs](https://github.com/EricLBuehler/mistral.rs) that hooks directly into the forward pass, capturing hidden states, steering internal representations, and analyzing the geometry of what the model learns.

The sections that follow cover:
- The introspection toolkit and how it works
- Replicating vgel's introspection detection experiments on this architecture
- How GDN layers differ from full attention layers in their representational geometry
- MoE routing patterns and what experts specialize in
- Optimal steering vector placement across layer types
- The "assistant axis" — first principal component of persona space
- Causal tracing to identify which layers are causally necessary
- Linear probing for concept detection

<div class="chart-placeholder">Chart: Architecture overview diagram showing 48 layers with GDN/Attn alternation and MoE routing</div>

## Key Questions

- Does the logit lens work the same way through GDN layers as through full attention layers?
- Do steering vectors need to target specific layer types (GDN vs. attention) to be effective?
- How do MoE routing patterns change under steering?
- Can we identify "concept circuits" that span both layer types?
- What is the representational geometry of this hybrid architecture?

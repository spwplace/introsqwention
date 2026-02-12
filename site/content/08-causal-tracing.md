+++
title = "Causal Tracing"
weight = 8
description = "Activation patching to identify which layers are causally necessary for introspection detection."

[extra]
num = "08"
+++

## From Correlation to Causation

The logit lens and concept activation measurements tell us *where* information appears, but not whether those representations are *causally necessary* for the model's behavior. Causal tracing (activation patching) replaces individual layers' activations to measure their causal contribution.

## Methodology

For each layer L:
1. Run a **clean forward pass** (with steering) and record all hidden states
2. Run a **corrupted forward pass** (without steering) up to layer L
3. **Patch**: replace layer L's hidden state from the clean run into the corrupted run
4. Continue the forward pass and measure the output change (detection P(yes))

A large recovery in P(yes) after patching means layer L carries causally important information for introspection detection.

<div class="chart-placeholder">Chart: Causal tracing heatmap — per-layer restoration of detection signal, colored by layer type</div>

## Layer Type Analysis

<div class="chart-placeholder">Chart: Mean causal importance by layer type (GDN vs. Attn)</div>

## Interaction Between Layer Types

We also test patching *pairs* of layers to look for interactions:
- Does patching a GDN layer *and* the subsequent attention layer produce a superlinear effect?
- Are there critical "GDN → Attn" transitions where information is transferred between mechanisms?

## Noise-Based Causal Tracing

As an alternative to clean/corrupted swap, we also measure the effect of adding Gaussian noise to individual layers' hidden states — a less binary measure of causal importance.

<div class="chart-placeholder">Chart: Noise sensitivity per layer — how much does output change when we perturb each layer?</div>

## Discussion

- Are the causally important layers the same ones where the logit lens shows concept emergence?
- Do GDN and attention layers have different causal roles?
- Is there a "critical path" through the network for introspection?

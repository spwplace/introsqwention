+++
title = "Hybrid Geometry"
weight = 4
description = "CKA analysis and representational geometry across GDN and full attention layers."

[extra]
num = "04"
+++

## The Hybrid Question

Qwen3-Next alternates between two fundamentally different attention mechanisms: Gated Delta Net (O(n) recurrence) and full quadratic attention (O(n^2) SDPA). A natural question is: do these layer types produce similar or different representations?

## Centered Kernel Alignment (CKA)

CKA measures the similarity between representations at different layers, producing a layer-by-layer similarity matrix. High CKA between two layers means they encode similar information; low CKA means they've transformed the representation significantly.

We compute CKA between all pairs of the 48 layers' hidden states across a diverse set of prompts, producing a 48x48 heatmap.

<div class="chart-placeholder">Chart: CKA heatmap (48x48) with GDN/Attn layer type annotations</div>

## Hypotheses

- **Block structure**: We expect to see 4-layer blocks (3 GDN + 1 Attn) appearing as high-similarity clusters, since the GDN layers within a block might perform incremental refinement while the attention layer performs a global recomputation.
- **Phase transitions**: Sharp drops in CKA might mark "representational phase transitions" — points where the model fundamentally reorganizes its representation.
- **GDN vs. Attn asymmetry**: Full attention layers, having access to the complete context, might show higher cross-layer similarity (they all see the same tokens) while GDN layers, limited to recurrent state, might show more progressive change.

## Results

<div class="chart-placeholder">Chart: CKA with layer type color-coding on axes</div>

## Representational Drift

We also measure how the cosine similarity between adjacent layers' hidden states changes across depth. This "representational velocity" tells us how rapidly the model is transforming its representations at each point.

<div class="chart-placeholder">Chart: Cosine similarity between adjacent layers, colored by layer type</div>

## Discussion

- Do GDN layers show more or less representational change per layer than attention layers?
- Where are the phase transitions, and do they align with architectural boundaries?
- What does this tell us about how information flows through the hybrid architecture?

+++
title = "Optimal Steering"
weight = 6
description = "Finding the best layers and scales for steering vector injection across GDN and attention layer types."

[extra]
num = "06"
+++

## The Steering Placement Question

With two layer types available, a natural question is: should steering vectors target GDN layers, attention layers, or both? And at what scale?

## Methodology

We perform a grid search over:
- **Layer selections**: GDN-only, Attn-only, both, and various contiguous ranges
- **Scales**: 1, 5, 10, 15, 20, 30, 50
- **Metrics**: Introspection detection (logit diff), control question corruption, generation quality

For each combination, we measure the tradeoff between detection strength and knowledge corruption.

<div class="chart-placeholder">Chart: Scale vs. detection P(yes) for GDN-only, Attn-only, and both</div>

## Layer Type Ablation

<div class="chart-placeholder">Chart: Heatmap of detection strength per layer, grouped by type</div>

## The Pareto Frontier

We plot detection strength against control question corruption to find the Pareto-optimal steering configurations — maximum signal with minimum collateral damage.

<div class="chart-placeholder">Chart: Pareto frontier — detection strength vs. mean control question shift</div>

## Mean-of-Differences vs. PCA

Recent work (Jan 2026) proves that mean-of-differences is provably optimal for extracting linear concept directions when the concept subspace is one-dimensional. We compare our contrastive PCA vectors against mean-of-differences vectors:

<div class="chart-placeholder">Chart: Detection strength comparison — PCA vs. mean-diff vectors at various scales</div>

## Discussion

- Which layer type is more effective for steering — and does this match the CKA geometry?
- Is there a "sweet spot" in the middle layers, as vgel found for Qwen2.5?
- Does the optimal steering configuration differ between GDN and attention layers?
- How does mean-of-differences compare to PCA in practice on this architecture?

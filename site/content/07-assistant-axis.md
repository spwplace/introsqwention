+++
title = "The Assistant Axis"
weight = 7
description = "Finding the first principal component of persona space and its relationship to introspection."

[extra]
num = "07"
+++

## Background

Recent work (Jan 2026) identified the "assistant axis" — the first principal component of the subspace spanned by different persona steering vectors. This axis appears to separate "assistant-like" behavior from "base model" behavior, and is geometrically central to how language models organize their internal representations of identity.

## Methodology

We train steering vectors for multiple persona-related concepts:
- Introspection / self-awareness
- Helpfulness
- Honesty
- Confidence
- Creativity
- Various named personas

For each, we extract the per-layer direction vector via contrastive PCA. We then perform PCA across this collection of vectors to find the principal components of "persona space."

<div class="chart-placeholder">Chart: Persona space PCA — 2D projection of concept vectors, colored by category</div>

## The First Principal Component

<div class="chart-placeholder">Chart: Variance explained by top-k principal components</div>

<div class="chart-placeholder">Chart: Projection of all concept vectors onto PC1 — the "assistant axis"</div>

## Per-Layer Analysis

Does the assistant axis look the same at every layer, or does it emerge at specific depths?

<div class="chart-placeholder">Chart: Cosine similarity of per-layer PC1 across layers</div>

## Relationship to Introspection Detection

We test whether the assistant axis direction has any special relationship to the model's ability to detect steering — i.e., does the introspection detection signal align with the assistant axis?

## Discussion

- Is there a single dominant axis in persona space for this architecture?
- Does the axis emerge primarily in GDN or attention layers?
- What does projection onto the assistant axis tell us about the model's "self-model"?

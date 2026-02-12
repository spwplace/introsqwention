+++
title = "Linear Probing"
weight = 9
description = "Training linear probes to detect concept presence from hidden states, and comparing probe accuracy across layer types."

[extra]
num = "09"
+++

## What Probes Tell Us

A linear probe is a simple classifier (logistic regression) trained to predict whether a concept is present from a layer's hidden state. If a linear probe achieves high accuracy at layer L, it means the concept is *linearly decodable* from that layer's representation — the information is present and accessible.

## Methodology

1. Generate a balanced dataset of hidden states: forward passes with and without a concept steering vector active
2. For each layer, train a logistic regression classifier on the hidden state vectors
3. Measure accuracy, AUC, and probe weight alignment with the steering vector direction

<div class="chart-placeholder">Chart: Probe accuracy per layer, colored by layer type</div>

## Probe Alignment

A key question: does the probe's learned weight vector align with the steering vector direction? If so, the probe is detecting the same feature that steering injected. If not, the concept might be encoded differently than the steering vector represents.

<div class="chart-placeholder">Chart: Cosine similarity between probe weights and steering direction, per layer</div>

## Layer Type Comparison

<div class="chart-placeholder">Chart: Mean probe accuracy at GDN layers vs. attention layers</div>

## Multi-Concept Probing

We train probes for multiple concepts and examine:
- Whether probes for different concepts activate at the same or different layers
- Whether probe accuracy correlates with steering effectiveness at that layer
- Whether GDN layers are better or worse at encoding concepts linearly

## Discussion

- At what depth does concept information become linearly decodable?
- Do GDN layers encode concepts differently (less linearly?) than attention layers?
- Is there a relationship between probe accuracy and causal importance (from Section 8)?

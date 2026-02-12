+++
title = "Expert Routing Analysis"
weight = 5
description = "MoE routing patterns: which experts activate, how routing changes under steering, and expert specialization."

[extra]
num = "05"
+++

## The MoE Routing Question

Every layer in Qwen3-Next routes through a Sparse Mixture of Experts with 64 total experts and top-8 routing. This means each token uses 12.5% of the expert capacity. We capture the router logits to analyze routing patterns.

## Capturing Router Logits

We added hooks to the `SparseMoeBlock::forward()` to capture:
- **Router logits**: Raw scores for all 64 experts before softmax/top-k
- **Selected experts**: Which 8 experts were chosen
- **Routing weights**: Softmax probabilities for the selected experts
- **Shared expert gate**: The sigmoid gate value for the always-on shared expert

## Expert Specialization

By aggregating routing patterns across a diverse corpus, we can identify which experts specialize in which functions:

<div class="chart-placeholder">Chart: Expert activation heatmap — experts (rows) vs. token categories (columns)</div>

### Key Questions

- Do certain experts specialize in syntactic vs. semantic content?
- Are there "universal" experts that fire frequently regardless of content?
- Do expert routing patterns differ between GDN and attention layers?

## Routing Under Steering

When we inject a steering vector, does the routing pattern change? If concept information is partially encoded in expert selection, we should see systematic routing shifts.

<div class="chart-placeholder">Chart: Expert routing diff — base vs. steered, per layer</div>

<div class="chart-placeholder">Chart: Top affected experts by routing probability shift</div>

## Shared Expert Analysis

The shared expert always fires with a learned sigmoid gate. We track the gate value across layers and prompts:

<div class="chart-placeholder">Chart: Shared expert gate values across layers, base vs. steered</div>

## Discussion

- Does steering primarily affect routing in GDN layers, attention layers, or both equally?
- Can we identify "concept experts" — experts whose activation strongly correlates with steering?
- What is the relationship between shared expert gating and steering vector effectiveness?

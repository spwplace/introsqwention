+++
title = "Replicating vgel's Experiments"
weight = 3
description = "Introspection detection via logit diff, control question validation, and logit lens comparison on Qwen3-Next."

[extra]
num = "03"
+++

## Background

[vgel's "Small Models Can Introspect, Too"](https://vgel.me/posts/qwen-introspection/) demonstrated that Qwen2.5-Coder 0.5B could detect when its internal representations had been steered — the model's P(yes) increased at the "The answer is" position when asked "Are you thinking about [concept]?" with the concept's steering vector active. We replicate this protocol on Qwen3-Next.

## Protocol

1. **Train a steering vector** for a concept (e.g., "love") using contrastive PCA over non-empty suffix pairs from the embedded dataset
2. **Run a base forward pass** on the detection conversation (3-turn ChatML) and record P(yes), P(no) at the final "The answer is" position
3. **Run a steered forward pass** with the concept vector injected at target layers, record the same probabilities
4. **Compare**: does P(yes) increase under steering?

We test three prompt variants:
- **no_info**: "Are you thinking about anything in particular right now?"
- **with_info**: "Are you thinking about [concept] right now?"
- **inaccurate_info**: "Are you thinking about [wrong concept] right now?"

## Results

<div class="chart-placeholder">Chart: Logit diff bar chart — P(yes) and P(no) base vs. steered for each variant</div>

<div class="chart-placeholder">Chart: Top tokens bar chart — base vs. steered probability distribution</div>

## Control Questions

To ensure the steering vector doesn't corrupt general knowledge, we run 16 factual yes/no questions (e.g., "Is the Earth round?", "Does water boil at 100C at sea level?") with and without steering, measuring P(yes)/P(no) shifts.

<div class="chart-placeholder">Chart: Control questions delta chart — per-question P(yes) and P(no) shifts</div>

## Logit Lens Comparison

We track specific tokens (yes, no, Yes, No, love, etc.) across all 48 layers, comparing base and steered forward passes. This reveals *where* in the network the model starts recognizing the injected concept.

<div class="chart-placeholder">Chart: Logit lens comparison — multi-line plot, base (dotted) vs. steered (solid) per tracked token</div>

## Discussion

Key questions:
- Does the detection signal emerge at GDN layers or full attention layers?
- Is the effect size comparable to the Qwen2.5-Coder result?
- Does the "with_info" variant show a stronger effect, as in the original?

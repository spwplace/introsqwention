+++
title = "Discussion"
weight = 10
description = "Synthesis of findings, implications for hybrid architectures, and open questions."

[extra]
num = "10"
+++

## Synthesis

This section brings together findings from all preceding experiments to paint a coherent picture of how Qwen3-Next processes and represents information.

## The Hybrid Architecture Story

What did we learn about how GDN and full attention layers collaborate?

- **Representational geometry**: How the CKA analysis reveals the division of labor
- **Information flow**: Where concepts emerge, as revealed by logit lens and probes
- **Causal structure**: Which layers and layer types are necessary, from causal tracing
- **Expert routing**: How MoE routing relates to concept processing

## Implications for Interpretability

How do our findings relate to the broader mechanistic interpretability research program?

- **Transferability**: To what extent do transformer interpretability techniques transfer to hybrid architectures?
- **New phenomena**: What novel phenomena arise from the GDN + Attn + MoE combination?
- **Tool development**: What interpretability tools do we need that don't yet exist?

## Prior Art Revisited

How do our findings compare to:
- IOI circuits in Mamba (ICLR 2025)
- "Does Transformer Interpretability Transfer to RNNs?" (AAAI 2025)
- Hidden attention matrices in Mamba
- Anthropic's circuit tracing work

## Limitations

- We study a 0.6B model — findings may not generalize to larger scales
- Contrastive PCA assumes a one-dimensional concept subspace
- Our causal tracing methodology has known limitations

## Open Questions

- Can SAEs be effectively trained on GDN layer activations?
- Do Patchscopes (model self-interpretation) work through GDN layers?
- How does the Delta Rule recurrence interact with steering?
- What happens in larger Qwen3-Next models?

// Main dashboard page — fetches all API data and populates the DOM.

function esc(s) {
    const d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
}

// ── Data Loading ────────────────────────────────────────────────────

async function loadDashboard() {
    const [infoResp, expsResp, vecsResp] = await Promise.all([
        fetch('/api/model_info'),
        fetch('/api/experiments'),
        fetch('/api/steering_vectors'),
    ]);

    const info = await infoResp.json();
    const exps = await expsResp.json();
    const vecs = await vecsResp.json();

    renderHeader(info);
    renderArchitecture(info);
    renderExperiments(exps);
    renderSteeringVectors(vecs);
    renderArchExplorer(info);
    setupEffects();
}

// ── Header ──────────────────────────────────────────────────────────

function renderHeader(info) {
    const modelName = { qwen2: 'Qwen2.5-Coder', qwen3_next: 'Qwen3-Coder-Next' }[info.model_type] || info.model_type;
    const expertTag = info.num_experts > 0 ? ` / ${info.num_experts}E` : '';

    document.querySelector('h1').textContent = 'Introspect';
    document.title = `Introspect \u2014 ${modelName}`;
    document.getElementById('header-meta').innerHTML =
        `<span data-tag="model">${esc(modelName)}</span><span data-sep>&middot;</span><span data-dim>${info.num_layers}L / ${info.hidden_size}d${expertTag}</span>`;
    document.getElementById('mcp-endpoint').textContent = `:${location.port || 80}/mcp`;
}

// ── Architecture Strip + Stats ──────────────────────────────────────

function renderArchitecture(info) {
    const section = document.getElementById('architecture-section');

    // Layer strip
    let layerBlocks = '';
    for (let i = 0; i < info.layer_types.length; i++) {
        const lt = info.layer_types[i];
        const label = ((i + 1) % 4 === 0 || i === 0) ? `<span data-layer-label>${i}</span>` : '';
        layerBlocks += `<div data-layer-type="${lt}" data-layer-idx="${i}" title="Layer ${i}: ${lt}" style="animation-delay:${i * 15}ms">${label}</div>`;
    }

    const gdnLegend = info.full_attention_interval > 1
        ? '<span data-legend-item><span data-legend-swatch="gdn"></span>Gated Delta Net</span>'
        : '';

    let expertsStat = '';
    if (info.num_experts > 0) {
        expertsStat = `<div data-stat><span data-stat-value>${info.num_experts}<small>/${info.num_experts_per_tok}</small></span><span data-stat-label>Experts / Active</span></div>`;
    }

    section.innerHTML = `
        <div data-section-header>
            <h2>Architecture</h2>
            <div data-legend>
                <span data-legend-item><span data-legend-swatch="attn"></span>Full Attention</span>
                ${gdnLegend}
            </div>
        </div>
        <div data-layer-strip>${layerBlocks}</div>
        <div data-arch-stats>
            <div data-stat><span data-stat-value>${info.num_layers}</span><span data-stat-label>Layers</span></div>
            <div data-stat><span data-stat-value>${info.hidden_size}</span><span data-stat-label>Hidden Dim</span></div>
            <div data-stat><span data-stat-value>${info.num_attention_heads}</span><span data-stat-label>Attn Heads</span></div>
            <div data-stat><span data-stat-value>${info.num_kv_heads}</span><span data-stat-label>KV Heads</span></div>
            ${expertsStat}
            <div data-stat><span data-stat-value>${Math.floor(info.vocab_size / 1000)}k</span><span data-stat-label>Vocab</span></div>
        </div>`;
}

// ── Experiments Table ───────────────────────────────────────────────

function renderExperiments(exps) {
    const sorted = exps.slice().sort((a, b) => b.created_at.localeCompare(a.created_at));
    const tbody = document.getElementById('exp-rows');
    document.getElementById('exp-count').textContent = sorted.length;

    if (sorted.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" data-empty>
            <div data-empty-icon>&#x2261;</div>
            <div>No experiments yet</div>
            <div data-empty-hint>Connect an MCP client to begin</div>
        </td></tr>`;
        return;
    }

    tbody.innerHTML = sorted.map((exp, i) => `
        <tr data-anim-delay="${i * 40}" style="animation-delay:${i * 40}ms">
            <td><a href="/experiment.html?id=${encodeURIComponent(exp.id)}">${esc(exp.id.substring(0, 8))}</a></td>
            <td data-role="name">${esc(exp.name)}</td>
            <td><span data-status="${exp.status}">${exp.status}</span></td>
            <td><span data-tag="type">${exp.config.experiment_type}</span></td>
            <td data-role="prompt">${esc(exp.config.prompt.substring(0, 60))}</td>
        </tr>`).join('');
}

// ── Steering Vectors Table ──────────────────────────────────────────

function renderSteeringVectors(vecs) {
    const tbody = document.getElementById('vec-rows');
    document.getElementById('vec-count').textContent = vecs.length;

    if (vecs.length === 0) {
        tbody.innerHTML = `<tr><td colspan="4" data-empty>
            <div data-empty-icon>&#x2300;</div>
            <div>No steering vectors trained</div>
            <div data-empty-hint>Use train_steering_vector via MCP</div>
        </td></tr>`;
        return;
    }

    tbody.innerHTML = vecs.map((svec, i) => `
        <tr data-anim-delay="${i * 40}" style="animation-delay:${i * 40}ms">
            <td data-role="vec-name">${esc(svec.name)}</td>
            <td>${esc(svec.concept)}</td>
            <td data-mono>${Object.keys(svec.vectors).length}</td>
            <td data-mono>${svec.num_training_pairs}</td>
        </tr>`).join('');
}

// ── Architecture Explorer ───────────────────────────────────────────

function renderArchExplorer(info) {
    const mount = document.getElementById('arch-explorer-mount');
    if (info.full_attention_interval > 1) {
        mount.innerHTML = buildHybridArchExplorer(info);
    } else {
        mount.innerHTML = buildStandardArchExplorer(info);
    }
}

function buildHybridArchExplorer(info) {
    const keyDim = info.linear_num_key_heads * info.linear_key_head_dim;
    const valueDim = info.linear_num_value_heads * info.linear_value_head_dim;
    const convDim = keyDim * 2 + valueDim;
    const qkvzDim = keyDim * 2 + valueDim * 2;
    const rotDim = Math.floor(info.head_dim * 0.25);
    const passDim = info.head_dim - rotDim;
    const fullAttnCount = info.layer_types.filter(t => t === 'full_attention').length;
    const gdnCount = info.num_layers - fullAttnCount;
    const gqaRatio = info.num_attention_heads / info.num_kv_heads;
    const qOutDim = info.num_attention_heads * info.head_dim;
    const expertPct = ((info.num_experts_per_tok / info.num_experts) * 100).toFixed(1);
    const baDim = info.linear_num_value_heads * 2;

    return `
    <section data-component="arch-explorer" id="arch-explorer">
        <div data-section-header>
            <h2>Model Architecture Explorer</h2>
            <span data-dim>An educational walkthrough of the complete forward pass</span>
        </div>

        <div data-pipeline>
            <!-- Embedding -->
            <div data-pipe-node data-component="embedding">
                <div data-pipe-node-inner>
                    <div data-pipe-icon>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="1.5">
                            <rect x="3" y="3" width="18" height="18" rx="3"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/>
                            <line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/>
                        </svg>
                    </div>
                    <div data-pipe-content>
                        <h3>Token Embedding</h3>
                        <div data-pipe-shape>${info.vocab_size} &rarr; ${info.hidden_size}d</div>
                    </div>
                </div>
                <div data-pipe-annotation>
                    Each token ID is looked up in an embedding table of <strong>${info.vocab_size}</strong> vectors, each <strong>${info.hidden_size}</strong>-dimensional.
                    This is the model&rsquo;s first transformation: from discrete symbols into a continuous geometric space where meaning can be computed on.
                </div>
            </div>

            <div data-pipe-flow><div data-flow-line></div><span data-flow-label>(batch, seq, ${info.hidden_size})</span></div>

            <!-- Decoder Stack Intro -->
            <div data-pipe-stack-header>
                <div data-pipe-stack-title>
                    Decoder Stack
                    <span data-pipe-stack-count>&times; ${info.num_layers} layers</span>
                </div>
                <div data-pipe-stack-breakdown>
                    <span data-tag="gdn-count">${gdnCount} GDN</span>
                    <span data-tag="attn-count">${fullAttnCount} Full Attention</span>
                    <span data-dim>repeating pattern: 3 GDN &rarr; 1 Full Attn</span>
                </div>
            </div>

            <!-- GDN Layer Detail -->
            <div data-pipe-layer data-layer-type="gdn" data-expandable data-expanded>
                <div data-layer-banner>
                    <span data-layer-badge="gdn">GDN</span>
                    <h3>Gated Delta Net &mdash; Linear Attention</h3>
                    <span data-dim>Layers 0, 1, 2, 4, 5, 6, &hellip; (${gdnCount} of ${info.num_layers})</span>
                    <button data-expand-toggle onclick="this.closest('[data-expandable]').toggleAttribute('data-expanded')">
                        <span data-expand-open>&#x25B4; collapse</span>
                        <span data-expand-closed>&#x25BE; expand</span>
                    </button>
                </div>

                <div data-layer-body>
                    <div data-layer-flow>
                        <div data-flow-block data-component="norm">
                            <div data-block-title>Input RMSNorm</div>
                            <div data-block-detail>Normalize activation magnitude while preserving direction. RMSNorm is simpler than LayerNorm &mdash; no mean subtraction, just divide by root-mean-square.</div>
                        </div>
                        <div data-flow-arrow></div>
                        <div data-flow-parallel>
                            <div data-flow-block data-component="proj">
                                <div data-block-title>QKVZ Projection</div>
                                <div data-block-shape>${info.hidden_size} &rarr; ${qkvzDim}</div>
                                <div data-block-detail>Single linear layer projects input into four subspaces: <strong>Query</strong> (where to look), <strong>Key</strong> (what to match), <strong>Value</strong> (what to read), <strong>Z-gate</strong> (output gating).</div>
                            </div>
                            <div data-flow-block data-component="proj">
                                <div data-block-title>&beta;&alpha; Projection</div>
                                <div data-block-shape>${info.hidden_size} &rarr; ${baDim}</div>
                                <div data-block-detail>Projects to per-head <strong>&beta;</strong> (write strength, sigmoidified) and <strong>&alpha;</strong> (state decay rate, through softplus). Controls how aggressively the recurrent state is updated.</div>
                            </div>
                        </div>
                        <div data-flow-arrow></div>
                        <div data-flow-block data-component="conv">
                            <div data-block-title>Causal Conv1D</div>
                            <div data-block-shape>kernel=${info.linear_conv_kernel_dim}, dim=${convDim}</div>
                            <div data-block-detail>A short causal convolution over the Q, K, V channels. This gives each position a local context window of ${info.linear_conv_kernel_dim} tokens <em>before</em> the recurrence. Followed by SiLU activation. Maintains a rolling state for autoregressive decode.</div>
                        </div>
                        <div data-flow-arrow></div>
                        <div data-flow-block data-component="recurrence" data-highlight>
                            <div data-block-title>&#x25C8; Delta Rule Recurrence</div>
                            <div data-block-shape>state: (${info.linear_num_value_heads}, ${info.linear_key_head_dim}, ${info.linear_value_head_dim})</div>
                            <div data-block-detail>
                                The core innovation. Instead of O(n&sup2;) attention, GDN maintains a fixed-size matrix <strong>S</strong> per head:
                                <pre data-code>for each token t:
  S &larr; S &middot; decay(t)           // forget old info
  &delta; &larr; (v(t) - S&middot;k(t)) &middot; &beta;(t)  // error signal
  S &larr; S + k(t) &otimes; &delta;         // Hebbian update
  y(t) &larr; S &middot; q(t)             // read with query</pre>
                                This is <strong>O(n)</strong> in sequence length &mdash; each token updates and reads from the state in constant time. The &ldquo;delta rule&rdquo; name comes from the error-correcting update: the value is compared against what the state <em>already</em> associates with that key.
                            </div>
                        </div>
                        <div data-flow-arrow></div>
                        <div data-flow-block data-component="norm">
                            <div data-block-title>Gated RMSNorm</div>
                            <div data-block-detail>Output is normalized, then element-wise multiplied by SiLU(z-gate). The z-gate learned earlier controls what information passes through. shape: (${info.linear_num_value_heads}, ${info.linear_value_head_dim}) &rarr; reshape to ${valueDim}d</div>
                        </div>
                        <div data-flow-arrow></div>
                        <div data-flow-block data-component="proj">
                            <div data-block-title>Output Projection</div>
                            <div data-block-shape>${valueDim} &rarr; ${info.hidden_size}</div>
                        </div>
                    </div>
                    <div data-residual-add>+ residual</div>
                    <div data-layer-flow>
                        <div data-flow-block data-component="norm">
                            <div data-block-title>Post-Attention RMSNorm</div>
                        </div>
                        <div data-flow-arrow></div>
                        <div data-flow-block data-component="moe" data-highlight>
                            <div data-block-title>&#x25C8; Sparse MoE</div>
                            <div data-block-detail>Detailed below</div>
                        </div>
                    </div>
                    <div data-residual-add>+ residual</div>
                </div>
            </div>

            <div data-pipe-flow><div data-flow-line></div></div>

            <!-- Full Attention Layer Detail -->
            <div data-pipe-layer data-layer-type="attn" data-expandable>
                <div data-layer-banner>
                    <span data-layer-badge="attn">ATTN</span>
                    <h3>Full Attention &mdash; Grouped Query Attention</h3>
                    <span data-dim>Every 4th layer: 3, 7, 11, &hellip; 47 (${fullAttnCount} of ${info.num_layers})</span>
                    <button data-expand-toggle onclick="this.closest('[data-expandable]').toggleAttribute('data-expanded')">
                        <span data-expand-open>&#x25B4; collapse</span>
                        <span data-expand-closed>&#x25BE; expand</span>
                    </button>
                </div>

                <div data-layer-body>
                    <div data-layer-flow>
                        <div data-flow-block data-component="norm">
                            <div data-block-title>Input RMSNorm</div>
                        </div>
                        <div data-flow-arrow></div>
                        <div data-flow-parallel>
                            <div data-flow-block data-component="proj">
                                <div data-block-title>Q Projection + Gate</div>
                                <div data-block-shape>${info.hidden_size} &rarr; ${info.num_attention_heads}&times;${info.head_dim}&times;2</div>
                                <div data-block-detail>Projects to queries <em>and</em> a learned output gate. The gate is a sigmoid modulator applied after attention &mdash; an extra degree of freedom absent in standard transformers.</div>
                            </div>
                            <div data-flow-block data-component="proj">
                                <div data-block-title>K Projection</div>
                                <div data-block-shape>${info.hidden_size} &rarr; ${info.num_kv_heads}&times;${info.head_dim}</div>
                            </div>
                            <div data-flow-block data-component="proj">
                                <div data-block-title>V Projection</div>
                                <div data-block-shape>${info.hidden_size} &rarr; ${info.num_kv_heads}&times;${info.head_dim}</div>
                            </div>
                        </div>
                        <div data-flow-arrow></div>
                        <div data-flow-parallel>
                            <div data-flow-block data-component="rope">
                                <div data-block-title>QK Norm + Partial RoPE</div>
                                <div data-block-detail>
                                    Queries and keys are first RMSNorm&rsquo;d per head (stabilizes training at scale), then <strong>only the first ${rotDim}</strong> of ${info.head_dim} dimensions get Rotary Position Embeddings.
                                    The remaining ${passDim} dims carry pure semantic info with no position encoding.
                                    This &ldquo;partial rotation&rdquo; (25%) lets the model separately control positional vs. semantic matching.
                                </div>
                            </div>
                        </div>
                        <div data-flow-arrow></div>
                        <div data-flow-block data-component="attention" data-highlight>
                            <div data-block-title>&#x25C8; Scaled Dot-Product Attention</div>
                            <div data-block-shape>${info.num_attention_heads} Q heads, ${info.num_kv_heads} KV heads (GQA &times;${gqaRatio})</div>
                            <div data-block-detail>
                                <strong>Grouped Query Attention</strong>: ${info.num_attention_heads} query heads share only ${info.num_kv_heads} key-value heads. Each KV head is broadcast to ${gqaRatio} query heads.
                                This reduces KV cache memory by ${gqaRatio}&times; with minimal quality loss.
                                <pre data-code>attn = softmax(Q &middot; K&#x1D40; / &radic;${info.head_dim}) &middot; V
output = attn &middot; sigmoid(gate)  // gated output</pre>
                            </div>
                        </div>
                        <div data-flow-arrow></div>
                        <div data-flow-block data-component="proj">
                            <div data-block-title>Output Projection</div>
                            <div data-block-shape>${qOutDim} &rarr; ${info.hidden_size}</div>
                        </div>
                    </div>
                    <div data-residual-add>+ residual</div>
                    <div data-layer-flow>
                        <div data-flow-block data-component="norm">
                            <div data-block-title>Post-Attention RMSNorm</div>
                        </div>
                        <div data-flow-arrow></div>
                        <div data-flow-block data-component="moe" data-highlight>
                            <div data-block-title>&#x25C8; Sparse MoE</div>
                            <div data-block-detail>Same MoE block as GDN layers (see below)</div>
                        </div>
                    </div>
                    <div data-residual-add>+ residual</div>
                </div>
            </div>

            <div data-pipe-flow><div data-flow-line></div></div>

            <!-- MoE Detail -->
            <div data-pipe-layer data-layer-type="moe" data-expandable>
                <div data-layer-banner>
                    <span data-layer-badge="moe">MoE</span>
                    <h3>Sparse Mixture of Experts</h3>
                    <span data-dim>Present in every layer &mdash; ${info.num_experts} total experts, top-${info.num_experts_per_tok} routing</span>
                    <button data-expand-toggle onclick="this.closest('[data-expandable]').toggleAttribute('data-expanded')">
                        <span data-expand-open>&#x25B4; collapse</span>
                        <span data-expand-closed>&#x25BE; expand</span>
                    </button>
                </div>

                <div data-layer-body>
                    <div data-moe-visual>
                        <div data-moe-router>
                            <div data-block-title>Router Gate</div>
                            <div data-block-shape>${info.hidden_size} &rarr; ${info.num_experts} logits &rarr; softmax &rarr; top-${info.num_experts_per_tok}</div>
                            <div data-block-detail>
                                A learned linear projection scores all ${info.num_experts} experts per token. After softmax, only the top ${info.num_experts_per_tok} experts are selected.
                                This means each token uses <strong>${expertPct}%</strong> of total expert capacity &mdash; massive model size with bounded compute.
                            </div>
                        </div>
                        <div data-moe-experts-grid>
                            <div data-moe-expert data-active>E<sub>i</sub></div>
                            <div data-moe-expert data-active>E<sub>j</sub></div>
                            <div data-moe-expert data-active>E<sub>k</sub></div>
                            <div data-moe-expert>...</div>
                            <div data-moe-expert data-active style="--hue:40">E<sub>n</sub></div>
                            <span data-moe-label>${info.num_experts_per_tok} of ${info.num_experts} activated</span>
                        </div>
                        <div data-flow-block data-component="proj" style="margin-top:12px">
                            <div data-block-title>Each Expert MLP</div>
                            <div data-block-shape>${info.hidden_size} &rarr; gate(${info.moe_intermediate_size}) &middot; up(${info.moe_intermediate_size}) &rarr; down &rarr; ${info.hidden_size}</div>
                            <div data-block-detail>Standard SwiGLU MLP: gate and up projections are element-wise multiplied with SiLU activation, then projected back down. Small per-expert (${info.moe_intermediate_size}d intermediate) but massive in aggregate.</div>
                        </div>
                    </div>

                    <div data-pipe-flow style="margin:16px 0"><div data-flow-line></div><span data-flow-label>weighted sum by routing probabilities</span></div>

                    <div data-flow-block data-component="shared">
                        <div data-block-title>+ Shared Expert</div>
                        <div data-block-shape>${info.hidden_size} &rarr; ${info.shared_expert_intermediate_size}d &rarr; ${info.hidden_size}, gated by &sigma;(gate)</div>
                        <div data-block-detail>
                            A separate MLP that <em>always</em> fires for every token, regardless of routing. Its output is multiplied by a learned sigmoid gate before being added to the routed expert output.
                            This captures common patterns that all tokens need, while the sparse experts specialize.
                        </div>
                    </div>
                </div>
            </div>

            <div data-pipe-flow><div data-flow-line></div></div>

            <!-- Final Norm -->
            <div data-pipe-node data-component="norm-final">
                <div data-pipe-node-inner>
                    <div data-pipe-content>
                        <h3>Final RMSNorm</h3>
                        <div data-pipe-shape>(${info.hidden_size}d)</div>
                    </div>
                </div>
                <div data-pipe-annotation>Normalize once more before projecting to vocabulary space.</div>
            </div>

            <div data-pipe-flow><div data-flow-line></div><span data-flow-label>(batch, seq, ${info.hidden_size})</span></div>

            <!-- LM Head -->
            <div data-pipe-node data-component="lm-head">
                <div data-pipe-node-inner>
                    <div data-pipe-icon>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--amber)" stroke-width="1.5">
                            <path d="M4 20 L12 4 L20 20"/><line x1="7" y1="14" x2="17" y2="14"/>
                        </svg>
                    </div>
                    <div data-pipe-content>
                        <h3>LM Head</h3>
                        <div data-pipe-shape>${info.hidden_size} &rarr; ${info.vocab_size} logits</div>
                    </div>
                </div>
                <div data-pipe-annotation>
                    Projects the final hidden state into a <strong>${info.vocab_size}</strong>-dimensional vector. After softmax, this becomes a probability distribution over all possible next tokens.
                    <em>This is the measurement point for logit lens and steering experiments.</em>
                </div>
            </div>

            <!-- Introspection callout -->
            <div data-callout>
                <h3>&#x1f52c; Where introspection happens</h3>
                <p>
                    <strong>Hidden state capture:</strong> After each layer, the residual stream tensor is cloned &mdash; giving a snapshot of the model&rsquo;s &ldquo;thinking&rdquo; at that depth.
                    Projecting these through the LM Head (the <em>logit lens</em>) reveals what the model would predict if it stopped processing early.
                </p>
                <p>
                    <strong>Steering vectors:</strong> Trained via contrastive PCA on hidden state <em>differences</em> between concept-positive and concept-negative prompts. When added to the residual stream at a target layer, they push the model&rsquo;s representations toward the concept &mdash; like adjusting a dial on the model&rsquo;s internal state.
                </p>
            </div>
        </div>
    </section>`;
}

function buildStandardArchExplorer(info) {
    const gqaRatio = info.num_kv_heads > 0 ? info.num_attention_heads / info.num_kv_heads : 1;
    const qOutDim = info.num_attention_heads * info.head_dim;

    return `
    <section data-component="arch-explorer" id="arch-explorer">
        <div data-section-header>
            <h2>Model Architecture Explorer</h2>
            <span data-dim>Standard transformer forward pass</span>
        </div>

        <div data-pipeline>
            <div data-pipe-node data-component="embedding">
                <div data-pipe-node-inner>
                    <div data-pipe-content>
                        <h3>Token Embedding</h3>
                        <div data-pipe-shape>${info.vocab_size} &rarr; ${info.hidden_size}d</div>
                    </div>
                </div>
            </div>

            <div data-pipe-flow><div data-flow-line></div><span data-flow-label>(batch, seq, ${info.hidden_size})</span></div>

            <div data-pipe-stack-header>
                <div data-pipe-stack-title>
                    Decoder Stack
                    <span data-pipe-stack-count>&times; ${info.num_layers} layers</span>
                </div>
                <div data-pipe-stack-breakdown>
                    <span data-tag="attn-count">${info.num_layers} Full Attention + MLP</span>
                </div>
            </div>

            <div data-pipe-layer data-layer-type="attn" data-expandable data-expanded>
                <div data-layer-banner>
                    <span data-layer-badge="attn">ATTN</span>
                    <h3>Grouped Query Attention + MLP</h3>
                    <span data-dim>All ${info.num_layers} layers</span>
                    <button data-expand-toggle onclick="this.closest('[data-expandable]').toggleAttribute('data-expanded')">
                        <span data-expand-open>&#x25B4; collapse</span>
                        <span data-expand-closed>&#x25BE; expand</span>
                    </button>
                </div>

                <div data-layer-body>
                    <div data-layer-flow>
                        <div data-flow-block data-component="norm">
                            <div data-block-title>Input RMSNorm</div>
                        </div>
                        <div data-flow-arrow></div>
                        <div data-flow-parallel>
                            <div data-flow-block data-component="proj">
                                <div data-block-title>Q Projection</div>
                                <div data-block-shape>${info.hidden_size} &rarr; ${info.num_attention_heads}&times;${info.head_dim}</div>
                            </div>
                            <div data-flow-block data-component="proj">
                                <div data-block-title>K Projection</div>
                                <div data-block-shape>${info.hidden_size} &rarr; ${info.num_kv_heads}&times;${info.head_dim}</div>
                            </div>
                            <div data-flow-block data-component="proj">
                                <div data-block-title>V Projection</div>
                                <div data-block-shape>${info.hidden_size} &rarr; ${info.num_kv_heads}&times;${info.head_dim}</div>
                            </div>
                        </div>
                        <div data-flow-arrow></div>
                        <div data-flow-block data-component="attention" data-highlight>
                            <div data-block-title>&#x25C8; Scaled Dot-Product Attention</div>
                            <div data-block-shape>${info.num_attention_heads} Q heads, ${info.num_kv_heads} KV heads (GQA &times;${gqaRatio})</div>
                        </div>
                        <div data-flow-arrow></div>
                        <div data-flow-block data-component="proj">
                            <div data-block-title>Output Projection</div>
                            <div data-block-shape>${qOutDim} &rarr; ${info.hidden_size}</div>
                        </div>
                    </div>
                    <div data-residual-add>+ residual</div>
                    <div data-layer-flow>
                        <div data-flow-block data-component="norm">
                            <div data-block-title>Post-Attention RMSNorm</div>
                        </div>
                        <div data-flow-arrow></div>
                        <div data-flow-block data-component="proj" data-highlight>
                            <div data-block-title>&#x25C8; SwiGLU MLP</div>
                            <div data-block-shape>${info.hidden_size} &rarr; gate &middot; up &rarr; down &rarr; ${info.hidden_size}</div>
                        </div>
                    </div>
                    <div data-residual-add>+ residual</div>
                </div>
            </div>

            <div data-pipe-flow><div data-flow-line></div></div>

            <div data-pipe-node data-component="norm-final">
                <div data-pipe-node-inner>
                    <div data-pipe-content>
                        <h3>Final RMSNorm</h3>
                        <div data-pipe-shape>(${info.hidden_size}d)</div>
                    </div>
                </div>
            </div>

            <div data-pipe-flow><div data-flow-line></div><span data-flow-label>(batch, seq, ${info.hidden_size})</span></div>

            <div data-pipe-node data-component="lm-head">
                <div data-pipe-node-inner>
                    <div data-pipe-content>
                        <h3>LM Head</h3>
                        <div data-pipe-shape>${info.hidden_size} &rarr; ${info.vocab_size} logits</div>
                    </div>
                </div>
                <div data-pipe-annotation>
                    <em>Measurement point for logit lens and steering experiments.</em>
                </div>
            </div>

            <div data-callout>
                <h3>&#x1f52c; Where introspection happens</h3>
                <p>
                    <strong>Hidden state capture:</strong> After each layer, the residual stream tensor is cloned &mdash; giving a snapshot of the model&rsquo;s &ldquo;thinking&rdquo; at that depth.
                    Projecting these through the LM Head (the <em>logit lens</em>) reveals what the model would predict if it stopped processing early.
                </p>
                <p>
                    <strong>Steering vectors:</strong> Trained via contrastive PCA on hidden state <em>differences</em> between concept-positive and concept-negative prompts. When added to the residual stream at a target layer, they push the model&rsquo;s representations toward the concept.
                </p>
            </div>
        </div>
    </section>`;
}

// ── Effects ─────────────────────────────────────────────────────────

function setupEffects() {
    // Mouse parallax on architecture explorer
    const el = document.querySelector('[data-component="arch-explorer"]');
    if (el) {
        document.addEventListener('mousemove', e => {
            const mx = (e.clientX / window.innerWidth - 0.5) * 2;
            const my = (e.clientY / window.innerHeight - 0.5) * 2;
            el.style.setProperty('--mx', mx.toFixed(3));
            el.style.setProperty('--my', my.toFixed(3));
        });
    }

    // Scroll-triggered reveal
    const obs = new IntersectionObserver(entries => {
        entries.forEach(e => {
            if (e.isIntersecting) { e.target.setAttribute('data-visible', ''); obs.unobserve(e.target); }
        });
    }, { threshold: 0.1 });
    document.querySelectorAll('[data-pipe-node], [data-pipe-layer], [data-callout]').forEach(el => obs.observe(el));
}

// ── Auto-refresh & Init ─────────────────────────────────────────────

loadDashboard();
setInterval(loadDashboard, 30000);

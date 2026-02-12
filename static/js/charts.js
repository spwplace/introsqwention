// Shared Plotly chart builders for experiment visualizations.
// Each function takes data (from the API JSON) and a container element or ID.

const PL_LAYOUT = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#d0d5dd', family: 'IBM Plex Mono' },
};
const PL_CONFIG = { displayModeBar: false, responsive: true };

function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

// ── Logit Lens Heatmap + Per-Layer Cards ────────────────────────────

function renderLogitLens(lens, mount) {
    // Heatmap
    const heatmapData = lens.layers.map(l => [l.top1_prob]);
    const layerLabels = lens.layers.map(l => `L${l.layer_idx} (${l.layer_type})`);

    const heatPanel = document.createElement('section');
    heatPanel.setAttribute('data-panel', '');
    heatPanel.innerHTML = `<div data-section-header><h2>Logit Lens &mdash; Prediction Confidence</h2></div><div id="logit-lens-heatmap"></div>`;
    mount.appendChild(heatPanel);

    Plotly.newPlot('logit-lens-heatmap', [{
        z: heatmapData, y: layerLabels, x: ['last token'],
        type: 'heatmap',
        colorscale: [[0,'#0a0c10'],[0.3,'#0d3b3b'],[0.6,'#00d4aa'],[1,'#fbbf24']],
        hovertemplate: '%{y}<br>P(top-1): %{z:.4f}<extra></extra>',
    }], {
        ...PL_LAYOUT,
        margin: { t:10, b:30, l:140, r:30 },
        height: Math.max(400, layerLabels.length * 16),
        yaxis: { autorange:'reversed', color:'#6b7280', tickfont:{ family:'IBM Plex Mono', size:10 } },
    }, PL_CONFIG);

    // Per-layer cards
    let cardsHtml = '';
    for (const layer of lens.layers) {
        let bars = '';
        for (const tp of layer.top_tokens.slice(0, 8)) {
            const w = Math.min(tp.probability * 300, 100);
            bars += `<div data-token-row><div data-token-track><div data-token-fill style="width:${w}%"></div></div><span data-token-prob>${tp.probability.toFixed(3)}</span><span data-token-text>${esc(JSON.stringify(tp.token))}</span></div>`;
        }
        cardsHtml += `<div data-layer-card><div data-layer-card-header>Layer ${layer.layer_idx} <span data-dim>${layer.layer_type}</span></div><div data-token-bars>${bars}</div></div>`;
    }

    const cardsPanel = document.createElement('section');
    cardsPanel.setAttribute('data-panel', '');
    cardsPanel.innerHTML = `<div data-section-header><h2>Per-Layer Predictions</h2></div><div data-layer-cards-grid>${cardsHtml}</div>`;
    mount.appendChild(cardsPanel);
}

// ── Logit Diff (Introspection Detection) ────────────────────────────

function renderLogitDiff(diff, mount) {
    const panel = document.createElement('section');
    panel.setAttribute('data-panel', '');
    panel.innerHTML = `<div data-section-header><h2>Introspection Detection</h2></div>
        <div data-meta-chips>
            <span data-meta-chip>concept: <strong>${esc(diff.concept)}</strong></span>
            <span data-meta-chip>scale: <strong>${diff.scale}</strong></span>
            <span data-meta-chip>variant: <strong>${esc(diff.variant)}</strong></span>
        </div>
        <div id="logit-diff-chart"></div>`;
    mount.appendChild(panel);

    Plotly.newPlot('logit-diff-chart', [
        { x:['P(yes)','P(no)'], y:[diff.base_p_yes, diff.base_p_no], name:'Base', type:'bar', marker:{ color:'rgba(107,114,128,0.6)', line:{ color:'#6b7280', width:1 } } },
        { x:['P(yes)','P(no)'], y:[diff.steered_p_yes, diff.steered_p_no], name:'Steered', type:'bar', marker:{ color:'rgba(0,212,170,0.7)', line:{ color:'#00d4aa', width:1 } } },
    ], {
        ...PL_LAYOUT, barmode:'group', bargap:0.3,
        margin:{ t:10, b:40, l:60, r:20 }, height:280,
        yaxis:{ title:'Probability', tickformat:'.1%', gridcolor:'#1e2330', zerolinecolor:'#1e2330' },
    }, PL_CONFIG);

    // Top tokens
    const tt = diff.top_tokens;
    const tokPanel = document.createElement('section');
    tokPanel.setAttribute('data-panel', '');
    tokPanel.innerHTML = `<div data-section-header><h2>Top Tokens</h2></div><div id="token-diff-chart"></div>`;
    mount.appendChild(tokPanel);

    Plotly.newPlot('token-diff-chart', [
        { x:tt.slice(0,10).map(t=>t.token), y:tt.slice(0,10).map(t=>t.base_prob), name:'Base', type:'bar', marker:{ color:'rgba(107,114,128,0.5)' } },
        { x:tt.slice(0,10).map(t=>t.token), y:tt.slice(0,10).map(t=>t.steered_prob), name:'Steered', type:'bar', marker:{ color:'rgba(0,212,170,0.7)' } },
    ], {
        ...PL_LAYOUT, barmode:'group',
        margin:{ t:10, b:60, l:60, r:20 }, height:300,
        yaxis:{ title:'Probability', tickformat:'.1%', gridcolor:'#1e2330' },
        xaxis:{ tickangle:-30 },
    }, PL_CONFIG);
}

// ── Control Questions ───────────────────────────────────────────────

function renderControlQuestions(cq, mount) {
    const s = cq.summary;
    let qr = '';
    for (const q of cq.questions) {
        const ys = q.steered_p_yes - q.base_p_yes;
        const ns = q.steered_p_no - q.base_p_no;
        const yc = Math.abs(ys) > 0.05 ? 'warn' : 'ok';
        const nc = Math.abs(ns) > 0.05 ? 'warn' : 'ok';
        qr += `<tr>
            <td>${esc(q.question)}</td>
            <td data-mono>${(q.base_p_yes*100).toFixed(1)}%</td>
            <td data-mono>${(q.steered_p_yes*100).toFixed(1)}%</td>
            <td data-mono data-shift="${yc}">${(ys*100) >= 0 ? '+' : ''}${(ys*100).toFixed(2)}%</td>
            <td data-mono>${(q.base_p_no*100).toFixed(1)}%</td>
            <td data-mono>${(q.steered_p_no*100).toFixed(1)}%</td>
            <td data-mono data-shift="${nc}">${(ns*100) >= 0 ? '+' : ''}${(ns*100).toFixed(2)}%</td>
        </tr>`;
    }

    const chartPanel = document.createElement('section');
    chartPanel.setAttribute('data-panel', '');
    chartPanel.innerHTML = `<div data-section-header><h2>Control Questions</h2></div>
        <div data-meta-chips>
            <span data-meta-chip>concept: <strong>${esc(cq.concept)}</strong></span>
            <span data-meta-chip>scale: <strong>${cq.scale}</strong></span>
            <span data-meta-chip>mean &Delta;yes: <strong>${(s.mean_yes_shift*100) >= 0 ? '+' : ''}${(s.mean_yes_shift*100).toFixed(2)}%</strong></span>
            <span data-meta-chip>mean &Delta;no: <strong>${(s.mean_no_shift*100) >= 0 ? '+' : ''}${(s.mean_no_shift*100).toFixed(2)}%</strong></span>
        </div>
        <div id="cq-chart"></div>`;
    mount.appendChild(chartPanel);

    const qs = cq.questions;
    Plotly.newPlot('cq-chart', [
        { x:qs.map(q=>q.question.substring(0,35)+'...'), y:qs.map(q=>(q.steered_p_yes-q.base_p_yes)*100), name:'\u0394 Yes', type:'bar', marker:{ color:'rgba(0,212,170,0.6)' } },
        { x:qs.map(q=>q.question.substring(0,35)+'...'), y:qs.map(q=>(q.steered_p_no-q.base_p_no)*100), name:'\u0394 No', type:'bar', marker:{ color:'rgba(251,191,36,0.5)' } },
    ], {
        ...PL_LAYOUT, barmode:'group',
        margin:{ t:10, b:110, l:60, r:20 }, height:380,
        yaxis:{ title:'Shift (%)', gridcolor:'#1e2330' },
        xaxis:{ tickangle:-45 },
    }, PL_CONFIG);

    const tablePanel = document.createElement('section');
    tablePanel.setAttribute('data-panel', '');
    tablePanel.innerHTML = `<div data-section-header><h2>Per-Question</h2></div>
        <div style="overflow-x:auto"><table>
            <thead><tr><th>Question</th><th>Base P(y)</th><th>Steer P(y)</th><th>&Delta;Yes</th><th>Base P(n)</th><th>Steer P(n)</th><th>&Delta;No</th></tr></thead>
            <tbody>${qr}</tbody>
        </table></div>`;
    mount.appendChild(tablePanel);
}

// ── Logit Lens Comparison ───────────────────────────────────────────

function renderLogitLensComparison(cmp, mount) {
    const palette = ['#00d4aa','#fbbf24','#a78bfa','#ef4444','#38bdf8','#f472b6'];
    const traces = [];
    for (let ti = 0; ti < cmp.tracked_tokens.length; ti++) {
        const c = palette[ti % palette.length];
        const tok = cmp.tracked_tokens[ti];
        const bp = cmp.base_layers.map(l => (l.tracked_probs[ti] || 0) * 100);
        const sp = cmp.steered_layers.map(l => (l.tracked_probs[ti] || 0) * 100);
        const li = cmp.base_layers.map(l => l.layer_idx);
        traces.push({ x:li, y:bp, name:`base: ${tok}`, type:'scatter', mode:'lines', line:{ color:c, dash:'dot', width:1.5 }, opacity:0.5 });
        traces.push({ x:li, y:sp, name:`steered: ${tok}`, type:'scatter', mode:'lines+markers', line:{ color:c, width:2.5 }, marker:{ size:4 } });
    }

    const panel = document.createElement('section');
    panel.setAttribute('data-panel', '');
    panel.innerHTML = `<div data-section-header><h2>Logit Lens Comparison</h2></div>
        <div data-meta-chips>
            <span data-meta-chip>concept: <strong>${esc(cmp.concept)}</strong></span>
            <span data-meta-chip>scale: <strong>${cmp.scale}</strong></span>
            <span data-meta-chip>tracking: <strong>${esc(cmp.tracked_tokens.join(', '))}</strong></span>
        </div>
        <div id="llc-chart"></div>`;
    mount.appendChild(panel);

    Plotly.newPlot('llc-chart', traces, {
        ...PL_LAYOUT,
        margin:{ t:10, b:50, l:60, r:20 }, height:460,
        xaxis:{ title:'Layer', dtick:4, gridcolor:'#1e2330' },
        yaxis:{ title:'P (%)', rangemode:'tozero', gridcolor:'#1e2330' },
        legend:{ bgcolor:'rgba(10,12,16,0.8)', bordercolor:'#1e2330', borderwidth:1 },
        hovermode:'x unified',
    }, PL_CONFIG);
}

// ── Top of Mind (Generation) ────────────────────────────────────────

function renderTopOfMind(tom, mount) {
    const panel = document.createElement('section');
    panel.setAttribute('data-panel', '');
    panel.innerHTML = `<div data-section-header><h2>Top of Mind &mdash; Generation</h2></div>
        <div data-meta-chips>
            <span data-meta-chip>concept: <strong>${esc(tom.concept)}</strong></span>
            <span data-meta-chip>scale: <strong>${tom.scale}</strong></span>
            <span data-meta-chip>temp: <strong>${tom.temperature}</strong></span>
            <span data-meta-chip>tokens: <strong>${tom.num_tokens}</strong></span>
            <span data-meta-chip>stop: <strong>${esc(tom.stop_reason)}</strong></span>
        </div>
        <div data-generation-block><pre data-generation-text>${esc(tom.generated_text)}</pre></div>`;
    mount.appendChild(panel);
}

// ── Concept Activation ──────────────────────────────────────────────

function renderConceptActivation(ca, mount) {
    const li = ca.activations.map(a => a.layer_idx);
    const av = ca.activations.map(a => a.activation);

    const panel = document.createElement('section');
    panel.setAttribute('data-panel', '');
    panel.innerHTML = `<div data-section-header><h2>Concept Activation</h2></div>
        <div data-meta-chips>
            <span data-meta-chip>concept: <strong>${esc(ca.concept)}</strong></span>
            <span data-meta-chip>mean: <strong>${ca.mean_activation.toFixed(4)}</strong></span>
            <span data-meta-chip>max: <strong>${ca.max_activation.toFixed(4)}</strong> (L${ca.max_layer})</span>
        </div>
        <div data-config-prompt><span data-config-label>Text</span><pre data-config-text>${esc(ca.text)}</pre></div>
        <div id="ca-chart" style="margin-top:16px"></div>`;
    mount.appendChild(panel);

    Plotly.newPlot('ca-chart', [{
        x: li, y: av, type: 'bar',
        marker: { color: av.map(v => v > 0 ? 'rgba(0,212,170,0.7)' : 'rgba(239,68,68,0.6)') },
    }], {
        ...PL_LAYOUT,
        margin:{ t:10, b:50, l:60, r:20 }, height:320,
        xaxis:{ title:'Layer', dtick:4 },
        yaxis:{ title:'Activation (dot product)', gridcolor:'#1e2330', zerolinecolor:'#374151' },
        showlegend: false,
    }, PL_CONFIG);
}

// ══════════════════════════════════════════════════════════════════════
// Hybrid Architecture Experiments (QWEN3-SPEC)
// ══════════════════════════════════════════════════════════════════════

// ── Layer Type Lens (3F) ──────────────────────────────────────────────

function renderLayerTypeLens(ltl, mount) {
    const li = ltl.layers.map(l => l.layer_idx);
    const probs = ltl.layers.map(l => l.top1_prob);
    const entropies = ltl.layers.map(l => l.entropy);
    const colors = ltl.layers.map(l => l.layer_type === 'gdn' ? 'rgba(0,212,170,0.7)' : 'rgba(167,139,250,0.7)');

    const panel = document.createElement('section');
    panel.setAttribute('data-panel', '');
    panel.innerHTML = `<div data-section-header><h2>Layer Type Lens</h2></div>
        <div data-meta-chips>
            <span data-meta-chip>GDN mean conf: <strong>${ltl.mean_gdn_confidence.toFixed(4)}</strong></span>
            <span data-meta-chip>Attn mean conf: <strong>${ltl.mean_attn_confidence.toFixed(4)}</strong></span>
        </div>
        <div id="ltl-conf-chart"></div>
        <div id="ltl-entropy-chart" style="margin-top:16px"></div>`;
    mount.appendChild(panel);

    // Confidence by layer, colored by type
    Plotly.newPlot('ltl-conf-chart', [{
        x: li, y: probs.map(p => p * 100), type: 'bar',
        marker: { color: colors },
        hovertemplate: 'L%{x}<br>%{y:.2f}%<extra></extra>',
    }], {
        ...PL_LAYOUT,
        margin:{ t:10, b:50, l:60, r:20 }, height:300,
        xaxis:{ title:'Layer', dtick:4 },
        yaxis:{ title:'Top-1 Confidence (%)', gridcolor:'#1e2330' },
        showlegend: false,
    }, PL_CONFIG);

    // Entropy by layer
    Plotly.newPlot('ltl-entropy-chart', [{
        x: li, y: entropies, type: 'scatter', mode: 'lines+markers',
        line: { color: '#fbbf24', width: 2 },
        marker: { size: 4, color: colors },
    }], {
        ...PL_LAYOUT,
        margin:{ t:10, b:50, l:60, r:20 }, height:260,
        xaxis:{ title:'Layer', dtick:4 },
        yaxis:{ title:'Entropy (nats)', gridcolor:'#1e2330' },
        showlegend: false,
    }, PL_CONFIG);

    // Attention jumps table
    if (ltl.attn_jumps && ltl.attn_jumps.length > 0) {
        let rows = '';
        for (const j of ltl.attn_jumps) {
            rows += `<tr><td data-mono>L${j.layer_idx}</td><td data-mono>${(j.value * 100).toFixed(2)}%</td></tr>`;
        }
        const jumpPanel = document.createElement('section');
        jumpPanel.setAttribute('data-panel', '');
        jumpPanel.innerHTML = `<div data-section-header><h2>Confidence Jumps at Attention Layers</h2></div>
            <div style="overflow-x:auto"><table>
                <thead><tr><th>Layer</th><th>&Delta; Confidence</th></tr></thead>
                <tbody>${rows}</tbody>
            </table></div>`;
        mount.appendChild(jumpPanel);
    }
}

// ── Steering Survival (3D) ────────────────────────────────────────────

function renderSteeringSurvival(ss, mount) {
    const palette = ['#00d4aa','#fbbf24','#a78bfa','#ef4444','#38bdf8','#f472b6','#34d399','#fb923c'];
    const traces = [];
    for (let i = 0; i < ss.traces.length; i++) {
        const tr = ss.traces[i];
        const c = palette[i % palette.length];
        traces.push({
            x: tr.activations.map(a => a.layer_idx),
            y: tr.activations.map(a => a.activation),
            name: `inject L${tr.injection_layer} (${tr.injection_layer_type})`,
            type: 'scatter', mode: 'lines+markers',
            line: { color: c, width: 2 },
            marker: { size: 4 },
        });
    }

    const panel = document.createElement('section');
    panel.setAttribute('data-panel', '');
    panel.innerHTML = `<div data-section-header><h2>Steering Survival Through Layers</h2></div>
        <div id="ss-chart"></div>`;
    mount.appendChild(panel);

    Plotly.newPlot('ss-chart', traces, {
        ...PL_LAYOUT,
        margin:{ t:10, b:50, l:60, r:20 }, height:400,
        xaxis:{ title:'Layer', dtick:4, gridcolor:'#1e2330' },
        yaxis:{ title:'Concept Activation (cosine)', gridcolor:'#1e2330', zerolinecolor:'#374151' },
        legend:{ bgcolor:'rgba(10,12,16,0.8)', bordercolor:'#1e2330', borderwidth:1 },
        hovermode:'x unified',
    }, PL_CONFIG);
}

// ── CKA Cross-Layer Similarity (3A) ──────────────────────────────────

function renderCka(cka, mount) {
    const panel = document.createElement('section');
    panel.setAttribute('data-panel', '');
    panel.innerHTML = `<div data-section-header><h2>CKA Cross-Layer Similarity</h2></div>
        <div id="cka-heatmap"></div>
        <div id="cka-consec" style="margin-top:16px"></div>`;
    mount.appendChild(panel);

    // Full similarity matrix heatmap
    Plotly.newPlot('cka-heatmap', [{
        z: cka.similarity_matrix,
        x: cka.layer_labels,
        y: cka.layer_labels,
        type: 'heatmap',
        colorscale: [[0,'#0a0c10'],[0.3,'#1a1040'],[0.6,'#a78bfa'],[1,'#fbbf24']],
        hovertemplate: '%{x} vs %{y}<br>CKA: %{z:.4f}<extra></extra>',
    }], {
        ...PL_LAYOUT,
        margin:{ t:10, b:100, l:100, r:30 },
        height: Math.max(500, cka.layer_labels.length * 14),
        xaxis:{ tickangle:-45, tickfont:{ size:9 } },
        yaxis:{ autorange:'reversed', tickfont:{ size:9 } },
    }, PL_CONFIG);

    // Consecutive CKA line
    const consecLabels = cka.layer_labels.slice(1).map((l, i) => `${cka.layer_labels[i]}→${l}`);
    Plotly.newPlot('cka-consec', [{
        x: consecLabels, y: cka.consecutive_cka,
        type: 'scatter', mode: 'lines+markers',
        line: { color: '#a78bfa', width: 2 },
        marker: { size: 5 },
    }], {
        ...PL_LAYOUT,
        margin:{ t:10, b:100, l:60, r:20 }, height:300,
        xaxis:{ tickangle:-45, tickfont:{ size:8 } },
        yaxis:{ title:'CKA', range:[0,1], gridcolor:'#1e2330' },
        showlegend: false,
    }, PL_CONFIG);
}

// ── MoE Routing Analysis (3B) ────────────────────────────────────────

function renderRoutingAnalysis(ra, mount) {
    const panel = document.createElement('section');
    panel.setAttribute('data-panel', '');
    panel.innerHTML = `<div data-section-header><h2>MoE Routing Analysis</h2></div>
        <div id="ra-entropy-chart"></div>
        <div id="ra-shared-gate-chart" style="margin-top:16px"></div>
        <div id="ra-divergence-chart" style="margin-top:16px"></div>
        <div id="ra-expert-heatmap" style="margin-top:16px"></div>`;
    mount.appendChild(panel);

    // Routing entropy per layer
    const eli = ra.layer_entropy.map(l => l.layer_idx);
    const ev = ra.layer_entropy.map(l => l.value);
    Plotly.newPlot('ra-entropy-chart', [{
        x: eli, y: ev, type: 'bar',
        marker: { color: 'rgba(0,212,170,0.6)' },
    }], {
        ...PL_LAYOUT,
        margin:{ t:10, b:50, l:60, r:20 }, height:280,
        xaxis:{ title:'Layer', dtick:4 },
        yaxis:{ title:'Routing Entropy (nats)', gridcolor:'#1e2330' },
        showlegend: false,
    }, PL_CONFIG);

    // Shared gate values
    const sgi = ra.shared_gate_values.map(l => l.layer_idx);
    const sgv = ra.shared_gate_values.map(l => l.value);
    Plotly.newPlot('ra-shared-gate-chart', [{
        x: sgi, y: sgv, type: 'bar',
        marker: { color: 'rgba(251,191,36,0.6)' },
    }], {
        ...PL_LAYOUT,
        margin:{ t:10, b:50, l:60, r:20 }, height:260,
        xaxis:{ title:'Layer', dtick:4 },
        yaxis:{ title:'Shared Gate Value', gridcolor:'#1e2330' },
        showlegend: false,
    }, PL_CONFIG);

    // Routing KL divergence (if present)
    if (ra.routing_divergence && ra.routing_divergence.length > 0) {
        const rdi = ra.routing_divergence.map(l => l.layer_idx);
        const rdv = ra.routing_divergence.map(l => l.value);
        Plotly.newPlot('ra-divergence-chart', [{
            x: rdi, y: rdv, type: 'bar',
            marker: { color: 'rgba(239,68,68,0.6)' },
        }], {
            ...PL_LAYOUT,
            margin:{ t:10, b:50, l:60, r:20 }, height:260,
            xaxis:{ title:'Layer', dtick:4 },
            yaxis:{ title:'KL(steered || base)', gridcolor:'#1e2330' },
            showlegend: false,
        }, PL_CONFIG);
    }

    // Expert frequency heatmap
    if (ra.expert_frequency && ra.expert_frequency.length > 0) {
        const z = ra.expert_frequency.map(l => l.frequencies);
        const ylabels = ra.expert_frequency.map(l => `L${l.layer_idx}`);
        const xlabels = z[0] ? z[0].map((_, i) => `E${i}`) : [];
        Plotly.newPlot('ra-expert-heatmap', [{
            z: z, x: xlabels, y: ylabels,
            type: 'heatmap',
            colorscale: [[0,'#0a0c10'],[0.5,'#0d3b3b'],[1,'#00d4aa']],
            hovertemplate: '%{y} %{x}<br>Freq: %{z:.4f}<extra></extra>',
        }], {
            ...PL_LAYOUT,
            margin:{ t:10, b:60, l:60, r:30 },
            height: Math.max(300, ylabels.length * 16),
            xaxis:{ tickfont:{ size:8 }, tickangle:-45 },
            yaxis:{ autorange:'reversed', tickfont:{ size:9 } },
        }, PL_CONFIG);
    }
}

// ── Causal Tracing (3C) ──────────────────────────────────────────────

function renderCausalTracing(ct, mount) {
    const li = ct.layer_recovery.map(l => l.layer_idx);
    const rv = ct.layer_recovery.map(l => l.recovery);
    const colors = ct.layer_recovery.map(l => l.layer_type === 'gdn' ? 'rgba(0,212,170,0.7)' : 'rgba(167,139,250,0.7)');

    const panel = document.createElement('section');
    panel.setAttribute('data-panel', '');
    panel.innerHTML = `<div data-section-header><h2>Causal Tracing</h2></div>
        <div data-meta-chips>
            <span data-meta-chip>clean: <strong>${esc(ct.clean_top_token)}</strong> (${(ct.clean_prob*100).toFixed(1)}%)</span>
            <span data-meta-chip>corrupted: <strong>${esc(ct.corrupted_top_token)}</strong> (${(ct.corrupted_prob*100).toFixed(1)}%)</span>
        </div>
        <div id="ct-chart"></div>`;
    mount.appendChild(panel);

    Plotly.newPlot('ct-chart', [{
        x: li, y: rv.map(v => v * 100), type: 'bar',
        marker: { color: colors },
        hovertemplate: 'L%{x}<br>Recovery: %{y:.1f}%<extra></extra>',
    }], {
        ...PL_LAYOUT,
        margin:{ t:10, b:50, l:60, r:20 }, height:320,
        xaxis:{ title:'Restored Layer', dtick:4 },
        yaxis:{ title:'Recovery (%)', gridcolor:'#1e2330', range:[0, 105] },
        showlegend: false,
    }, PL_CONFIG);
}

// ── GDN Recurrent State Stats (3E) ───────────────────────────────────

function renderGdnStateStats(gdn, mount) {
    const li = gdn.layers.map(l => l.layer_idx);

    const panel = document.createElement('section');
    panel.setAttribute('data-panel', '');
    panel.innerHTML = `<div data-section-header><h2>GDN Recurrent State</h2></div>
        <div id="gdn-frob-chart"></div>
        <div id="gdn-rank-chart" style="margin-top:16px"></div>
        <div id="gdn-spectral-chart" style="margin-top:16px"></div>`;
    mount.appendChild(panel);

    // Frobenius norm
    Plotly.newPlot('gdn-frob-chart', [{
        x: li, y: gdn.layers.map(l => l.frobenius_norm), type: 'bar',
        marker: { color: 'rgba(0,212,170,0.6)' },
    }], {
        ...PL_LAYOUT,
        margin:{ t:10, b:50, l:60, r:20 }, height:260,
        xaxis:{ title:'GDN Layer', dtick:2 },
        yaxis:{ title:'Frobenius Norm', gridcolor:'#1e2330' },
        showlegend: false,
    }, PL_CONFIG);

    // Effective rank + top singular value (dual axis)
    Plotly.newPlot('gdn-rank-chart', [
        {
            x: li, y: gdn.layers.map(l => l.effective_rank),
            name: 'Effective Rank', type: 'scatter', mode: 'lines+markers',
            line: { color: '#a78bfa', width: 2 }, marker: { size: 5 },
        },
        {
            x: li, y: gdn.layers.map(l => l.top_singular_value),
            name: 'Top Singular Value', type: 'scatter', mode: 'lines+markers',
            yaxis: 'y2',
            line: { color: '#fbbf24', width: 2 }, marker: { size: 5 },
        },
    ], {
        ...PL_LAYOUT,
        margin:{ t:10, b:50, l:60, r:60 }, height:300,
        xaxis:{ title:'GDN Layer', dtick:2 },
        yaxis:{ title:'Effective Rank', gridcolor:'#1e2330', titlefont:{ color:'#a78bfa' } },
        yaxis2:{ title:'Top SV', overlaying:'y', side:'right', titlefont:{ color:'#fbbf24' }, gridcolor:'transparent' },
        legend:{ bgcolor:'rgba(10,12,16,0.8)', bordercolor:'#1e2330', borderwidth:1 },
    }, PL_CONFIG);

    // Spectral entropy
    Plotly.newPlot('gdn-spectral-chart', [{
        x: li, y: gdn.layers.map(l => l.spectral_entropy), type: 'bar',
        marker: { color: 'rgba(56,189,248,0.6)' },
    }], {
        ...PL_LAYOUT,
        margin:{ t:10, b:50, l:60, r:20 }, height:260,
        xaxis:{ title:'GDN Layer', dtick:2 },
        yaxis:{ title:'Spectral Entropy (nats)', gridcolor:'#1e2330' },
        showlegend: false,
    }, PL_CONFIG);
}

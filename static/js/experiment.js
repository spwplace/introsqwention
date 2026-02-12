// Experiment detail page — reads ?id= from URL, fetches experiment, renders results.

(async function() {
    const params = new URLSearchParams(window.location.search);
    const id = params.get('id');
    if (!id) {
        document.getElementById('exp-name').textContent = 'No experiment ID';
        return;
    }

    const resp = await fetch(`/api/experiments/${encodeURIComponent(id)}`);
    if (!resp.ok) {
        document.getElementById('exp-name').textContent = 'Not found';
        document.getElementById('exp-meta').innerHTML = `<span data-dim>${id}</span>`;
        return;
    }

    const exp = await resp.json();
    document.title = `${exp.name} \u2014 Introspect`;

    // Header
    document.getElementById('exp-name').textContent = exp.name;
    document.getElementById('exp-meta').innerHTML =
        `<span data-tag="type">${exp.config.experiment_type}</span><span data-sep>&middot;</span><span data-dim>${exp.id.substring(0,12)}</span>`;

    const statusEl = document.getElementById('exp-status');
    statusEl.setAttribute('data-status', exp.status);
    statusEl.textContent = exp.status;

    // Config
    const configSection = document.getElementById('config-section');
    configSection.style.display = '';
    document.getElementById('exp-prompt').textContent = exp.config.prompt;

    // Visualizations
    const mount = document.getElementById('viz-mount');
    if (exp.results) {
        const r = exp.results;
        if (r.logit_lens) renderLogitLens(r.logit_lens, mount);
        if (r.logit_diff) renderLogitDiff(r.logit_diff, mount);
        if (r.control_questions) renderControlQuestions(r.control_questions, mount);
        if (r.logit_lens_comparison) renderLogitLensComparison(r.logit_lens_comparison, mount);
        if (r.top_of_mind) renderTopOfMind(r.top_of_mind, mount);
        if (r.concept_activation) renderConceptActivation(r.concept_activation, mount);
        // Hybrid architecture experiments (QWEN3-SPEC)
        if (r.layer_type_lens) renderLayerTypeLens(r.layer_type_lens, mount);
        if (r.steering_survival) renderSteeringSurvival(r.steering_survival, mount);
        if (r.cka) renderCka(r.cka, mount);
        if (r.routing_analysis) renderRoutingAnalysis(r.routing_analysis, mount);
        if (r.causal_tracing) renderCausalTracing(r.causal_tracing, mount);
        if (r.gdn_state_stats) renderGdnStateStats(r.gdn_state_stats, mount);
    }

    if (!exp.results || mount.children.length === 0) {
        const noData = document.createElement('p');
        noData.setAttribute('data-empty-hint', '');
        noData.style.padding = '40px';
        noData.textContent = 'No results available.';
        mount.appendChild(noData);
    }

    // Raw JSON
    const rawSection = document.getElementById('raw-section');
    rawSection.style.display = '';
    document.getElementById('exp-json').textContent = JSON.stringify(exp, null, 2);
})();

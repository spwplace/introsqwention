#!/bin/bash
# Full experiment pipeline for Qwen3-Coder-Next (80B/3B-active hybrid GDN+MoE).
#
# Architecture: 48 layers = 12 × (3×GDN + 1×FullAttn), 512 experts (10 active)
# Middle third (steering sweet spot): layers 17-32
# Lower scale than Qwen2.5 since this model has different dynamics.
#
# Usage: bash data/run_qwen3_next.sh [scale] [num_suffixes]

set -euo pipefail

SERVER="http://127.0.0.1:3131"
CONCEPT="love"
SCALE="${1:-8.0}"
NUM_SUFFIXES="${2:-10}"
# Middle third of 48 layers (introspection indices 1-based)
LAYERS='[17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]'
VARIANT="with_info"
TIMEOUT=3600  # 1 hour max per experiment

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Qwen3-Coder-Next Introspection Experiments                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Config: concept=$CONCEPT, scale=$SCALE, suffixes=$NUM_SUFFIXES"
echo "Layers: middle third (17-32), variant=$VARIANT"
echo "Server: $SERVER"
echo ""

# Verify server is up
if ! curl -sf "$SERVER/api/model_info" > /dev/null 2>&1; then
    echo "ERROR: Server not responding at $SERVER"
    exit 1
fi
MODEL_TYPE=$(curl -s "$SERVER/api/model_info" | python3 -c "import json,sys; print(json.load(sys.stdin)['model_type'])")
echo "Model type: $MODEL_TYPE"
NUM_LAYERS=$(curl -s "$SERVER/api/model_info" | python3 -c "import json,sys; print(json.load(sys.stdin)['num_layers'])")
echo "Layers: $NUM_LAYERS"
echo ""

# ── Step 0: Train steering vector ──────────────────────────────────
echo "[$(date -u)] Step 0/7: Training steering vector..."
echo "  Training '$CONCEPT' with $NUM_SUFFIXES suffixes on layers 17-32"
echo "  (2 forwards per pair × $NUM_SUFFIXES pairs)"
curl -s -N -X POST "$SERVER/api/train" \
  -H 'Content-Type: application/json' \
  -d "{\"concept\": \"$CONCEPT\", \"num_suffixes\": $NUM_SUFFIXES, \"layers\": $LAYERS}" \
  --max-time $TIMEOUT | while IFS= read -r line; do
    if echo "$line" | grep -q "^data:"; then
        data=$(echo "$line" | sed 's/^data: //')
        done=$(echo "$data" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('done',''))" 2>/dev/null)
        total=$(echo "$data" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('total',''))" 2>/dev/null)
        if [ -n "$done" ] && [ -n "$total" ]; then
            echo "  Progress: $done/$total pairs"
        fi
    fi
    if echo "$line" | grep -q "event: complete"; then
        echo "  Training complete!"
    fi
done
echo ""

# Verify vector was saved
VECTORS=$(curl -s "$SERVER/api/steering_vectors")
HAS_VECTOR=$(echo "$VECTORS" | python3 -c "import json,sys; vecs=json.load(sys.stdin); print('yes' if any(v['concept']=='$CONCEPT' for v in vecs) else 'no')" 2>/dev/null)
if [ "$HAS_VECTOR" != "yes" ]; then
    echo "ERROR: Steering vector for '$CONCEPT' not found after training!"
    exit 1
fi
echo "[$(date -u)] Steering vector '$CONCEPT' ready."
echo ""

# ── Step 1: Logit Diff ─────────────────────────────────────────────
echo "[$(date -u)] Step 1/7: Running logit diff (18 forwards)..."
curl -s -X POST "$SERVER/api/run/logit_diff" \
  -H 'Content-Type: application/json' \
  --max-time $TIMEOUT \
  -d "{\"concept\": \"$CONCEPT\", \"scale\": $SCALE, \"user_turn1_variant\": \"$VARIANT\", \"layers\": $LAYERS}" \
  > data/result_logit_diff.json
echo "  Saved: data/result_logit_diff.json"
python3 -c "
import json
r = json.load(open('data/result_logit_diff.json'))['result']
print(f\"  P(yes): base {r['base_p_yes']*100:.2f}% -> steered {r['steered_p_yes']*100:.2f}% (shift {r['mean_yes_shift']*100:+.2f}%)\")
print(f\"  Random control P(yes): {r['random_control_p_yes']*100:.2f}%\")
print(f\"  95% CI: [{r['yes_shift_ci95_low']*100:+.2f}%, {r['yes_shift_ci95_high']*100:+.2f}%]\")
" 2>/dev/null || echo "  (could not parse result)"
echo ""

# ── Step 2: Control Questions ──────────────────────────────────────
echo "[$(date -u)] Step 2/7: Running control questions (32 forwards)..."
curl -s -X POST "$SERVER/api/run/control_questions" \
  -H 'Content-Type: application/json' \
  --max-time $TIMEOUT \
  -d "{\"concept\": \"$CONCEPT\", \"scale\": $SCALE, \"user_turn1_variant\": \"$VARIANT\", \"layers\": $LAYERS}" \
  > data/result_control_questions.json
echo "  Saved: data/result_control_questions.json"
python3 -c "
import json
r = json.load(open('data/result_control_questions.json'))['result']['summary']
print(f\"  Accuracy: base {r['base_accuracy']*100:.1f}% -> steered {r['steered_accuracy']*100:.1f}% ({r['accuracy_shift']*100:+.1f}%)\")
print(f\"  Yes shift: {r['mean_yes_shift']*100:+.3f}% (std {r['std_yes_shift']*100:.3f}%)\")
" 2>/dev/null || echo "  (could not parse result)"
echo ""

# ── Step 3: Logit Lens Comparison ──────────────────────────────────
echo "[$(date -u)] Step 3/7: Running logit lens comparison (2 forwards)..."
curl -s -X POST "$SERVER/api/run/logit_lens_comparison" \
  -H 'Content-Type: application/json' \
  --max-time $TIMEOUT \
  -d "{\"concept\": \"$CONCEPT\", \"scale\": $SCALE, \"tracked_tokens\": [\"yes\", \"no\", \"Yes\", \"No\"], \"user_turn1_variant\": \"$VARIANT\", \"layers\": $LAYERS}" \
  > data/result_logit_lens_comparison.json
echo "  Saved: data/result_logit_lens_comparison.json"
echo ""

# ── Step 4: Top of Mind ────────────────────────────────────────────
echo "[$(date -u)] Step 4/7: Running top of mind generation..."
curl -s -X POST "$SERVER/api/run/top_of_mind" \
  -H 'Content-Type: application/json' \
  --max-time $TIMEOUT \
  -d "{\"concept\": \"$CONCEPT\", \"scale\": $SCALE, \"layers\": $LAYERS}" \
  > data/result_top_of_mind.json
echo "  Saved: data/result_top_of_mind.json"
python3 -c "
import json
r = json.load(open('data/result_top_of_mind.json'))['result']
print(f\"  Generated ({r['num_tokens']} tokens, {r['stop_reason']}):\")
print(f\"  {r['generated_text'][:200]}\")
" 2>/dev/null || echo "  (could not parse result)"
echo ""

# ── Step 5: Layer Type Lens (hybrid-specific) ──────────────────────
echo "[$(date -u)] Step 5/7: Running layer type lens (1 forward)..."
curl -s -X POST "$SERVER/api/run/layer_type_lens" \
  -H 'Content-Type: application/json' \
  --max-time $TIMEOUT \
  -d "{\"text\": \"The meaning of life is\"}" \
  > data/result_layer_type_lens.json
echo "  Saved: data/result_layer_type_lens.json"
python3 -c "
import json
r = json.load(open('data/result_layer_type_lens.json'))['result']
print(f\"  Mean GDN confidence: {r['mean_gdn_confidence']:.4f}\")
print(f\"  Mean Attn confidence: {r['mean_attn_confidence']:.4f}\")
jumps = r.get('attn_jumps', [])
if jumps:
    top_jump = max(jumps, key=lambda j: j['value'])
    print(f\"  Biggest attention jump: layer {top_jump['layer_idx']} ({top_jump['value']:+.4f})\")
" 2>/dev/null || echo "  (could not parse result)"
echo ""

# ── Step 6: Steering Survival (hybrid-specific) ───────────────────
echo "[$(date -u)] Step 6/7: Running steering survival..."
# Inject at layers that span GDN and attention boundaries
INJECTION_LAYERS='[12,15,16,17,20,24,28,32]'
curl -s -X POST "$SERVER/api/run/steering_survival" \
  -H 'Content-Type: application/json' \
  --max-time $TIMEOUT \
  -d "{\"concept\": \"$CONCEPT\", \"injection_layers\": $INJECTION_LAYERS, \"scale\": $SCALE, \"probe_text\": \"The meaning of life is\"}" \
  > data/result_steering_survival.json
echo "  Saved: data/result_steering_survival.json"
python3 -c "
import json
r = json.load(open('data/result_steering_survival.json'))['result']
print(f\"  {len(r['traces'])} injection traces\")
for t in r['traces'][:3]:
    acts = t['activations']
    if acts:
        print(f\"    Layer {t['injection_layer']} ({t['injection_layer_type']}): first act={acts[0]['activation']:.4f}, last act={acts[-1]['activation']:.4f}\")
" 2>/dev/null || echo "  (could not parse result)"
echo ""

# ── Step 7: GDN State Stats (hybrid-specific) ─────────────────────
echo "[$(date -u)] Step 7/7: Running GDN state stats (1 forward)..."
curl -s -X POST "$SERVER/api/run/gdn_state_stats" \
  -H 'Content-Type: application/json' \
  --max-time $TIMEOUT \
  -d "{\"text\": \"The meaning of life is\"}" \
  > data/result_gdn_state_stats.json
echo "  Saved: data/result_gdn_state_stats.json"
python3 -c "
import json
r = json.load(open('data/result_gdn_state_stats.json'))['result']
layers = r['layers']
if layers:
    avg_rank = sum(l['effective_rank'] for l in layers) / len(layers)
    avg_norm = sum(l['frobenius_norm'] for l in layers) / len(layers)
    print(f\"  {len(layers)} GDN layers analyzed\")
    print(f\"  Avg effective rank: {avg_rank:.2f}\")
    print(f\"  Avg Frobenius norm: {avg_norm:.2f}\")
" 2>/dev/null || echo "  (could not parse result)"
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  All experiments complete!                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "[$(date -u)]"
echo ""
echo "Results:"
ls -lh data/result_*.json 2>/dev/null

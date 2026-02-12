#!/bin/bash
# Run the full vgel replication experiment suite after training completes.
# Results are saved to data/ as JSON files.

SERVER="http://127.0.0.1:3131"
CONCEPT="love"
SCALE=8.0
LAYERS='[22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]'
VARIANT="with_info"

echo "=== vgel Replication Experiments on Qwen2.5-Coder-32B ==="
echo "Concept: $CONCEPT, Scale: $SCALE, Layers: middle third (22-43)"
echo ""

# 1. Logit Diff
echo "[$(date -u)] Step 1/4: Running logit diff..."
curl -s -X POST "$SERVER/api/run/logit_diff" \
  -H 'Content-Type: application/json' \
  -d "{\"concept\": \"$CONCEPT\", \"scale\": $SCALE, \"user_turn1_variant\": \"$VARIANT\", \"layers\": $LAYERS}" \
  > data/result_logit_diff.json
echo "  Result saved to data/result_logit_diff.json"
python3 -c "
import json
r = json.load(open('data/result_logit_diff.json'))['result']
print(f\"  P(yes): base {r['base_p_yes']*100:.2f}% -> steered {r['steered_p_yes']*100:.2f}% (shift {r['mean_yes_shift']*100:+.2f}%)\")
print(f\"  Random control P(yes): {r['random_control_p_yes']*100:.2f}%\")
print(f\"  95% CI: [{r['yes_shift_ci95_low']*100:+.2f}%, {r['yes_shift_ci95_high']*100:+.2f}%]\")
"
echo ""

# 2. Control Questions
echo "[$(date -u)] Step 2/4: Running control questions..."
curl -s -X POST "$SERVER/api/run/control_questions" \
  -H 'Content-Type: application/json' \
  -d "{\"concept\": \"$CONCEPT\", \"scale\": $SCALE, \"user_turn1_variant\": \"$VARIANT\", \"layers\": $LAYERS}" \
  > data/result_control_questions.json
echo "  Result saved to data/result_control_questions.json"
python3 -c "
import json
r = json.load(open('data/result_control_questions.json'))['result']['summary']
print(f\"  Accuracy: base {r['base_accuracy']*100:.1f}% -> steered {r['steered_accuracy']*100:.1f}% ({r['accuracy_shift']*100:+.1f}%)\")
print(f\"  Yes shift: {r['mean_yes_shift']*100:+.3f}% (std {r['std_yes_shift']*100:.3f}%)\")
"
echo ""

# 3. Logit Lens Comparison
echo "[$(date -u)] Step 3/4: Running logit lens comparison..."
curl -s -X POST "$SERVER/api/run/logit_lens_comparison" \
  -H 'Content-Type: application/json' \
  -d "{\"concept\": \"$CONCEPT\", \"scale\": $SCALE, \"tracked_tokens\": [\"yes\", \"no\", \"Yes\", \"No\"], \"user_turn1_variant\": \"$VARIANT\", \"layers\": $LAYERS}" \
  > data/result_logit_lens_comparison.json
echo "  Result saved to data/result_logit_lens_comparison.json"
echo ""

# 4. Top of Mind
echo "[$(date -u)] Step 4/4: Running top of mind generation..."
curl -s -X POST "$SERVER/api/run/top_of_mind" \
  -H 'Content-Type: application/json' \
  -d "{\"concept\": \"$CONCEPT\", \"scale\": $SCALE, \"layers\": $LAYERS}" \
  > data/result_top_of_mind.json
echo "  Result saved to data/result_top_of_mind.json"
python3 -c "
import json
r = json.load(open('data/result_top_of_mind.json'))['result']
print(f\"  Generated ({r['num_tokens']} tokens, {r['stop_reason']}):\")
print(f\"  {r['generated_text'][:200]}\")
"
echo ""

echo "=== All experiments complete ==="
echo "[$(date -u)]"

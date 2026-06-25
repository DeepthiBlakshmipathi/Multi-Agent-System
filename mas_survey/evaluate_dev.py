import json
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon

# Load ground truth
with open("data/dev_groundtruth.json", "r", encoding="utf-8") as f:
    groundtruth = json.load(f)

# Load predictions
pred_df = pd.read_csv("artifacts/submission_js.csv")

js_scores = []
map_scores = []

for _, row in pred_df.iterrows():
    question = row["question"]
    pred_dist = json.loads(row["distribution"])
    pred_supports = json.loads(row["supports"])

    # Ground truth
    gt_dist = groundtruth[question]["distribution"]
    gt_supports = groundtruth[question]["supports"]

    # Align distributions
    keys = list(gt_dist.keys())
    p = np.array([pred_dist[k] for k in keys])
    q = np.array([gt_dist[k] for k in keys])

    # JS divergence
    js = jensenshannon(p, q, base=2)
    js_scores.append(js)

    # MAP@100
    hits = 0
    precision_sum = 0
    for i, doc_id in enumerate(pred_supports[:100]):
        if doc_id in gt_supports:
            hits += 1
            precision_sum += hits / (i + 1)
    map_score = precision_sum / min(len(gt_supports), 100)
    map_scores.append(map_score)

# Final scores
avg_js = np.mean(js_scores)
avg_map = np.mean(map_scores)

print(f"Average JS Score (S₀.3): {avg_js:.4f}")
print(f"Average MAP@100: {avg_map:.4f}")

import os
import csv
import yaml
from mas_survey.run_2 import run_pipeline  # your pipeline function

# ------------------------------
# CONFIG
# ------------------------------
SEED = 42
DEV_SLICE = "data/mini_dev.json"
OUTPUT_DIR = "artifacts/ir_experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base config (ensure paths match your setup)
BASE_CONFIG = {
    "paths": {
        "documents_path": "data/mini_documents.jsonl",
        "questions_path": DEV_SLICE,
        "output_path": os.path.join(OUTPUT_DIR, "mini_submission.csv")
    },
    "params": {
        "top_k": 200,
        "seed": SEED,
        "temperature": 1.0,
        "supports": 100
    }
}

# ------------------------------
# Helper function to save CSV
# ------------------------------
def save_results_csv(filename, data):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", newline="", encoding="utf-8") as f:  # <- add encoding="utf-8"
        writer = csv.writer(f)
        writer.writerow(["experiment", "variant", "intrinsic_metric", "js_score"])
        for row in data:
            writer.writerow(row)
    print(f"Saved results to {filepath}")

# ------------------------------
# 1. Disable/Enable experiment
# ------------------------------
disable_enable_results = []

print("Running Disable/Enable experiment: Rerank")

# DISABLE rerank
cfg_disabled = BASE_CONFIG.copy()
cfg_disabled["params"]["rerank"] = False  # disable rerank
result_disabled = run_pipeline(cfg_disabled)
# Example metrics (replace with real evaluation functions)
intrinsic_disabled = 0.85  # placeholder
js_disabled = 0.45          # placeholder
disable_enable_results.append(["rerank", "disabled", intrinsic_disabled, js_disabled])

# ENABLE rerank
cfg_enabled = BASE_CONFIG.copy()
cfg_enabled["params"]["rerank"] = True  # enable rerank
result_enabled = run_pipeline(cfg_enabled)
intrinsic_enabled = 0.92  # placeholder
js_enabled = 0.78          # placeholder
disable_enable_results.append(["rerank", "enabled", intrinsic_enabled, js_enabled])

save_results_csv("disable_enable_rerank.csv", disable_enable_results)


# ------------------------------
# 2. Multi-Variant experiment
# ------------------------------
multi_variant_results = []

print("Running Multi-Variant experiment: BM25 vs BM25→Dense")

# Variant A: BM25 only
cfg_a = BASE_CONFIG.copy()
cfg_a["params"]["use_dense"] = False
result_a = run_pipeline(cfg_a)
intrinsic_a = 0.80  # placeholder
js_a = 0.33          # placeholder
multi_variant_results.append(["bm25_pipeline", "BM25 only", intrinsic_a, js_a])

# Variant B: BM25 → Dense
cfg_b = BASE_CONFIG.copy()
cfg_b["params"]["use_dense"] = True
result_b = run_pipeline(cfg_b)
intrinsic_b = 0.88  # placeholder
js_b = 0.76          # placeholder
multi_variant_results.append(["bm25_pipeline", "BM25→Dense", intrinsic_b, js_b])

save_results_csv("multi_variant_bm25.csv", multi_variant_results)

print("All IR experiments completed. Check artifacts/ir_experiments for CSVs.")

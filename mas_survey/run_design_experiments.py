import os
import json
import csv
import yaml
from mas_survey.run_2 import run_pipeline  # your MAS pipeline function

# CONFIG 
SEED = 42
DEV_SLICE = "data/mini_dev.json"
DEV_GROUNDTRUTH = "data/dev_groundtruth.json"
OUTPUT_DIR = "artifacts/experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG_FILE = "config_2.yaml"  # config used by run_pipeline

# Load config
with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)

# Helper function to save CSV
def save_results_csv(filename, data):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "variant", "intrinsic_metric", "js_score"])
        for row in data:
            writer.writerow(row)
    print(f"Saved results to {filepath}")

# Placeholder evaluation function
def evaluate_js(result, groundtruth_path):
    """
    Dummy function: replace with your actual JS calculation.
    """
    # For now return a random float to simulate JS score
    import random
    return round(random.uniform(0.0, 1.0), 4)

# 1. Disable/Enable experiment
disable_enable_results = []

print("Running Disable/Enable experiment: Supervisor agent")

# Example: disable supervisor flag in config
cfg_disabled = cfg.copy()
cfg_disabled["params"]["disable_supervisor"] = True
result_disabled = run_pipeline(cfg_disabled)
intrinsic_disabled = 0.95  # replace with actual intrinsic metric from result
js_disabled = evaluate_js(result_disabled, DEV_GROUNDTRUTH)
disable_enable_results.append(["supervisor", "disabled", intrinsic_disabled, js_disabled])

# Enable supervisor
cfg_enabled = cfg.copy()
cfg_enabled["params"]["disable_supervisor"] = False
result_enabled = run_pipeline(cfg_enabled)
intrinsic_enabled = 0.97  # replace with actual intrinsic metric from result
js_enabled = evaluate_js(result_enabled, DEV_GROUNDTRUTH)
disable_enable_results.append(["supervisor", "enabled", intrinsic_enabled, js_enabled])

save_results_csv("disable_enable_supervisor.csv", disable_enable_results)

# 2. Multi-Variant experiment
multi_variant_results = []

print("Running Multi-Variant experiment: Routing threshold")

# Variant A: threshold 0.5
cfg_a = cfg.copy()
cfg_a["params"]["routing_threshold"] = 0.5
result_a = run_pipeline(cfg_a)
intrinsic_a = 0.88  # replace with actual metric
js_a = evaluate_js(result_a, DEV_GROUNDTRUTH)
multi_variant_results.append(["routing_threshold", "0.5", intrinsic_a, js_a])

# Variant B: threshold 0.7
cfg_b = cfg.copy()
cfg_b["params"]["routing_threshold"] = 0.7
result_b = run_pipeline(cfg_b)
intrinsic_b = 0.91
js_b = evaluate_js(result_b, DEV_GROUNDTRUTH)
multi_variant_results.append(["routing_threshold", "0.7", intrinsic_b, js_b])

save_results_csv("multi_variant_routing.csv", multi_variant_results)

print("All experiments completed. Check artifacts/experiments for CSVs.")

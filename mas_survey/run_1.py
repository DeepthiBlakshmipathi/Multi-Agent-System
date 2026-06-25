import json
import yaml
import os
import csv
from tqdm import tqdm
from .retrieval import Retriever
from .predict_agent import Predictor

def run_pipeline(cfg):
    """
    Basic MAS pipeline: retrieves documents, predicts distributions, slices supports,
    and writes output CSV with question, distribution, and supports.
    """
    retriever = Retriever(docs_path=cfg["paths"]["docs"])
    predictor = Predictor(seed=cfg["params"]["seed"])

    # Load dev.json (dictionary of questions)
    with open(cfg["paths"]["dev"], "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    rows = []
    for q_text, q_info in tqdm(dev_data.items(), desc="Processing questions"):
        options = list(q_info["distribution"].keys())

        candidate_docs, doc_ids, _ = retriever.retrieve(q_text, top_k=cfg["params"]["top_k"])
        probs = predictor.predict(q_text, options, candidate_docs)

        # Normalize probabilities to sum exactly 1.0
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        keys = list(probs.keys())
        for k in keys[:-1]:
            probs[k] = round(probs[k], 6)
        probs[keys[-1]] = round(1.0 - sum(probs[k] for k in keys[:-1]), 6)

        supports = doc_ids[:cfg["params"]["supports"]]

        row = {
            "question": q_text,
            "distribution": probs,
            "supports": supports
        }
        rows.append(row)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(cfg["paths"]["output"]), exist_ok=True)

    # Write CSV with 3 columns: question, distribution, supports
    with open(cfg["paths"]["output"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "distribution", "supports"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "question": r["question"],
                "distribution": json.dumps(r["distribution"]),
                "supports": json.dumps(r["supports"])
            })

    print(f"Saved results to {cfg['paths']['output']}")

if __name__ == "__main__":
    with open("config_1.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_pipeline(cfg)

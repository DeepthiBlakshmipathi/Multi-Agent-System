import json
import yaml
import os
import csv
import time
import random
from tqdm import tqdm
from .retrieval import Retriever
from .predict_agent import Predictor

def ensure_100_unique(ids_list, desired=100, fallback_pool=None):
    """
    Ensure exactly `desired` unique IDs. Always pad using fallback_pool if needed.
    """
    unique = list(dict.fromkeys(ids_list))
    fallback_pool = fallback_pool or []

    # Add from fallback pool until we reach desired count
    for doc_id in fallback_pool:
        if doc_id not in unique:
            unique.append(doc_id)
        if len(unique) == desired:
            break

    # If still not enough, cycle through fallback_pool again
    while len(unique) < desired:
        for doc_id in fallback_pool:
            unique.append(doc_id)
            if len(unique) == desired:
                break

    return unique[:desired]

def run_pipeline(cfg):
    start_time = time.time()

    retriever = Retriever(
        docs_path=cfg["paths"]["documents_path"],
        emb_path=cfg["paths"].get("embeddings", "data/id_to_embedding.npz")
    )
    predictor = Predictor(
        seed=cfg["params"].get("seed", 42),
        temperature=cfg["params"].get("temperature", 1.0)
    )

    # Load dev questions
    with open(cfg["paths"]["questions_path"], "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    # Load all valid document IDs for fallback
    valid_ids = []
    with open(cfg["paths"]["documents_path"], "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            valid_ids.append(doc["id"])
    random.shuffle(valid_ids)

    rows = []
    for q_text, q_info in dev_data.items():
        options = list(q_info["distribution"].keys())

        # Retrieve documents
        candidate_docs, doc_ids, retriever_scores = retriever.retrieve(
            q_text, top_k=cfg["params"].get("top_k", 200)
        )

        # Predict distribution
        probs = predictor.predict(q_text, options, candidate_docs)

        # Normalize and round
        total = sum(probs.values())
        if total <= 0:
            probs = {k: 1.0 / len(probs) for k in probs}
        else:
            probs = {k: float(v / total) for k, v in probs.items()}

        keys = list(probs.keys())
        for k in keys[:-1]:
            probs[k] = round(probs[k], 6)
        probs[keys[-1]] = round(1.0 - sum(probs[k] for k in keys[:-1]), 6)

        # Ensure 100 unique supports
        supports = ensure_100_unique(list(doc_ids), desired=100, fallback_pool=valid_ids)
        print(f"Final supports count: {len(supports)}")

        rows.append({
            "question": q_text,
            "distribution": probs,
            "supports": supports
        })

    # Write output CSV
    os.makedirs(os.path.dirname(cfg["paths"]["output_path"]), exist_ok=True)
    with open(cfg["paths"]["output_path"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "distribution", "supports"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "question": r["question"],
                "distribution": json.dumps(r["distribution"], ensure_ascii=True),
                "supports": json.dumps(r["supports"], ensure_ascii=True)
            })

    print(f"Saved results to {cfg['paths']['output_path']}")
    elapsed = time.time() - start_time
    print(f"Pipeline completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_2.yaml", help="Path to config file")
    parser.add_argument("--api_key", type=str, default="", help="API key (unused)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    run_pipeline(cfg)

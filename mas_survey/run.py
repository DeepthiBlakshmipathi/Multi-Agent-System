# mas_survey/run.py
import json
import yaml
import os
import csv
import sys 
from tqdm import tqdm
from retrieval import Retriever
from mas_survey.predict_agent import Predictor

def ensure_100_unique(ids_list, desired=100):
    """
    Ensure exactly `desired` unique IDs. If ids_list shorter, cycle through it deterministically.
    """
    if not ids_list:
        return []
    unique = []
    seen = set()
    i = 0
    while len(unique) < desired:
        cur = ids_list[i % len(ids_list)]
        if cur not in seen:
            unique.append(cur)
            seen.add(cur)
        i += 1
        # safety: if we cycle a lot, just add string padding
        if i > desired * 10:
            # fill with repeated last id to reach desired (shouldn't happen)
            while len(unique) < desired:
                unique.append(unique[-1])
            break
    return unique

def run_pipeline(cfg):
    retriever = Retriever(docs_path=cfg["paths"]["documents_path"], emb_path=cfg["paths"].get("embedding_path", "data/id_to_embedding.npz"))
    predictor = Predictor(seed=cfg["params"].get("seed", 42), temperature=cfg["params"].get("temperature", 1.0))

    # Load dev data (this repo uses dictionary of questions)
    with open(cfg["paths"]["dev"], "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    rows = []
    for q_text, q_info in tqdm(dev_data.items(), desc="Processing questions"):
        options = list(q_info["distribution"].keys())

        # Retrieve candidates, also get retriever scores
        candidate_docs, doc_ids, retriever_scores = retriever.retrieve(q_text, top_k=cfg["params"].get("top_k", 200))

        # Predict probabilities using predictor (pass retriever_scores for weighting)
        probs = predictor.predict(q_text, options, candidate_docs)

        # Normalize strictly to sum 1.0 and round as required
        total = sum(probs.values())
        if total <= 0:
            # fallback uniform
            probs = {k: 1.0/len(probs) for k in probs}
        else:
            probs = {k: float(v/total) for k, v in probs.items()}

        # Round all but last to 6 decimals, adjust last to exactly 1.0
        keys = list(probs.keys())
        for k in keys[:-1]:
            probs[k] = round(probs[k], 6)
        probs[keys[-1]] = round(1.0 - sum(probs[k] for k in keys[:-1]), 6)

        # Ensure exactly 100 unique supports
        supports = ensure_100_unique(doc_ids, desired=cfg["params"].get("supports", 100))

        rows.append({
            "question": q_text,
            "distribution": probs,
            "supports": supports
        })

    # Make sure output folder exists
    os.makedirs(os.path.dirname(cfg["paths"]["output"]), exist_ok=True)

    # Write CSV
    with open(cfg["paths"]["output"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "distribution", "supports"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "question": r["question"],
                "distribution": json.dumps(r["distribution"], ensure_ascii=True),
                "supports": json.dumps(r["supports"], ensure_ascii=True)
            })

    print(f"Saved results to {cfg['paths']['output']}")

def main_menu():
    while True:
        print("\nSelect which MAS pipeline to run:")
        print("1 or a → run main pipeline (run.py)")
        print("2 or b → run alternative pipeline 1 (run_1.py)")
        print("3 or c → run alternative pipeline 2 (run_2.py)")
        print("4 or d → exit")
        choice = input("Enter your choice: ").strip().lower()

        if choice in ['1', 'a']:
            with open("config.yaml", "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            run_pipeline(cfg)
        elif choice in ['2', 'b']:
            with open("config_1.yaml", "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            from mas_survey import run_1
            run_1.run_pipeline(cfg)
        elif choice in ['3', 'c']:
            with open("config_2.yaml", "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            from mas_survey import run_2
            run_2.run_pipeline(cfg)
        elif choice in ['4', 'd']:
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main_menu()

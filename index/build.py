# index/build.py
"""
Minimal placeholder for the index builder.

Keeps the CLI contract only (arguments parsing). No indexing logic.
Usage:
    python -m index.build --config config.yaml --api_key <YOUR_KEY>
"""
import argparse
import yaml
import os
import numpy as np
import faiss
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_sparse_index(documents_path, index_dir):
    from whoosh.index import create_in
    from whoosh.fields import Schema, TEXT, ID

    schema = Schema(id=ID(stored=True), content=TEXT(analyzer=StemmingAnalyzer()))
    os.makedirs(index_dir, exist_ok=True)
    ix = create_in(index_dir, schema)
    writer = ix.writer()

    with open(documents_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = eval(line.strip())  # safer: use json.loads
            doc_id = doc.get("id")
            text = "\n".join([doc.get(k, "") for k in ["title", "description", "post_content", "content"] if doc.get(k)])
            writer.add_document(id=doc_id, content=text)
    writer.commit()
    print(f"Sparse index built at {index_dir}")

def build_dense_index(embedding_path, faiss_index_path):
    data = np.load(embedding_path, allow_pickle=True)
    ids = data["ids"]
    embeddings = data["embeddings"]

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, faiss_index_path)
    np.save(faiss_index_path + "_ids.npy", ids)
    print(f"Dense index built at {faiss_index_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--api_key", required=False)  # accepted but unused
    args = parser.parse_args()

    cfg = load_config(args.config)

    build_sparse_index(
        documents_path=cfg["paths"]["documents_path"],
        index_dir=cfg["paths"]["sparse_index_path"]
    )

    build_dense_index(
        embedding_path=cfg["paths"]["embedding_path"],
        faiss_index_path=cfg["paths"]["dense_index_path"]
    )

if __name__ == "__main__":
    main()

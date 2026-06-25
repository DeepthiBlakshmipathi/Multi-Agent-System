import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def split_text(doc):
    """Combine all text fields of a document into a single string."""
    return "\n".join([
        doc.get("title", ""),
        doc.get("description", ""),
        doc.get("post_content", ""),
        doc.get("content", "")
    ])

class Retriever:
    def __init__(self, docs_path="data/documents.jsonl", emb_path=None):
        # Load documents
        with open(docs_path, "r", encoding="utf-8") as f:
            self.docs = [json.loads(line) for line in f]

        self.doc_ids = [doc["id"] for doc in self.docs]
        self.doc_texts = [split_text(doc) for doc in self.docs]

        # TF-IDF setup
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

        # Optional embeddings (if provided)
        self.embeddings = None
        if emb_path:
            arr = np.load(emb_path, allow_pickle=True)
            if "arr_0" in arr:
                self.embeddings = arr["arr_0"].item()  # dict: id -> embedding
            else:
                key = list(arr.keys())[0]
                self.embeddings = arr[key].item()

    def retrieve(self, query, top_k=100):
        """
        Retrieve top_k documents based on TF-IDF similarity.
        If embeddings are available, you can combine them (currently optional).
        """
        # Compute TF-IDF similarity
        query_vec = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Currently ignore embeddings for simplicity
        combined_scores = tfidf_scores

        # Rank documents
        ranked = sorted(zip(self.doc_ids, combined_scores), key=lambda x: x[1], reverse=True)
        top_ids = [doc_id for doc_id, _ in ranked[:top_k]]
        top_docs = [doc for doc in self.docs if doc["id"] in top_ids]

        return top_docs, top_ids, combined_scores[:top_k]

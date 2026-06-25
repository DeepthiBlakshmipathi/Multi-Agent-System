import json
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def score_overlap(question, doc_text):
    q_tokens = set(tokenize(question))
    d_tokens = set(tokenize(doc_text))
    return len(q_tokens & d_tokens)

def split_sentences(text):
    return re.split(r"[.!?]", text)

def rerank_supports(question, candidate_docs, top_k=100):
    question_embedding = model.encode(question, convert_to_tensor=True)

    doc_scores = []
    for doc in candidate_docs:
        doc_id = doc["id"]
        text = "\n".join([
            doc.get("title", ""),
            doc.get("description", ""),
            doc.get("post_content", ""),
            doc.get("content", "")
        ])
        sentences = split_sentences(text.lower())
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

        # Compute similarity for each sentence
        similarities = util.cos_sim(question_embedding, sentence_embeddings)[0]
        max_sim = float(similarities.max())

        # Metadata boost
        boost = 0.0
        subreddit = doc.get("subreddit", "").lower()
        if subreddit in {"askanamerican", "immigration", "askreddit"}:
            boost += 0.3
        for phrase in ["a great deal", "some", "not much", "nothing at all"]:
            if any(phrase in s for s in sentences):
                boost += 0.2

        final_score = 0.7 * max_sim + 0.3 * boost
        doc_scores.append((doc_id, final_score))

    # Rank and return top_k
    ranked = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    top_ids = [doc_id for doc_id, _ in ranked[:top_k]]
    return top_ids

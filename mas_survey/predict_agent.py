import numpy as np
import re

class Predictor:
    def __init__(self, seed=42, temperature=1.2, alpha=0.7):
        np.random.seed(seed)
        self.temperature = temperature
        self.alpha = alpha  # weight for TF-IDF vs uniform

    def predict(self, question, options, docs):
        counts = np.zeros(len(options))

        for i, opt in enumerate(options):
            for doc in docs:
                text = " ".join([doc.get(f, "") for f in ["title","description","post_content","content"]])
                # Simple match: count number of times option words appear
                opt_words = re.findall(r"\w+", opt.lower())
                doc_words = re.findall(r"\w+", text.lower())
                matches = sum(doc_words.count(w) for w in opt_words)
                counts[i] += matches

        # Add small smoothing to avoid zero probabilities
        counts = counts + 1e-3

        # Normalize
        probs = counts / counts.sum()

        # Temperature smoothing
        probs = probs ** (1 / self.temperature)
        probs = probs / probs.sum()

        return {opt: float(p) for opt, p in zip(options, probs)}

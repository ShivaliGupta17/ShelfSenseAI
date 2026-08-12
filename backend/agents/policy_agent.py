"""
Policy Agent — RAG over business policy documents.
Uses TF-IDF + cosine similarity for retrieval (no external embeddings API
required, works fully offline). Swap in a sentence-transformers /
FAISS vector store here for production without changing the interface.
"""
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_PATH = os.path.join(os.path.dirname(__file__), "..", "policies", "policies.json")


class PolicyAgent:
    def __init__(self):
        with open(POLICY_PATH) as f:
            self.docs = json.load(f)
        corpus = [d["title"] + ". " + d["text"] for d in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(corpus)

    def query(self, question: str, top_k: int = 2):
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.doc_matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        results = []
        for i in ranked:
            if sims[i] <= 0:
                continue
            results.append({
                "id": self.docs[i]["id"],
                "title": self.docs[i]["title"],
                "text": self.docs[i]["text"],
                "relevance": round(float(sims[i]), 3),
            })
        return results

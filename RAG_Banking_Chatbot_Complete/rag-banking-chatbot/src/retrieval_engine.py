"""
Lightweight TF-IDF retrieval engine for the banking knowledge base.

This mirrors the client-side retrieval logic used in demo_orange.html so the
backend API and the browser demo behave identically. In the full production
pipeline (see ingest.py / chatbot.py), this is swapped for real
sentence-transformer embeddings + an LLM via LlamaIndex — this module exists
so the REST API works instantly with zero downloads, for demo purposes.
"""
import json
import math
import re
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "banking_kb.json"

STOPWORDS = set((
    "a an the to of for and or is are was were be been being in on at by with "
    "as from this that these those it its your you can will may within up out "
    "per if not no do does using use used than into also which what when where "
    "how i my me"
).split())

CONFIDENCE_THRESHOLD = 0.055


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9%$.\s-]", " ", text)
    return [t for t in text.split() if len(t) > 1 and t not in STOPWORDS]


class RetrievalEngine:
    def __init__(self, data_path: Path = DATA_PATH):
        with open(data_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        self.docs = []
        for r in self.records:
            tokens = tokenize(f"{r['title']} {r['title']} {r['content']}")
            self.docs.append({**r, "tokens": tokens})

        # document frequency
        self.df = {}
        for d in self.docs:
            for t in set(d["tokens"]):
                self.df[t] = self.df.get(t, 0) + 1
        self.n = len(self.docs)

        for d in self.docs:
            d["vec"] = self._tf_vector(d["tokens"])

    def _idf(self, term: str) -> float:
        return math.log((self.n + 1) / (self.df.get(term, 0) + 1)) + 1

    def _tf_vector(self, tokens):
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        n_tokens = len(tokens) or 1
        return {t: (c / n_tokens) * self._idf(t) for t, c in tf.items()}

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def retrieve(self, query: str, k: int = 3):
        q_vec = self._tf_vector(tokenize(query))
        scored = [
            {**{key: d[key] for key in ("id", "category", "title", "content")},
             "score": self._cosine(q_vec, d["vec"])}
            for d in self.docs
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def ask(self, query: str, k: int = 3):
        hits = self.retrieve(query, k)
        if not hits or hits[0]["score"] < CONFIDENCE_THRESHOLD:
            return {
                "ok": False,
                "answer": ("I don't have that information in this knowledge base — "
                           "please contact customer service."),
                "cited_source": None,
                "sources": hits,
            }
        top = hits[0]
        return {
            "ok": True,
            "answer": top["content"],
            "cited_source": top["title"],
            "sources": hits,
        }

    def categories(self):
        counts = {}
        for d in self.docs:
            counts[d["category"]] = counts.get(d["category"], 0) + 1
        return counts

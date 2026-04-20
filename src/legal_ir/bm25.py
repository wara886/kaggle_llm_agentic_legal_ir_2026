import math
import re
from collections import Counter, defaultdict


TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: list[int] = []
        self.avg_doc_len = 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_freq: dict[str, int] = {}
        self.docs: list[dict] = []

    def build(self, docs: list[dict], text_key: str = "text") -> None:
        self.docs = docs
        for doc_id, doc in enumerate(docs):
            tokens = tokenize(doc.get(text_key, ""))
            freqs = Counter(tokens)
            self.doc_len.append(sum(freqs.values()))
            for token, tf in freqs.items():
                self.postings[token].append((doc_id, tf))
        self.avg_doc_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.doc_freq = {term: len(posts) for term, posts in self.postings.items()}

    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        q_tokens = tokenize(query)
        if not q_tokens or not self.docs:
            return []

        for term in q_tokens:
            idf = self._idf(term)
            if idf <= 0:
                continue
            for doc_id, tf in self.postings.get(term, []):
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / (self.avg_doc_len + 1e-9)))
                score = idf * (tf * (self.k1 + 1)) / (denom + 1e-9)
                scores[doc_id] += score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


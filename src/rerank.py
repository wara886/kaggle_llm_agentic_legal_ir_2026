from __future__ import annotations

from abc import ABC, abstractmethod

from legal_ir.bm25 import tokenize


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        raise NotImplementedError


class NoOpReranker(BaseReranker):
    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        return candidates[:top_n]


class TokenOverlapReranker(BaseReranker):
    """
    轻量 reranker 脚手架:
    - 不训练
    - 使用 query-token 与候选文本 token overlap 做二次排序
    """

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        qset = set(tokenize(query))
        rescored = []
        for c in candidates:
            # 兼容 baseline_v1_1 新候选字段（candidate fields）
            candidate_text = c.get("text") or c.get("candidate_text") or c.get("citation", "")
            tset = set(tokenize(candidate_text))
            overlap = len(qset & tset)
            base = float(c.get("fused_score", c.get("score", 0.0)))
            source_bonus = 0.005 if c.get("source") == "laws_de" else 0.0
            c2 = dict(c)
            c2["rerank_score"] = base + 0.02 * overlap + source_bonus
            rescored.append(c2)
        rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return rescored[:top_n]

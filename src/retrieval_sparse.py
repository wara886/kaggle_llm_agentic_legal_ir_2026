from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

from citation_normalizer import normalize_citation
from legal_ir.bm25 import BM25Index
from legal_ir.corpus_builder import iter_corpus_rows


@dataclass
class RetrievedItem:
    citation: str
    source: str
    score: float
    method: str


class SparseRetriever:
    def __init__(self, text_max_chars: int = 1000):
        self.text_max_chars = text_max_chars
        self.docs: dict[str, list[dict]] = {"laws_de": [], "court_considerations": []}
        self.indices: dict[str, BM25Index] = {}
        self.field_docs: dict[str, dict[str, list[dict]]] = {
            "laws_de": {"citation": [], "title": [], "text": []},
            "court_considerations": {"citation": [], "text": []},
        }
        self.field_indices: dict[str, dict[str, BM25Index]] = {"laws_de": {}, "court_considerations": {}}
        self.enable_field_aware = False
        # laws 标题模板统计（title_template_df）: 仅用于诊断/实验，不改变默认检索行为。
        self.laws_title_template_by_citation: dict[str, str] = {}
        self.laws_title_df_count: dict[str, int] = {}
        self.laws_title_df_ratio: dict[str, float] = {}

    @staticmethod
    def normalize_title_template(title: str) -> str:
        """
        标题模板归一化（title_template_normalization）：
        将高频法规标题头部压缩为可聚合模板，供 high_df penalty 实验使用。
        """
        text = (title or "").lower()
        text = (
            text.replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("ß", "ss")
        )
        head = text.split(" - ")[0]
        head = re.sub(r"\([^)]*\)", " ", head)
        head = re.sub(r"\b\d{1,4}\b", " <num> ", head)
        head = re.sub(r"[^a-z0-9<>\s]", " ", head)
        head = re.sub(r"\s+", " ", head).strip()
        return head

    def _build_laws_title_df_profile(self) -> None:
        self.laws_title_template_by_citation = {}
        self.laws_title_df_count = {}
        self.laws_title_df_ratio = {}
        laws_docs = self.docs.get("laws_de", [])
        if not laws_docs:
            return
        for d in laws_docs:
            citation = d["citation"]
            template = self.normalize_title_template(d.get("title", ""))
            self.laws_title_template_by_citation[citation] = template
            self.laws_title_df_count[template] = self.laws_title_df_count.get(template, 0) + 1
        total = float(len(laws_docs))
        for template, cnt in self.laws_title_df_count.items():
            self.laws_title_df_ratio[template] = cnt / total if total > 0 else 0.0

    def get_laws_title_df_features(self, citation: str) -> dict:
        template = self.laws_title_template_by_citation.get(citation, "")
        return {
            "title_template": template,
            "title_df_count": self.laws_title_df_count.get(template, 0),
            "title_df_ratio": self.laws_title_df_ratio.get(template, 0.0),
        }

    @staticmethod
    def extract_statute_family(citation: str) -> str:
        """
        法典家族抽取（statute_family_extraction）。
        示例：`Art. 221 Abs. 1 StPO` -> `STPO`
        """
        if not citation:
            return ""
        tokens = re.findall(r"[A-Za-z]{2,8}", citation)
        if not tokens:
            return ""
        # 优先取末尾近法典缩写模式。
        for tok in reversed(tokens):
            up = tok.upper()
            if up in {"ZGB", "OR", "BGG", "STPO", "BV", "ZPO", "SVG", "VVG", "DBG", "MWSTG"}:
                return up
        # 回退：最后一个字母 token。
        return tokens[-1].upper()

    @staticmethod
    def extract_citation_tokens(text: str) -> set[str]:
        """
        引用 token 提取（citation_token_extraction）。
        仅提取结构型 token 与缩写，供 overlap 信号使用。
        """
        if not text:
            return set()
        t = text.lower()
        out = set()
        for tok in re.findall(r"\b(?:art|abs|lit|ziff?|para|section|article)\b", t):
            out.add(tok)
        for tok in re.findall(r"\b[a-z]{2,8}\b", t):
            if tok.upper() in {"ZGB", "OR", "BGG", "STPO", "BV", "ZPO", "SVG", "VVG", "DBG", "MWSTG"}:
                out.add(tok.upper())
        # 保留少量数字 token，用于 Art./Abs. 结构匹配。
        for num in re.findall(r"\b\d{1,4}\b", t):
            out.add(num)
        return out

    @staticmethod
    def _join_text(row: dict, text_max_chars: int) -> str:
        title = row.get("title", "")
        text = row.get("text", "")
        body = text[:text_max_chars] if text_max_chars > 0 else text
        return f"{row['citation']} {title} {body}".strip()

    def build(
        self,
        max_laws_rows: int | None = None,
        max_court_rows: int | None = None,
        enable_field_aware: bool = False,
    ) -> dict:
        self.enable_field_aware = enable_field_aware
        self.docs = {"laws_de": [], "court_considerations": []}
        self.field_docs = {
            "laws_de": {"citation": [], "title": [], "text": []},
            "court_considerations": {"citation": [], "text": []},
        }
        for row in iter_corpus_rows(
            include_laws=True,
            include_court=True,
            max_laws_rows=max_laws_rows,
            max_court_rows=max_court_rows,
        ):
            citation = normalize_citation(row["citation"])
            if not citation:
                continue
            source = row["source"]
            self.docs[source].append(
                {
                    "citation": citation,
                    "source": source,
                    "text": self._join_text(row, self.text_max_chars),
                    "title": row.get("title", ""),
                    "raw_text": row.get("text", "")[: self.text_max_chars] if self.text_max_chars > 0 else row.get("text", ""),
                }
            )
            if enable_field_aware:
                if source == "laws_de":
                    self.field_docs[source]["citation"].append({"citation": citation, "source": source, "text": citation})
                    self.field_docs[source]["title"].append({"citation": citation, "source": source, "text": row.get("title", "")})
                    self.field_docs[source]["text"].append(
                        {"citation": citation, "source": source, "text": row.get("text", "")[: self.text_max_chars]}
                    )
                else:
                    self.field_docs[source]["citation"].append({"citation": citation, "source": source, "text": citation})
                    self.field_docs[source]["text"].append(
                        {"citation": citation, "source": source, "text": row.get("text", "")[: self.text_max_chars]}
                    )

        self.indices = {}
        for source, docs in self.docs.items():
            index = BM25Index()
            index.build(docs, text_key="text")
            self.indices[source] = index

        self.field_indices = {"laws_de": {}, "court_considerations": {}}
        if enable_field_aware:
            for source, fmap in self.field_docs.items():
                for field, fdocs in fmap.items():
                    idx = BM25Index()
                    idx.build(fdocs, text_key="text")
                    self.field_indices[source][field] = idx

        # 仅构建可观测统计，不改变主流程排序行为。
        self._build_laws_title_df_profile()

        return {
            "laws_docs": len(self.docs["laws_de"]),
            "court_docs": len(self.docs["court_considerations"]),
            "text_max_chars": self.text_max_chars,
            "enable_field_aware": enable_field_aware,
        }

    def search(self, query: str, top_k_laws: int = 50, top_k_court: int = 50) -> list[RetrievedItem]:
        out: list[RetrievedItem] = []
        for source, top_k in [("laws_de", top_k_laws), ("court_considerations", top_k_court)]:
            index = self.indices[source]
            docs = self.docs[source]
            for doc_id, score in index.search(query, top_k=top_k):
                out.append(
                    RetrievedItem(
                        citation=docs[doc_id]["citation"],
                        source=source,
                        score=float(score),
                        method="sparse_bm25",
                    )
                )
        return out

    def search_multi_view(
        self,
        query_original: str,
        query_keywords: list[str] | None = None,
        expanded_query_de: str | None = None,
        top_k_laws: int = 50,
        top_k_court: int = 50,
    ) -> list[RetrievedItem]:
        """
        多视图稀疏检索（multi_view_sparse_retrieval）：
        - query_original
        - query_keywords
        - expanded_query_de
        """
        query_keywords = query_keywords or []
        view_queries = [query_original]
        if query_keywords:
            view_queries.append(" ".join(query_keywords[:16]))
        if expanded_query_de:
            view_queries.append(expanded_query_de)

        agg: dict[tuple[str, str], RetrievedItem] = {}
        for view_idx, q in enumerate(view_queries):
            if not q:
                continue
            view_items = self.search(q, top_k_laws=top_k_laws, top_k_court=top_k_court)
            boost = 1.0 if view_idx == 0 else 0.9
            for it in view_items:
                key = (it.citation, it.source)
                new_score = float(it.score) * boost
                prev = agg.get(key)
                if prev is None or new_score > prev.score:
                    agg[key] = RetrievedItem(
                        citation=it.citation,
                        source=it.source,
                        score=new_score,
                        method="sparse_bm25_multi_view",
                    )
        return sorted(agg.values(), key=lambda x: x.score, reverse=True)

    def search_field_aware(
        self,
        laws_query_pack: dict,
        court_query_pack: dict,
        laws_query_pack_v2: dict | None = None,
        enable_laws_query_pack_v2: bool = False,
        top_k_laws: int = 50,
        top_k_court: int = 50,
        laws_citation_weight: float = 1.0,
        laws_title_weight: float = 1.0,
        laws_text_weight: float = 1.0,
        court_citation_weight: float = 1.0,
        court_text_weight: float = 1.0,
    ) -> list[RetrievedItem]:
        """
        字段感知检索（field_aware_retrieval）：
        - laws: citation/title/text
        - court: citation/text
        """
        if not self.enable_field_aware:
            # 向后兼容：未构建 field index 时回退 multi_view
            return self.search_multi_view(
                query_original=laws_query_pack.get("query_original", ""),
                query_keywords=laws_query_pack.get("query_keywords", []),
                expanded_query_de=laws_query_pack.get("expanded_query_de", ""),
                top_k_laws=top_k_laws,
                top_k_court=top_k_court,
            )

        agg: dict[tuple[str, str], RetrievedItem] = {}

        def _consume(source: str, field: str, query: str, weight: float, top_k: int):
            if not query or weight == 0:
                return
            idx = self.field_indices[source].get(field)
            docs = self.field_docs[source].get(field, [])
            if idx is None:
                return
            for doc_id, score in idx.search(query, top_k=top_k):
                d = docs[doc_id]
                key = (d["citation"], source)
                s = float(score) * float(weight)
                prev = agg.get(key)
                if prev is None or s > prev.score:
                    agg[key] = RetrievedItem(
                        citation=d["citation"],
                        source=source,
                        score=s,
                        method=f"sparse_field_{field}",
                    )

        effective_laws_pack = laws_query_pack_v2 if (enable_laws_query_pack_v2 and laws_query_pack_v2) else laws_query_pack

        laws_query = " ".join(
            [
                effective_laws_pack.get("query_original", ""),
                " ".join(effective_laws_pack.get("query_keywords", [])[:16]),
                effective_laws_pack.get("expanded_query_de", ""),
            ]
        ).strip()
        # 裁判查询增强（court_query_enhanced）：补充编号锚点与德语扩展关键词，提升 court seed 命中率。
        court_query = " ".join(
            [
                court_query_pack.get("query_original", ""),
                " ".join(court_query_pack.get("query_keywords", [])[:16]),
                " ".join(court_query_pack.get("query_legal_phrases", [])[:8]),
                " ".join(court_query_pack.get("query_number_patterns", [])[:12]),
                " ".join(court_query_pack.get("expanded_keywords_de", [])[:16]),
                court_query_pack.get("expanded_query_de", ""),
            ]
        ).strip()

        _consume("laws_de", "citation", laws_query, laws_citation_weight, top_k_laws)
        _consume("laws_de", "title", laws_query, laws_title_weight, top_k_laws)
        _consume("laws_de", "text", laws_query, laws_text_weight, top_k_laws)

        _consume("court_considerations", "citation", court_query, court_citation_weight, top_k_court)
        _consume("court_considerations", "text", court_query, court_text_weight, top_k_court)

        return sorted(agg.values(), key=lambda x: x.score, reverse=True)

    def search_route_aware(
        self,
        laws_query_pack: dict,
        court_query_pack: dict,
        laws_top_k: int,
        court_top_k: int,
        laws_query_pack_v2: dict | None = None,
        enable_laws_query_pack_v2: bool = False,
        laws_citation_weight: float = 1.0,
        laws_title_weight: float = 1.0,
        laws_text_weight: float = 1.0,
        court_citation_weight: float = 1.0,
        court_text_weight: float = 1.0,
    ) -> list[RetrievedItem]:
        """
        路由感知配额检索（route_aware_quota_search）：
        仅暴露 laws/court top-k 接口，内部复用 field-aware 检索。
        """
        return self.search_field_aware(
            laws_query_pack=laws_query_pack,
            court_query_pack=court_query_pack,
            laws_query_pack_v2=laws_query_pack_v2,
            enable_laws_query_pack_v2=enable_laws_query_pack_v2,
            top_k_laws=max(0, int(laws_top_k)),
            top_k_court=max(0, int(court_top_k)),
            laws_citation_weight=laws_citation_weight,
            laws_title_weight=laws_title_weight,
            laws_text_weight=laws_text_weight,
            court_citation_weight=court_citation_weight,
            court_text_weight=court_text_weight,
        )

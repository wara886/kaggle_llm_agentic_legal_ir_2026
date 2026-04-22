from __future__ import annotations

import math
from dataclasses import dataclass

from citation_normalizer import normalize_citation
from legal_ir.corpus_builder import iter_corpus_rows

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


@dataclass
class DenseRetrievedItem:
    citation: str
    source: str
    score: float
    method: str


class DenseRetriever:
    """
    Multilingual semantic retrieval:
    - Preferred backend: SentenceTransformer (multilingual model)
    - Fallback backend: Hashing + SVD latent semantic vectors (no training labels)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        use_sbert: bool = True,
        text_max_chars: int = 600,
        svd_dim: int = 256,
    ):
        self.model_name = model_name
        self.use_sbert = use_sbert and SentenceTransformer is not None
        self.text_max_chars = text_max_chars
        self.svd_dim = svd_dim

        self.backend = "sbert" if self.use_sbert else "hashing_svd"
        self.docs: dict[str, list[dict]] = {"laws_de": [], "court_considerations": []}
        self.doc_matrix: dict[str, object] = {}
        self.source_views: dict[str, list[dict]] = {"laws_de": [], "court_considerations": []}
        self.source_view_matrix: dict[str, object] = {}
        self.vectorizer = None
        self.svd = None
        self.model = None
        self.enable_field_aware = False

    @staticmethod
    def _join_text(row: dict, text_max_chars: int, source: str, enable_field_aware: bool) -> str:
        title = row.get("title", "")
        text = row.get("text", "")
        body = text[:text_max_chars] if text_max_chars > 0 else text
        if enable_field_aware:
            if source == "laws_de":
                # 法规侧优先 title + text
                return f"{title} {body} {row['citation']}".strip()
            # 裁判侧优先 text
            return f"{body} {row['citation']}".strip()
        return f"{row['citation']} {title} {body}".strip()

    def _fit_sbert(self, all_texts: list[str]) -> None:
        self.model = SentenceTransformer(self.model_name)
        embeddings = self.model.encode(
            all_texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.doc_matrix["all"] = embeddings

    def _fit_hashing_svd(self, all_texts: list[str]) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=2**18,
            alternate_sign=False,
            lowercase=True,
            norm=None,
            ngram_range=(1, 2),
        )
        x = self.vectorizer.transform(all_texts)
        rank_cap = max(8, min(self.svd_dim, x.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=rank_cap, random_state=42)
        dense = self.svd.fit_transform(x)
        self.doc_matrix["all"] = normalize(dense)

    def build(
        self,
        max_laws_rows: int | None = 80000,
        max_court_rows: int | None = 120000,
        enable_field_aware: bool = False,
    ) -> dict:
        self.enable_field_aware = enable_field_aware
        self.docs = {"laws_de": [], "court_considerations": []}
        self.source_views = {"laws_de": [], "court_considerations": []}
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
                    "text": self._join_text(row, self.text_max_chars, source, enable_field_aware=False),
                    "combined_text_view": self._join_text(row, self.text_max_chars, source, enable_field_aware=True),
                }
            )
            self.source_views[source].append(
                {
                    "citation": citation,
                    "source": source,
                    "text": self._join_text(row, self.text_max_chars, source, enable_field_aware=enable_field_aware),
                }
            )

        all_docs = self.docs["laws_de"] + self.docs["court_considerations"]
        all_texts = [d["text"] for d in all_docs]
        self.doc_matrix["all_docs"] = all_docs

        if self.backend == "sbert":
            try:
                self._fit_sbert(all_texts)
            except Exception:
                self.backend = "hashing_svd"
                self._fit_hashing_svd(all_texts)
        else:
            self._fit_hashing_svd(all_texts)

        # source specific matrix for field-aware route
        self.source_view_matrix = {}
        laws_docs = self.source_views["laws_de"]
        court_docs = self.source_views["court_considerations"]
        self.source_view_matrix["laws_docs"] = laws_docs
        self.source_view_matrix["court_docs"] = court_docs
        if self.backend == "sbert":
            try:
                self.source_view_matrix["laws_mat"] = self.model.encode(
                    [d["text"] for d in laws_docs],
                    batch_size=64,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                self.source_view_matrix["court_mat"] = self.model.encode(
                    [d["text"] for d in court_docs],
                    batch_size=64,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception:
                self.backend = "hashing_svd"
                self._fit_hashing_svd(all_texts)
        if self.backend != "sbert":
            # hashing_svd 下重建 source matrix
            laws_x = self.vectorizer.transform([d["text"] for d in laws_docs])
            court_x = self.vectorizer.transform([d["text"] for d in court_docs])
            self.source_view_matrix["laws_mat"] = normalize(self.svd.transform(laws_x))
            self.source_view_matrix["court_mat"] = normalize(self.svd.transform(court_x))

        return {
            "backend": self.backend,
            "laws_docs": len(self.docs["laws_de"]),
            "court_docs": len(self.docs["court_considerations"]),
            "text_max_chars": self.text_max_chars,
            "enable_field_aware": enable_field_aware,
        }

    def _encode_query(self, query: str):
        if self.backend == "sbert":
            emb = self.model.encode(
                [query],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return emb[0]
        q = self.vectorizer.transform([query])
        q2 = self.svd.transform(q)
        return normalize(q2)[0]

    @staticmethod
    def _dot(a, b) -> float:
        return float((a * b).sum())

    def search(self, query: str, top_k_laws: int = 40, top_k_court: int = 40) -> list[DenseRetrievedItem]:
        qvec = self._encode_query(query)
        matrix = self.doc_matrix["all"]
        all_docs = self.doc_matrix["all_docs"]
        scores = matrix @ qvec

        ranked_idx = sorted(range(len(all_docs)), key=lambda i: float(scores[i]), reverse=True)
        out: list[DenseRetrievedItem] = []
        quota = {"laws_de": top_k_laws, "court_considerations": top_k_court}
        kept = {"laws_de": 0, "court_considerations": 0}

        for i in ranked_idx:
            doc = all_docs[i]
            source = doc["source"]
            if kept[source] >= quota[source]:
                continue
            out.append(
                DenseRetrievedItem(
                    citation=doc["citation"],
                    source=source,
                    score=float(scores[i]),
                    method=f"dense_{self.backend}",
                )
            )
            kept[source] += 1
            if kept["laws_de"] >= top_k_laws and kept["court_considerations"] >= top_k_court:
                break
        return out

    def search_multi_view(
        self,
        bilingual_query_pack: dict,
        top_k_laws: int = 40,
        top_k_court: int = 40,
    ) -> list[DenseRetrievedItem]:
        """
        多视图稠密检索（multi_view_dense_retrieval）：
        输入 bilingual_query_pack，融合英文与德文扩展视图。
        """
        query_views = [
            bilingual_query_pack.get("query_en_original", ""),
            bilingual_query_pack.get("query_en_clean", ""),
            bilingual_query_pack.get("query_de_expanded", ""),
        ]
        legal_phrases = bilingual_query_pack.get("query_en_legal_phrases", []) or []
        if legal_phrases:
            query_views.append(" ".join(legal_phrases[:8]))

        agg: dict[tuple[str, str], DenseRetrievedItem] = {}
        for idx, q in enumerate(query_views):
            if not q:
                continue
            boost = 1.0 if idx == 0 else 0.9
            items = self.search(query=q, top_k_laws=top_k_laws, top_k_court=top_k_court)
            for it in items:
                key = (it.citation, it.source)
                new_score = float(it.score) * boost
                prev = agg.get(key)
                if prev is None or new_score > prev.score:
                    agg[key] = DenseRetrievedItem(
                        citation=it.citation,
                        source=it.source,
                        score=new_score,
                        method=f"{it.method}_multi_view",
                    )
        return sorted(agg.values(), key=lambda x: x.score, reverse=True)

    def search_source_aware(
        self,
        laws_query_pack: dict,
        court_query_pack: dict,
        top_k_laws: int = 40,
        top_k_court: int = 40,
    ) -> list[DenseRetrievedItem]:
        """
        字段感知稠密检索（field_aware_dense_retrieval）：
        - laws 使用 title+text 组合视图
        - court 使用 text 视图
        - 支持 source-specific query pack
        """
        laws_query = " ".join(
            [
                laws_query_pack.get("query_clean", ""),
                " ".join(laws_query_pack.get("query_legal_phrases", [])[:8]),
                " ".join(laws_query_pack.get("expanded_keywords_de", [])[:16]),
                laws_query_pack.get("expanded_query_de", ""),
            ]
        ).strip()
        court_query = " ".join(
            [
                court_query_pack.get("query_clean", ""),
                " ".join(court_query_pack.get("query_legal_phrases", [])[:8]),
                " ".join(court_query_pack.get("query_number_patterns", [])[:12]),
                " ".join(court_query_pack.get("expanded_keywords_de", [])[:16]),
                court_query_pack.get("expanded_query_de", ""),
            ]
        ).strip()

        def _encode(q: str):
            if self.backend == "sbert":
                emb = self.model.encode(
                    [q],
                    batch_size=1,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return emb[0]
            qx = self.vectorizer.transform([q])
            return normalize(self.svd.transform(qx))[0]

        out: list[DenseRetrievedItem] = []
        if laws_query and len(self.source_view_matrix.get("laws_docs", [])) > 0:
            qv = _encode(laws_query)
            mat = self.source_view_matrix["laws_mat"]
            docs = self.source_view_matrix["laws_docs"]
            scores = mat @ qv
            for i in sorted(range(len(docs)), key=lambda x: float(scores[x]), reverse=True)[:top_k_laws]:
                out.append(
                    DenseRetrievedItem(
                        citation=docs[i]["citation"],
                        source="laws_de",
                        score=float(scores[i]),
                        method=f"dense_{self.backend}_source_aware",
                    )
                )
        if court_query and len(self.source_view_matrix.get("court_docs", [])) > 0:
            qv = _encode(court_query)
            mat = self.source_view_matrix["court_mat"]
            docs = self.source_view_matrix["court_docs"]
            scores = mat @ qv
            for i in sorted(range(len(docs)), key=lambda x: float(scores[x]), reverse=True)[:top_k_court]:
                out.append(
                    DenseRetrievedItem(
                        citation=docs[i]["citation"],
                        source="court_considerations",
                        score=float(scores[i]),
                        method=f"dense_{self.backend}_source_aware",
                    )
                )
        return out

    def search_court_multi_view(
        self,
        bilingual_query_pack: dict,
        top_k_court: int = 80,
    ) -> list[DenseRetrievedItem]:
        """
        court 专用稠密检索（court_dense_retrieval）：
        - 只在 court_considerations 侧做语义召回
        - 复用 multi_view 查询输入
        """
        items = self.search_multi_view(
            bilingual_query_pack=bilingual_query_pack,
            top_k_laws=0,
            top_k_court=top_k_court,
        )
        return [x for x in items if x.source == "court_considerations"]

import atexit
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from rag.providers.embeddings.factory import create_embedding_provider
from rag.providers.rerankers.factory import create_reranker_provider
from rag.types import RetrievalConfig, RetrievalFilters, RetrievalResult, RetrievedChunk
from rag.utils import normalize_scores, tokenize
from weaviate_store import (
    create_weaviate_client,
    ensure_weaviate_collection,
    get_chunk_collection_name,
    get_document_collection_name,
    get_section_collection_name,
)

from settings import settings


@dataclass(frozen=True)
class QueryRoute:
    intent: str
    parent_level: str
    alpha: float
    top_k: int
    parent_top_k: int
    operator_mode: str
    filter_strictness: str
    lexical_query: str
    must_match_tokens: tuple[str, ...]


class RetrievalEngine:
    MUST_MATCH_PATTERN = re.compile(
        r"(?iu)(?:[\w.-]+\.(?:pdf|docx|txt|xlsx|xls))|(?:[\w./\\-]*[/\\][\w./\\-]+)|(?:[a-zа-я]*\d+[a-zа-я\d._/-]*)"
    )

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.document_lookup_pattern = re.compile(
            config.document_lookup_pattern, re.IGNORECASE
        )
        self.entity_lookup_pattern = re.compile(
            config.entity_lookup_pattern, re.IGNORECASE
        )
        ensure_weaviate_collection(recreate=False)
        self.client = create_weaviate_client()
        atexit.register(self.client.close)
        self.chunk_collection = self.client.collections.use(get_chunk_collection_name())
        self.section_collection = self.client.collections.use(get_section_collection_name())
        self.document_collection = self.client.collections.use(
            get_document_collection_name()
        )
        self.embedding_provider = create_embedding_provider()
        self.reranker = create_reranker_provider() if config.reranker_enabled else None

    def detect_query_intent(self, query: str) -> str:
        if self.document_lookup_pattern.search(query):
            return "document_lookup"
        if self._is_entity_lookup_query(query):
            return "document_lookup"
        return "qa"

    def extract_reference_tokens(self, query: str) -> set[str]:
        return {
            token
            for token in tokenize(query)
            if token not in self.config.reference_stopwords
        }

    def _is_entity_lookup_query(self, query: str) -> bool:
        reference_tokens = self.extract_reference_tokens(query)
        if not reference_tokens:
            return False
        if len(reference_tokens) > self.config.entity_query_max_tokens:
            return False
        return bool(self.entity_lookup_pattern.search(query))

    def _build_reranker_passage(self, item: RetrievedChunk) -> str:
        return f"Title: {item.title}\n" f"Path: {item.path}\n" f"Text: {item.text}"

    def _path_penalty(self, path: str) -> float:
        lower_path = path.lower()
        if any(marker in lower_path for marker in self.config.noisy_path_markers):
            return self.config.noisy_path_penalty
        return 0.0

    def _encode_query(self, query: str):
        return self.embedding_provider.encode([query])

    def _import_weaviate(self):
        try:
            import weaviate.classes as wvc
        except ImportError as exc:
            raise RuntimeError(
                "Пакет weaviate-client не установлен. "
                "Установи зависимости заново через `uv sync`."
            ) from exc

        return wvc

    def _get_hybrid_fusion(self):
        wvc = self._import_weaviate()
        fusion_name = self.config.hybrid_fusion.strip().upper()
        try:
            return getattr(wvc.query.HybridFusion, fusion_name)
        except AttributeError:
            return wvc.query.HybridFusion.RELATIVE_SCORE

    def _build_hybrid_query_properties(self) -> list[str]:
        return [
            f"title^{self.config.bm25_title_weight}",
            f"path^{self.config.bm25_path_weight}",
            f"text^{self.config.bm25_text_weight}",
        ]

    def _dedupe_tokens(self, tokens: list[str]) -> list[str]:
        seen: set[str] = set()
        unique_tokens: list[str] = []
        for token in tokens:
            normalized = token.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_tokens.append(normalized)
        return unique_tokens

    def _extract_significant_lexical_tokens(self, query: str) -> list[str]:
        return self._dedupe_tokens(
            [
                token
                for token in tokenize(query)
                if token not in self.config.lexical_stopwords
            ]
        )

    def _extract_must_match_tokens(
        self,
        query: str,
        *,
        intent: str,
        significant_tokens: list[str],
    ) -> list[str]:
        must_match_tokens = self._dedupe_tokens(self.MUST_MATCH_PATTERN.findall(query))
        if intent == "document_lookup" and 0 < len(significant_tokens) <= 2:
            must_match_tokens.extend(significant_tokens)
        return self._dedupe_tokens(must_match_tokens)

    def _looks_like_reference_lookup(
        self,
        query: str,
        *,
        must_match_tokens: list[str],
    ) -> bool:
        lower_query = query.lower()
        if "№" in query or " no " in f" {lower_query} ":
            return True
        return any(
            any(marker in token for marker in ("/", "\\", ".", "-"))
            or any(char.isdigit() for char in token)
            for token in must_match_tokens
        )

    def _is_broad_topic_query(self, query: str, *, significant_tokens: list[str]) -> bool:
        lowered = query.lower()
        broad_markers = (
            "расскажи про",
            "расскажи о",
            "в целом",
            "в общем",
            "обзор",
            "какие есть",
            "что известно",
            "что знаешь",
            "всё про",
            "все про",
        )
        return len(significant_tokens) >= 7 or any(
            marker in lowered for marker in broad_markers
        )

    def _build_lexical_query(
        self,
        query: str,
        *,
        intent: str,
        significant_tokens: list[str],
        must_match_tokens: list[str],
    ) -> str:
        if intent in {"document_lookup", "reference_lookup"} and must_match_tokens:
            return " ".join(must_match_tokens)
        if significant_tokens:
            return " ".join(significant_tokens)
        return query

    def _plan_query_route(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
    ) -> QueryRoute:
        del filters  # Reserved for future route decisions based on provided metadata scope.

        significant_tokens = self._extract_significant_lexical_tokens(query)
        reference_candidates = self._extract_must_match_tokens(
            query,
            intent="document_lookup",
            significant_tokens=significant_tokens,
        )

        if self._looks_like_reference_lookup(
            query, must_match_tokens=reference_candidates
        ):
            intent = "reference_lookup"
        elif self.document_lookup_pattern.search(query):
            intent = "document_lookup"
        elif self._is_entity_lookup_query(query) or self._is_broad_topic_query(
            query, significant_tokens=significant_tokens
        ):
            intent = "broad_topic"
        else:
            intent = "qa"

        must_match_tokens = (
            self._extract_must_match_tokens(
                query,
                intent="document_lookup",
                significant_tokens=significant_tokens,
            )
            if intent in {"document_lookup", "reference_lookup"}
            else []
        )
        lexical_query = self._build_lexical_query(
            query,
            intent=intent,
            significant_tokens=significant_tokens,
            must_match_tokens=must_match_tokens,
        )

        if intent == "reference_lookup":
            return QueryRoute(
                intent=intent,
                parent_level="section",
                alpha=0.35,
                top_k=50,
                parent_top_k=12,
                operator_mode="strict_and",
                filter_strictness="strict",
                lexical_query=lexical_query,
                must_match_tokens=tuple(must_match_tokens),
            )
        if intent == "document_lookup":
            return QueryRoute(
                intent=intent,
                parent_level="section",
                alpha=0.45,
                top_k=70,
                parent_top_k=16,
                operator_mode="minimum_match",
                filter_strictness="strict",
                lexical_query=lexical_query,
                must_match_tokens=tuple(must_match_tokens),
            )
        if intent == "broad_topic":
            return QueryRoute(
                intent=intent,
                parent_level="document",
                alpha=0.75,
                top_k=120,
                parent_top_k=12,
                operator_mode="relaxed_or",
                filter_strictness="relaxed",
                lexical_query=lexical_query,
                must_match_tokens=(),
            )

        return QueryRoute(
            intent="qa",
            parent_level="section",
            alpha=self.config.hybrid_alpha,
            top_k=80,
            parent_top_k=18,
            operator_mode="adaptive",
            filter_strictness="balanced",
            lexical_query=lexical_query,
            must_match_tokens=(),
        )

    def _build_bm25_operator(self, lexical_query: str, *, route: QueryRoute):
        wvc = self._import_weaviate()
        lexical_tokens = tokenize(lexical_query)
        if len(lexical_tokens) <= 1:
            return None

        if route.operator_mode == "strict_and":
            return wvc.query.BM25Operator.and_()

        if route.operator_mode == "minimum_match":
            if any(marker in lexical_query for marker in ("/", "\\", ".", "-")):
                return wvc.query.BM25Operator.and_()
            if len(lexical_tokens) <= 3:
                return wvc.query.BM25Operator.and_()
            minimum_match = max(2, math.ceil(len(lexical_tokens) * 0.6))
            return wvc.query.BM25Operator.or_(min(minimum_match, len(lexical_tokens)))

        if route.operator_mode == "relaxed_or":
            minimum_match = max(2, math.ceil(len(lexical_tokens) * 0.4))
            return wvc.query.BM25Operator.or_(min(minimum_match, len(lexical_tokens)))

        if len(lexical_tokens) >= 4:
            minimum_match = max(2, math.ceil(len(lexical_tokens) * 0.5))
            return wvc.query.BM25Operator.or_(min(minimum_match, len(lexical_tokens)))

        return None

    def _normalize_filter_values(self, values: tuple[str, ...]) -> list[str]:
        normalized_values: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized_values.append(normalized)
        return normalized_values

    def _normalize_filter_datetime(self, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _build_any_of_equal_filter(self, property_name: str, values: list[str]):
        wvc = self._import_weaviate()
        if not values:
            return None
        if len(values) == 1:
            return wvc.query.Filter.by_property(property_name).equal(values[0])
        return wvc.query.Filter.any_of(
            [
                wvc.query.Filter.by_property(property_name).equal(value)
                for value in values
            ]
        )

    def _build_metadata_filters(
        self,
        filters: RetrievalFilters | None,
        *,
        strictness: str,
    ):
        if filters is None:
            return None

        # Filter strictness is already routed explicitly so the retrieval path
        # can evolve per-intent without changing the API contract.
        del strictness
        wvc = self._import_weaviate()
        clauses = []

        domains = self._normalize_filter_values(filters.domains)
        if domains:
            clauses.append(self._build_any_of_equal_filter("domain", domains))

        departments = self._normalize_filter_values(filters.departments)
        if departments:
            clauses.append(self._build_any_of_equal_filter("department", departments))

        doc_types = self._normalize_filter_values(filters.doc_types)
        if doc_types:
            clauses.append(self._build_any_of_equal_filter("doc_type", doc_types))

        extensions = self._normalize_filter_values(filters.extensions)
        if extensions:
            clauses.append(self._build_any_of_equal_filter("extension", extensions))

        languages = self._normalize_filter_values(filters.languages)
        if languages:
            clauses.append(self._build_any_of_equal_filter("language", languages))

        acl_groups = self._normalize_filter_values(filters.acl_groups)
        if acl_groups:
            clauses.append(
                wvc.query.Filter.by_property("acl_groups").contains_any(acl_groups)
            )

        created_from = self._normalize_filter_datetime(filters.created_from)
        if created_from is not None:
            clauses.append(
                wvc.query.Filter.by_property("created_at").greater_or_equal(created_from)
            )

        created_to = self._normalize_filter_datetime(filters.created_to)
        if created_to is not None:
            clauses.append(
                wvc.query.Filter.by_property("created_at").less_or_equal(created_to)
            )

        if not clauses:
            return None

        return wvc.query.Filter.all_of(clauses)

    def _hybrid_limit(self, *, route: QueryRoute) -> int:
        requested_limit = max(
            route.top_k,
            self.config.reranker_top_k,
            self.config.final_top_k,
            self.config.doc_lookup_final_top_k,
        )
        requested_limit = max(requested_limit, 50)
        return min(requested_limit, 200)

    def _parent_limit(self, *, route: QueryRoute) -> int:
        requested_limit = max(route.parent_top_k, 5)
        return min(requested_limit, 50)

    def _build_ranked_chunks_from_objects(
        self,
        *,
        objects: list,
        query: str,
        intent: str,
    ) -> list[RetrievedChunk]:
        reference_tokens = self.extract_reference_tokens(query)
        ranked: list[RetrievedChunk] = []

        for obj in objects:
            properties = obj.properties or {}
            chunk_id = properties.get("chunk_id")
            if chunk_id is None:
                continue

            doc_id = int(chunk_id)
            path = str(properties.get("path", "") or "")
            title = str(properties.get("title", "") or "")
            text = str(properties.get("text", "") or "")
            page_start = int(properties.get("page_start", 0) or 0) or None
            page_end = int(properties.get("page_end", 0) or 0) or None
            source_locator = str(properties.get("source_locator", "") or "")
            title_tokens = set(tokenize(title))
            path_tokens = set(tokenize(path))

            dense_distance = getattr(obj.metadata, "distance", None)
            dense_score = (
                max(0.0, 1.0 - float(dense_distance))
                if dense_distance is not None
                else 0.0
            )
            raw_hybrid_score = getattr(obj.metadata, "score", None)
            hybrid_score = (
                float(raw_hybrid_score) if raw_hybrid_score is not None else 0.0
            )

            title_coverage = (
                len(reference_tokens.intersection(title_tokens)) / len(reference_tokens)
                if reference_tokens
                else 0.0
            )
            path_coverage = (
                len(reference_tokens.intersection(path_tokens)) / len(reference_tokens)
                if reference_tokens
                else 0.0
            )
            all_reference_tokens = title_tokens | path_tokens
            exact_reference_match = (
                1.0
                if reference_tokens and reference_tokens.issubset(all_reference_tokens)
                else 0.0
            )
            path_penalty = self._path_penalty(path)

            # Keep the retrieval stage simple: native hybrid score from Weaviate,
            # plus only a small penalty for obviously noisy paths.
            final_score = hybrid_score - path_penalty

            ranked.append(
                RetrievedChunk(
                    chunk_id=doc_id,
                    path=path,
                    title=title,
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    source_locator=source_locator,
                    dense_score=dense_score,
                    bm25_score=hybrid_score,
                    raw_bm25=None,
                    reranker_score=None,
                    title_coverage=title_coverage,
                    path_coverage=path_coverage,
                    exact_reference_match=exact_reference_match,
                    score=final_score,
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    def _run_hybrid_query(
        self,
        *,
        collection,
        query: str,
        route: QueryRoute,
        filters,
        limit: int,
        return_properties: list[str],
    ):
        wvc = self._import_weaviate()
        query_embedding = self._encode_query(query)[0].tolist()
        return collection.query.hybrid(
            query=route.lexical_query,
            vector=query_embedding,
            alpha=route.alpha,
            fusion_type=self._get_hybrid_fusion(),
            bm25_operator=self._build_bm25_operator(
                route.lexical_query,
                route=route,
            ),
            filters=filters,
            limit=limit,
            query_properties=self._build_hybrid_query_properties(),
            return_properties=return_properties,
            return_metadata=wvc.query.MetadataQuery(
                score=True,
                distance=True,
                explain_score=True,
            ),
        )

    def _collect_parent_ids(self, response_objects: list, *, route: QueryRoute) -> list[int]:
        key = "document_id" if route.parent_level == "document" else "section_id"
        parent_ids: list[int] = []
        seen: set[int] = set()
        for obj in response_objects:
            properties = obj.properties or {}
            value = properties.get(key)
            if value is None:
                continue
            parent_id = int(value)
            if parent_id in seen:
                continue
            seen.add(parent_id)
            parent_ids.append(parent_id)
        return parent_ids

    def _build_parent_scope_filter(self, parent_ids: list[int], *, route: QueryRoute):
        wvc = self._import_weaviate()
        if not parent_ids:
            return None

        property_name = "document_id" if route.parent_level == "document" else "section_id"
        if len(parent_ids) == 1:
            return wvc.query.Filter.by_property(property_name).equal(parent_ids[0])
        return wvc.query.Filter.any_of(
            [
                wvc.query.Filter.by_property(property_name).equal(parent_id)
                for parent_id in parent_ids
            ]
        )

    def _combine_filters(self, *filters):
        wvc = self._import_weaviate()
        non_empty = [filter_ for filter_ in filters if filter_ is not None]
        if not non_empty:
            return None
        if len(non_empty) == 1:
            return non_empty[0]
        return wvc.query.Filter.all_of(non_empty)

    def _retrieve_parent_ids(
        self,
        query: str,
        *,
        route: QueryRoute,
        metadata_filters,
    ) -> list[int]:
        parent_collection = (
            self.document_collection
            if route.parent_level == "document"
            else self.section_collection
        )
        parent_properties = (
            ["document_id", "path", "title", "text"]
            if route.parent_level == "document"
            else [
                "section_id",
                "document_id",
                "path",
                "title",
                "text",
                "page_start",
                "page_end",
                "source_locator",
            ]
        )
        response = self._run_hybrid_query(
            collection=parent_collection,
            query=query,
            route=route,
            filters=metadata_filters,
            limit=self._parent_limit(route=route),
            return_properties=parent_properties,
        )
        return self._collect_parent_ids(response.objects, route=route)

    def hybrid_search(
        self,
        query: str,
        *,
        route: QueryRoute,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        metadata_filters = self._build_metadata_filters(
            filters,
            strictness=route.filter_strictness,
        )
        parent_ids = self._retrieve_parent_ids(
            query,
            route=route,
            metadata_filters=metadata_filters,
        )
        if not parent_ids:
            return []

        scoped_filters = self._combine_filters(
            metadata_filters,
            self._build_parent_scope_filter(parent_ids, route=route),
        )
        response = self._run_hybrid_query(
            collection=self.chunk_collection,
            query=query,
            route=route,
            filters=scoped_filters,
            limit=self._hybrid_limit(route=route),
            return_properties=[
                "chunk_id",
                "document_id",
                "section_id",
                "path",
                "title",
                "text",
                "domain",
                "department",
                "doc_type",
                "language",
                "acl_groups",
                "created_at",
                "page_start",
                "page_end",
                "source_locator",
            ],
        )

        return self._build_ranked_chunks_from_objects(
            objects=response.objects,
            query=query,
            intent=route.intent,
        )

    def _debug_print_stage(
        self,
        *,
        stage: str,
        query: str,
        chunks: Iterable[RetrievedChunk],
    ) -> None:
        if not settings.debug_retrieval:
            return

        chunk_list = list(chunks)
        print(f"\n[retrieval-debug] stage={stage} query={query}", flush=True)
        if not chunk_list:
            print("[retrieval-debug] no chunks", flush=True)
            return

        seen_paths: set[str] = set()
        printed_index = 0
        print("[retrieval-debug] files:", flush=True)
        for item in chunk_list:
            if item.path in seen_paths:
                continue
            seen_paths.add(item.path)
            printed_index += 1
            print(
                (
                    f"  {printed_index:02d}. score={item.score:.4f} "
                    f"dense={item.dense_score:.4f} bm25={item.bm25_score:.4f} "
                    f"path={item.path} title={item.title}"
                ),
                flush=True,
            )

        print("[retrieval-debug] chunks:", flush=True)
        for index, item in enumerate(chunk_list, start=1):
            reranker_value = (
                f"{item.reranker_score:.4f}" if item.reranker_score is not None else "-"
            )
            raw_bm25 = f"{item.raw_bm25:.4f}" if item.raw_bm25 is not None else "-"
            print(
                (
                    f"  {index:02d}. chunk_id={item.chunk_id} score={item.score:.4f} "
                    f"dense={item.dense_score:.4f} bm25={item.bm25_score:.4f} "
                    f"raw_bm25={raw_bm25} reranker={reranker_value} "
                    f"title_cov={item.title_coverage:.3f} path_cov={item.path_coverage:.3f} "
                    f"path={item.path}"
                ),
                flush=True,
            )

    def search(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalResult:
        route = self._plan_query_route(query, filters=filters)
        ranked = self.hybrid_search(query, route=route, filters=filters)
        if not ranked:
            self._debug_print_stage(stage="coarse_combined", query=query, chunks=[])
            return RetrievalResult(chunks=[], intent=route.intent)

        self._debug_print_stage(stage="coarse_combined", query=query, chunks=ranked)
        self._debug_print_stage(
            stage="before_reranker",
            query=query,
            chunks=ranked[: self.config.reranker_top_k],
        )
        ranked = self._apply_reranker(query, ranked, intent=route.intent)
        self._debug_print_stage(stage="after_reranker", query=query, chunks=ranked)
        final_chunks = self._limit_results(ranked, intent=route.intent)
        self._debug_print_stage(
            stage="final_selected", query=query, chunks=final_chunks
        )
        return RetrievalResult(
            chunks=final_chunks,
            intent=route.intent,
        )

    def _apply_reranker(
        self,
        query: str,
        ranked: list[RetrievedChunk],
        *,
        intent: str,
    ) -> list[RetrievedChunk]:
        if not self.reranker or not ranked:
            return ranked

        reranker_top_k = min(self.config.reranker_top_k, len(ranked))
        rerank_candidates = ranked[:reranker_top_k]
        reranker_scores_raw = self.reranker.score(
            query,
            [self._build_reranker_passage(item) for item in rerank_candidates],
        )
        normalized_reranker_scores = normalize_scores(
            {idx: score for idx, score in enumerate(reranker_scores_raw)}
        )

        updated_candidates: list[RetrievedChunk] = []
        reranker_weight = max(0.0, min(1.0, self.config.reranker_weight))
        heuristic_weight = 1.0 - reranker_weight

        for idx, item in enumerate(rerank_candidates):
            reranker_score = normalized_reranker_scores.get(idx, 0.0)
            final_score = (heuristic_weight * item.score) + (
                reranker_weight * reranker_score
            )

            updated_candidates.append(
                RetrievedChunk(
                    chunk_id=item.chunk_id,
                    path=item.path,
                    title=item.title,
                    text=item.text,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    source_locator=item.source_locator,
                    dense_score=item.dense_score,
                    bm25_score=item.bm25_score,
                    raw_bm25=item.raw_bm25,
                    reranker_score=reranker_score,
                    title_coverage=item.title_coverage,
                    path_coverage=item.path_coverage,
                    exact_reference_match=item.exact_reference_match,
                    score=final_score,
                )
            )

        combined = updated_candidates + ranked[reranker_top_k:]
        combined.sort(key=lambda item: item.score, reverse=True)
        return combined

    def _limit_results(
        self,
        ranked: list[RetrievedChunk],
        *,
        intent: str,
    ) -> list[RetrievedChunk]:
        per_document_limit = (
            self.config.doc_lookup_max_chunks_per_document
            if intent == "document_lookup"
            else self.config.max_chunks_per_document
        )
        final_top_k = (
            self.config.doc_lookup_final_top_k
            if intent == "document_lookup"
            else self.config.final_top_k
        )

        per_document_counter: Counter[str] = Counter()
        filtered: list[RetrievedChunk] = []

        for item in ranked:
            path = item.path or f"chunk:{item.chunk_id}"
            if per_document_counter[path] >= per_document_limit:
                continue

            filtered.append(item)
            per_document_counter[path] += 1

            if len(filtered) >= final_top_k:
                break

        return filtered

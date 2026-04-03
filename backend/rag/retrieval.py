import atexit
import re
import sqlite3
from collections import Counter
from pathlib import Path

from settings import settings
from rag.providers.embeddings.factory import create_embedding_provider
from rag.types import ChunkRecord, RetrievedChunk, RetrievalConfig, RetrievalResult
from rag.utils import normalize_scores, normalize_text, tokenize
from weaviate_store import create_weaviate_client, ensure_weaviate_collection


class RetrievalEngine:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.document_lookup_pattern = re.compile(
            config.document_lookup_pattern, re.IGNORECASE
        )
        ensure_weaviate_collection(recreate=False)
        self.client = create_weaviate_client()
        atexit.register(self.client.close)
        self.collection = self.client.collections.use(settings.weaviate_collection)
        self.dataset = self._load_dataset()
        self.embedding_provider = create_embedding_provider()
        self.fts_connection = self._build_fts_index()

    def _load_dataset(self) -> list[ChunkRecord]:
        raw_dataset = []
        for item in self.collection.iterator():
            raw_dataset.append(item.properties)

        if not raw_dataset:
            raise FileNotFoundError(
                f"Weaviate collection `{settings.weaviate_collection}` is empty. "
                "Run scripts/download_index.py first."
            )

        return [self._prepare_record(item) for item in raw_dataset]

    def _prepare_record(self, item: dict) -> ChunkRecord:
        path = str(item.get("path", "") or "")
        title = str(item.get("title", "") or Path(path).name or "chunk")
        text = normalize_text(str(item.get("text", "") or ""))
        chunk_id = int(item.get("chunk_id", 0) or 0)

        return {
            "chunk_id": chunk_id,
            "path": path,
            "title": title,
            "text": text,
            "title_tokens": set(tokenize(title)),
            "path_tokens": set(tokenize(path)),
        }

    def _build_fts_index(self) -> sqlite3.Connection:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute(
            """
            CREATE VIRTUAL TABLE fts_chunks
            USING fts5(chunk_id UNINDEXED, title, path, text)
            """
        )
        connection.executemany(
            """
            INSERT INTO fts_chunks (chunk_id, title, path, text)
            VALUES (?, ?, ?, ?)
            """,
            [
                (item["chunk_id"], item["title"], item["path"], item["text"])
                for item in self.dataset
            ],
        )
        connection.commit()
        return connection

    def detect_query_intent(self, query: str) -> str:
        if self.document_lookup_pattern.search(query):
            return "document_lookup"
        return "qa"

    def extract_reference_tokens(self, query: str) -> set[str]:
        return {
            token
            for token in tokenize(query)
            if token not in self.config.reference_stopwords
        }

    def _encode_query(self, query: str):
        return self.embedding_provider.encode([query])

    def dense_search(self, query: str) -> dict[int, float]:
        wvc = self._import_weaviate()
        query_embedding = self._encode_query(query)[0].tolist()
        response = self.collection.query.near_vector(
            near_vector=query_embedding,
            limit=self.config.dense_top_k,
            return_metadata=wvc.query.MetadataQuery(distance=True),
        )

        results: dict[int, float] = {}
        for obj in response.objects:
            properties = obj.properties or {}
            chunk_id = properties.get("chunk_id")
            if chunk_id is None:
                continue

            distance = getattr(obj.metadata, "distance", None)
            if distance is None:
                continue

            dense_score = max(0.0, 1.0 - float(distance))
            results[int(chunk_id)] = dense_score

        return results

    def _import_weaviate(self):
        try:
            import weaviate.classes as wvc
        except ImportError as exc:
            raise RuntimeError(
                "Пакет weaviate-client не установлен. "
                "Установи зависимости заново через `uv sync`."
            ) from exc

        return wvc

    def bm25_search(self, query: str) -> dict[int, dict[str, float]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return {}

        match_query = " OR ".join(f'"{token}"' for token in query_tokens)
        cursor = self.fts_connection.execute(
            """
            SELECT
                CAST(chunk_id AS INTEGER) AS chunk_id,
                bm25(fts_chunks, ?, ?, ?) AS bm25_score
            FROM fts_chunks
            WHERE fts_chunks MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (
                self.config.bm25_title_weight,
                self.config.bm25_path_weight,
                self.config.bm25_text_weight,
                match_query,
                self.config.bm25_top_k,
            ),
        )

        raw_scores = {
            int(row["chunk_id"]): float(row["bm25_score"]) for row in cursor.fetchall()
        }
        normalized_scores = normalize_scores(raw_scores, reverse=True)

        return {
            doc_id: {
                "raw_bm25": raw_scores[doc_id],
                "bm25_score": normalized_scores[doc_id],
            }
            for doc_id in raw_scores
        }

    def search(self, query: str) -> RetrievalResult:
        dense_results = self.dense_search(query)
        bm25_results = self.bm25_search(query)
        candidate_ids = set(dense_results) | set(bm25_results)

        if not candidate_ids:
            return RetrievalResult(chunks=[], intent="qa")

        intent = self.detect_query_intent(query)
        reference_tokens = self.extract_reference_tokens(query)
        ranked: list[RetrievedChunk] = []
        dataset_by_chunk_id = {item["chunk_id"]: item for item in self.dataset}

        for doc_id in candidate_ids:
            item = dataset_by_chunk_id.get(doc_id)
            if item is None:
                continue

            dense_score = dense_results.get(doc_id, 0.0)
            bm25_score = bm25_results.get(doc_id, {}).get("bm25_score", 0.0)
            raw_bm25 = bm25_results.get(doc_id, {}).get("raw_bm25")

            title_coverage = (
                len(reference_tokens.intersection(item["title_tokens"]))
                / len(reference_tokens)
                if reference_tokens
                else 0.0
            )
            path_coverage = (
                len(reference_tokens.intersection(item["path_tokens"]))
                / len(reference_tokens)
                if reference_tokens
                else 0.0
            )
            all_reference_tokens = item["title_tokens"] | item["path_tokens"]
            exact_reference_match = (
                1.0
                if reference_tokens and reference_tokens.issubset(all_reference_tokens)
                else 0.0
            )
            semantic_coverage = max(title_coverage, path_coverage)

            final_score = (
                (self.config.rerank_dense_weight * dense_score)
                + (self.config.rerank_bm25_weight * bm25_score)
                + (self.config.rerank_title_weight * title_coverage)
                + (self.config.rerank_path_weight * path_coverage)
                + (self.config.rerank_coverage_weight * semantic_coverage)
            )

            if intent == "document_lookup":
                final_score += self.config.doc_lookup_exact_boost * exact_reference_match
                final_score += self.config.doc_lookup_title_boost * title_coverage
                final_score += self.config.doc_lookup_path_boost * path_coverage

            ranked.append(
                RetrievedChunk(
                    chunk_id=doc_id,
                    path=item["path"],
                    title=item["title"],
                    text=item["text"],
                    dense_score=dense_score,
                    bm25_score=bm25_score,
                    raw_bm25=raw_bm25,
                    title_coverage=title_coverage,
                    path_coverage=path_coverage,
                    exact_reference_match=exact_reference_match,
                    score=final_score,
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return RetrievalResult(
            chunks=self._limit_results(ranked, intent=intent),
            intent=intent,
        )

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

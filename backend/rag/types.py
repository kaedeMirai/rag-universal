from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    path: str
    title: str
    text: str
    page_start: int | None
    page_end: int | None
    source_locator: str
    dense_score: float
    bm25_score: float
    raw_bm25: float | None
    reranker_score: float | None
    title_coverage: float
    path_coverage: float
    exact_reference_match: float
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    intent: str


@dataclass(frozen=True)
class RetrievalFilters:
    domains: tuple[str, ...] = ()
    departments: tuple[str, ...] = ()
    doc_types: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    acl_groups: tuple[str, ...] = ()
    created_from: datetime | None = None
    created_to: datetime | None = None


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[str]
    confidence: float


@dataclass(frozen=True)
class RetrievalConfig:
    hybrid_alpha: float
    hybrid_fusion: str
    dense_top_k: int
    bm25_top_k: int
    reranker_enabled: bool
    reranker_top_k: int
    reranker_weight: float
    final_top_k: int
    max_chunks_per_document: int
    doc_lookup_final_top_k: int
    doc_lookup_max_chunks_per_document: int
    bm25_title_weight: float
    bm25_path_weight: float
    bm25_text_weight: float
    rerank_dense_weight: float
    rerank_bm25_weight: float
    rerank_title_weight: float
    rerank_path_weight: float
    rerank_coverage_weight: float
    doc_lookup_exact_boost: float
    doc_lookup_title_boost: float
    doc_lookup_path_boost: float
    entity_lookup_pattern: str
    entity_query_max_tokens: int
    entity_exact_boost: float
    entity_title_boost: float
    entity_path_boost: float
    noisy_path_markers: set[str]
    noisy_path_penalty: float
    document_lookup_pattern: str
    lexical_stopwords: set[str]
    reference_stopwords: set[str]


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str
    max_context_tokens: int
    gpu_context_budgets: tuple[int, ...]
    cpu_context_budgets: tuple[int, ...]
    gpu_max_new_tokens: tuple[int, ...]
    cpu_max_new_tokens: tuple[int, ...]
    temperature: float
    top_p: float
    do_sample: bool
    system_prompt: str
    document_lookup_prompt_suffix: str


@dataclass(frozen=True)
class RAGProfile:
    name: str
    description: str
    retrieval: RetrievalConfig
    generation: GenerationConfig


ChunkRecord = dict[str, Any]

from dataclasses import replace

from settings import settings

from rag.types import GenerationConfig, RAGProfile, RetrievalConfig


def _base_retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        final_top_k=settings.final_top_k,
        max_chunks_per_document=settings.max_chunks_per_document,
        doc_lookup_final_top_k=settings.doc_lookup_final_top_k,
        doc_lookup_max_chunks_per_document=settings.doc_lookup_max_chunks_per_document,
        bm25_title_weight=settings.bm25_title_weight,
        bm25_path_weight=settings.bm25_path_weight,
        bm25_text_weight=settings.bm25_text_weight,
        rerank_dense_weight=settings.rerank_dense_weight,
        rerank_bm25_weight=settings.rerank_bm25_weight,
        rerank_title_weight=settings.rerank_title_weight,
        rerank_path_weight=settings.rerank_path_weight,
        rerank_coverage_weight=settings.rerank_coverage_weight,
        doc_lookup_exact_boost=settings.doc_lookup_exact_boost,
        doc_lookup_title_boost=settings.doc_lookup_title_boost,
        doc_lookup_path_boost=settings.doc_lookup_path_boost,
        document_lookup_pattern=settings.document_lookup_pattern,
        reference_stopwords=settings.reference_stopwords,
    )


def _base_generation_config() -> GenerationConfig:
    return GenerationConfig(
        model_name=settings.generation_model_name,
        max_context_tokens=settings.max_context_tokens,
        gpu_context_budgets=settings.gpu_context_budgets,
        cpu_context_budgets=settings.cpu_context_budgets,
        gpu_max_new_tokens=settings.gpu_max_new_tokens,
        cpu_max_new_tokens=settings.cpu_max_new_tokens,
        temperature=settings.generation_temperature,
        top_p=settings.generation_top_p,
        do_sample=settings.generation_do_sample,
        system_prompt=settings.system_prompt,
        document_lookup_prompt_suffix=settings.document_lookup_prompt_suffix,
    )


def _balanced_profile() -> RAGProfile:
    return RAGProfile(
        name="balanced",
        description="Сбалансированный режим для обычных запросов.",
        retrieval=_base_retrieval_config(),
        generation=_base_generation_config(),
    )


def _fast_profile() -> RAGProfile:
    return RAGProfile(
        name="fast",
        description="Быстрый режим для коротких ответов и дешевых прогонов.",
        retrieval=replace(
            _base_retrieval_config(),
            dense_top_k=min(settings.dense_top_k, 12),
            bm25_top_k=min(settings.bm25_top_k, 12),
            final_top_k=min(settings.final_top_k, 4),
            max_chunks_per_document=1,
            doc_lookup_final_top_k=min(settings.doc_lookup_final_top_k, 3),
            doc_lookup_max_chunks_per_document=1,
        ),
        generation=replace(
            _base_generation_config(),
            max_context_tokens=min(settings.max_context_tokens, 512),
            gpu_context_budgets=tuple(min(value, 512) for value in settings.gpu_context_budgets),
            cpu_context_budgets=tuple(min(value, 700) for value in settings.cpu_context_budgets),
            gpu_max_new_tokens=tuple(min(value, 128) for value in settings.gpu_max_new_tokens),
            cpu_max_new_tokens=tuple(min(value, 128) for value in settings.cpu_max_new_tokens),
            do_sample=False,
            temperature=min(settings.generation_temperature, 0.1),
        ),
    )


def _deep_profile() -> RAGProfile:
    return RAGProfile(
        name="deep",
        description="Режим для более тщательного retrieval и длинного контекста.",
        retrieval=replace(
            _base_retrieval_config(),
            dense_top_k=max(settings.dense_top_k, 40),
            bm25_top_k=max(settings.bm25_top_k, 40),
            final_top_k=max(settings.final_top_k, 8),
            max_chunks_per_document=max(settings.max_chunks_per_document, 3),
            doc_lookup_final_top_k=max(settings.doc_lookup_final_top_k, 6),
            doc_lookup_max_chunks_per_document=max(settings.doc_lookup_max_chunks_per_document, 4),
        ),
        generation=replace(
            _base_generation_config(),
            max_context_tokens=max(settings.max_context_tokens, 1600),
            gpu_context_budgets=tuple(max(value, 800) for value in settings.gpu_context_budgets),
            cpu_context_budgets=tuple(max(value, 1200) for value in settings.cpu_context_budgets),
            gpu_max_new_tokens=tuple(max(value, 192) for value in settings.gpu_max_new_tokens),
            cpu_max_new_tokens=tuple(max(value, 192) for value in settings.cpu_max_new_tokens),
        ),
    )


PROFILES: dict[str, RAGProfile] = {
    "balanced": _balanced_profile(),
    "fast": _fast_profile(),
    "deep": _deep_profile(),
}


def get_active_profile() -> RAGProfile:
    return PROFILES.get(settings.rag_profile, PROFILES["balanced"])

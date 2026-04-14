import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(ROOT_DIR / ".env")


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _get_optional_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _get_int(name: str, default: int) -> int:
    return int(_get_str(name, str(default)))


def _get_float(name: str, default: float) -> float:
    return float(_get_str(name, str(default)))


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = _get_str(name, ",".join(str(value) for value in default))
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def _get_str_set(name: str, default: set[str]) -> set[str]:
    raw = os.getenv(name)
    if raw is None:
        return set(default)
    return {value.strip() for value in raw.split(",") if value.strip()}


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    scripts_dir: Path

    database_url: str
    session_ttl_days: int

    bootstrap_admin_username: str | None
    bootstrap_admin_password: str | None
    bootstrap_admin_first_name: str
    bootstrap_admin_last_name: str
    bootstrap_admin_email: str | None

    backend_url: str

    chat_request_timeout_seconds: int

    embedding_provider: str
    generation_provider: str
    embedding_model_name: str
    generation_model_name: str
    generation_dtype: str
    generation_device_map: str
    eval_dataset_path: Path | None

    vector_index_path: Path
    metadata_path: Path
    source_csv_path: Path
    weaviate_http_host: str
    weaviate_http_port: int
    weaviate_http_secure: bool
    weaviate_grpc_host: str
    weaviate_grpc_port: int
    weaviate_grpc_secure: bool
    weaviate_api_key: str | None
    weaviate_collection: str
    weaviate_skip_init_checks: bool

    dense_top_k: int
    bm25_top_k: int
    final_top_k: int
    reranker_enabled: bool
    reranker_top_k: int
    reranker_weight: float
    max_context_tokens: int
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

    gpu_context_budgets: tuple[int, ...]
    cpu_context_budgets: tuple[int, ...]
    gpu_max_new_tokens: tuple[int, ...]
    cpu_max_new_tokens: tuple[int, ...]

    generation_temperature: float
    generation_top_p: float
    generation_do_sample: bool
    rag_profile: str
    preload_models_on_startup: bool

    chunk_tokens: int
    chunk_overlap: int
    embedding_batch_size: int
    embedding_device: str
    reranker_provider: str
    reranker_model_name: str
    reranker_device: str
    reranker_batch_size: int
    reranker_max_length: int

    document_lookup_pattern: str
    reference_stopwords: set[str]
    system_prompt: str
    document_lookup_prompt_suffix: str


def load_settings() -> Settings:
    scripts_dir = ROOT_DIR / "scripts"

    return Settings(
        root_dir=ROOT_DIR,
        scripts_dir=scripts_dir,
        database_url=_get_str(
            "RAG_DATABASE_URL",
            os.getenv(
                "DATABASE_URL", "postgresql://admin:admin@127.0.0.1:5432/rag_system"
            ),
        ),
        session_ttl_days=_get_int("RAG_SESSION_TTL_DAYS", 7),
        bootstrap_admin_username=_get_optional_str("RAG_BOOTSTRAP_ADMIN_USERNAME")
        or "admin",
        bootstrap_admin_password=_get_optional_str("RAG_BOOTSTRAP_ADMIN_PASSWORD")
        or "admin",
        bootstrap_admin_first_name=_get_str("RAG_BOOTSTRAP_ADMIN_FIRST_NAME", "Admin"),
        bootstrap_admin_last_name=_get_str("RAG_BOOTSTRAP_ADMIN_LAST_NAME", "User"),
        bootstrap_admin_email=_get_optional_str("RAG_BOOTSTRAP_ADMIN_EMAIL"),
        backend_url=_get_str("RAG_BACKEND_URL", "http://127.0.0.1:8006"),
        chat_request_timeout_seconds=_get_int("RAG_CHAT_REQUEST_TIMEOUT_SECONDS", 300),
        embedding_provider=_get_str("RAG_EMBEDDING_PROVIDER", "huggingface"),
        generation_provider=_get_str("RAG_GENERATION_PROVIDER", "huggingface"),
        embedding_model_name=_get_str("RAG_EMBEDDING_MODEL", "BAAI/bge-m3"),
        generation_model_name=_get_str(
            "RAG_GENERATION_MODEL", "Qwen/Qwen2.5-7B-Instruct"
        ),
        generation_dtype=_get_str("RAG_GENERATION_DTYPE", "auto"),
        generation_device_map=_get_str("RAG_GENERATION_DEVICE_MAP", "auto"),
        eval_dataset_path=(
            ROOT_DIR / _get_str("RAG_EVAL_DATASET_PATH", "")
            if _get_optional_str("RAG_EVAL_DATASET_PATH")
            else None
        ),
        vector_index_path=ROOT_DIR
        / _get_str("RAG_VECTOR_INDEX_PATH", "scripts/vector.index"),
        metadata_path=ROOT_DIR
        / _get_str("RAG_METADATA_PATH", "scripts/meta_data.json"),
        source_csv_path=ROOT_DIR / _get_str("RAG_SOURCE_CSV_PATH", "scripts/temp.csv"),
        weaviate_http_host=_get_str("RAG_WEAVIATE_HTTP_HOST", "127.0.0.1"),
        weaviate_http_port=_get_int("RAG_WEAVIATE_HTTP_PORT", 8080),
        weaviate_http_secure=_get_bool("RAG_WEAVIATE_HTTP_SECURE", False),
        weaviate_grpc_host=_get_str("RAG_WEAVIATE_GRPC_HOST", "127.0.0.1"),
        weaviate_grpc_port=_get_int("RAG_WEAVIATE_GRPC_PORT", 50051),
        weaviate_grpc_secure=_get_bool("RAG_WEAVIATE_GRPC_SECURE", False),
        weaviate_api_key=_get_optional_str("RAG_WEAVIATE_API_KEY"),
        weaviate_collection=_get_str("RAG_WEAVIATE_COLLECTION", "DocumentChunk"),
        weaviate_skip_init_checks=_get_bool("RAG_WEAVIATE_SKIP_INIT_CHECKS", False),
        dense_top_k=_get_int("RAG_DENSE_TOP_K", 24),
        bm25_top_k=_get_int("RAG_BM25_TOP_K", 24),
        final_top_k=_get_int("RAG_FINAL_TOP_K", 6),
        reranker_enabled=_get_bool("RAG_RERANKER_ENABLED", True),
        reranker_top_k=_get_int("RAG_RERANKER_TOP_K", 24),
        reranker_weight=_get_float("RAG_RERANKER_WEIGHT", 0.65),
        max_context_tokens=_get_int("RAG_MAX_CONTEXT_TOKENS", 1200),
        max_chunks_per_document=_get_int("RAG_MAX_CHUNKS_PER_DOCUMENT", 2),
        doc_lookup_final_top_k=_get_int("RAG_DOC_LOOKUP_FINAL_TOP_K", 4),
        doc_lookup_max_chunks_per_document=_get_int(
            "RAG_DOC_LOOKUP_MAX_CHUNKS_PER_DOCUMENT", 3
        ),
        bm25_title_weight=_get_float("RAG_BM25_TITLE_WEIGHT", 8.0),
        bm25_path_weight=_get_float("RAG_BM25_PATH_WEIGHT", 3.0),
        bm25_text_weight=_get_float("RAG_BM25_TEXT_WEIGHT", 1.0),
        rerank_dense_weight=_get_float("RAG_RERANK_DENSE_WEIGHT", 0.45),
        rerank_bm25_weight=_get_float("RAG_RERANK_BM25_WEIGHT", 0.30),
        rerank_title_weight=_get_float("RAG_RERANK_TITLE_WEIGHT", 0.15),
        rerank_path_weight=_get_float("RAG_RERANK_PATH_WEIGHT", 0.05),
        rerank_coverage_weight=_get_float("RAG_RERANK_COVERAGE_WEIGHT", 0.05),
        doc_lookup_exact_boost=_get_float("RAG_DOC_LOOKUP_EXACT_BOOST", 0.25),
        doc_lookup_title_boost=_get_float("RAG_DOC_LOOKUP_TITLE_BOOST", 0.18),
        doc_lookup_path_boost=_get_float("RAG_DOC_LOOKUP_PATH_BOOST", 0.10),
        gpu_context_budgets=_get_int_tuple(
            "RAG_GPU_CONTEXT_BUDGETS", (1200, 800, 512, 320)
        ),
        cpu_context_budgets=_get_int_tuple("RAG_CPU_CONTEXT_BUDGETS", (1200, 900)),
        gpu_max_new_tokens=_get_int_tuple("RAG_GPU_MAX_NEW_TOKENS", (250, 160, 96, 64)),
        cpu_max_new_tokens=_get_int_tuple("RAG_CPU_MAX_NEW_TOKENS", (220, 128)),
        generation_temperature=_get_float("RAG_GENERATION_TEMPERATURE", 0.2),
        generation_top_p=_get_float("RAG_GENERATION_TOP_P", 0.9),
        generation_do_sample=_get_bool("RAG_GENERATION_DO_SAMPLE", True),
        rag_profile=_get_str("RAG_PROFILE", "balanced"),
        preload_models_on_startup=_get_bool("RAG_PRELOAD_MODELS_ON_STARTUP", False),
        chunk_tokens=_get_int("RAG_CHUNK_TOKENS", 512),
        chunk_overlap=_get_int("RAG_CHUNK_OVERLAP", 100),
        embedding_batch_size=_get_int("RAG_EMBEDDING_BATCH_SIZE", 8),
        embedding_device=_get_str("RAG_EMBEDDING_DEVICE", "cuda"),
        reranker_provider=_get_str("RAG_RERANKER_PROVIDER", "huggingface"),
        reranker_model_name=_get_str(
            "RAG_RERANKER_MODEL",
            "BAAI/bge-reranker-v2-m3",
        ),
        reranker_device=_get_str("RAG_RERANKER_DEVICE", "cuda"),
        reranker_batch_size=_get_int("RAG_RERANKER_BATCH_SIZE", 8),
        reranker_max_length=_get_int("RAG_RERANKER_MAX_LENGTH", 512),
        document_lookup_pattern=_get_str(
            "RAG_DOCUMENT_LOOKUP_PATTERN",
            r"(№|no|служебн\w*\s+задан\w*|приказ|положение|регламент|инструкц\w*|документ)",
        ),
        reference_stopwords=_get_str_set(
            "RAG_REFERENCE_STOPWORDS",
            {
                "мне",
                "бы",
                "почитать",
                "покажи",
                "показать",
                "найди",
                "нужен",
                "нужно",
                "документ",
                "документа",
                "документе",
                "положение",
                "приказ",
                "регламент",
                "инструкция",
                "из",
                "про",
                "по",
                "могу",
                "я",
                "хочу",
            },
        ),
        system_prompt=_get_str(
            "RAG_SYSTEM_PROMPT",
            "Ты корпоративный ассистент. Отвечай только по предоставленному контексту. "
            "Если контекста недостаточно, прямо скажи об этом. Не выдумывай детали.",
        ),
        document_lookup_prompt_suffix=_get_str(
            "RAG_DOCUMENT_LOOKUP_PROMPT_SUFFIX",
            "Если пользователь ищет конкретный документ, сначала назови наиболее вероятный "
            "документ и кратко перескажи, что удалось извлечь из найденных фрагментов.",
        ),
    )


settings = load_settings()

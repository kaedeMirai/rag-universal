from __future__ import annotations

from settings import settings


def _import_weaviate():
    try:
        import weaviate
        import weaviate.classes as wvc
        from weaviate.auth import Auth
    except ImportError as exc:
        raise RuntimeError(
            "Пакет weaviate-client не установлен. "
            "Установи зависимости заново через `uv sync`."
        ) from exc

    return weaviate, wvc, Auth


def create_weaviate_client():
    weaviate, _, Auth = _import_weaviate()

    auth_credentials = (
        Auth.api_key(settings.weaviate_api_key) if settings.weaviate_api_key else None
    )

    return weaviate.connect_to_custom(
        http_host=settings.weaviate_http_host,
        http_port=settings.weaviate_http_port,
        http_secure=settings.weaviate_http_secure,
        grpc_host=settings.weaviate_grpc_host,
        grpc_port=settings.weaviate_grpc_port,
        grpc_secure=settings.weaviate_grpc_secure,
        auth_credentials=auth_credentials,
        skip_init_checks=settings.weaviate_skip_init_checks,
    )


def ensure_weaviate_collection(*, recreate: bool = False):
    _, wvc, _ = _import_weaviate()
    client = create_weaviate_client()
    try:
        collection_name = settings.weaviate_collection
        if recreate and client.collections.exists(collection_name):
            client.collections.delete(collection_name)

        if client.collections.exists(collection_name):
            return

        client.collections.create(
            name=collection_name,
            vector_config=wvc.config.Configure.Vectors.self_provided(
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE
                )
            ),
            properties=[
                wvc.config.Property(
                    name="chunk_id",
                    data_type=wvc.config.DataType.INT,
                ),
                wvc.config.Property(
                    name="chunk_index",
                    data_type=wvc.config.DataType.INT,
                ),
                wvc.config.Property(
                    name="source_row",
                    data_type=wvc.config.DataType.INT,
                ),
                wvc.config.Property(
                    name="path",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="title",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="extension",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="page_start",
                    data_type=wvc.config.DataType.INT,
                ),
                wvc.config.Property(
                    name="page_end",
                    data_type=wvc.config.DataType.INT,
                ),
                wvc.config.Property(
                    name="source_locator",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="text",
                    data_type=wvc.config.DataType.TEXT,
                ),
            ],
        )
    finally:
        client.close()

from __future__ import annotations

from settings import settings


def get_chunk_collection_name() -> str:
    return settings.weaviate_collection


def get_section_collection_name() -> str:
    return f"{settings.weaviate_collection}Section"


def get_document_collection_name() -> str:
    return f"{settings.weaviate_collection}Document"


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


def _common_metadata_properties(wvc):
    return [
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
            name="domain",
            data_type=wvc.config.DataType.TEXT,
        ),
        wvc.config.Property(
            name="department",
            data_type=wvc.config.DataType.TEXT,
        ),
        wvc.config.Property(
            name="doc_type",
            data_type=wvc.config.DataType.TEXT,
        ),
        wvc.config.Property(
            name="language",
            data_type=wvc.config.DataType.TEXT,
        ),
        wvc.config.Property(
            name="acl_groups",
            data_type=wvc.config.DataType.TEXT_ARRAY,
        ),
        wvc.config.Property(
            name="created_at",
            data_type=wvc.config.DataType.DATE,
        ),
    ]


def _create_collection_if_missing(client, wvc, *, name: str, properties: list):
    if client.collections.exists(name):
        return

    client.collections.create(
        name=name,
        vector_config=wvc.config.Configure.Vectors.self_provided(
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE
            )
        ),
        properties=properties,
    )


def ensure_weaviate_collection(*, recreate: bool = False):
    _, wvc, _ = _import_weaviate()
    client = create_weaviate_client()
    try:
        document_collection = get_document_collection_name()
        section_collection = get_section_collection_name()
        chunk_collection = get_chunk_collection_name()

        if recreate:
            for collection_name in (
                chunk_collection,
                section_collection,
                document_collection,
            ):
                if client.collections.exists(collection_name):
                    client.collections.delete(collection_name)

        _create_collection_if_missing(
            client,
            wvc,
            name=document_collection,
            properties=[
                wvc.config.Property(
                    name="document_id",
                    data_type=wvc.config.DataType.INT,
                ),
                wvc.config.Property(
                    name="source_row",
                    data_type=wvc.config.DataType.INT,
                ),
                *_common_metadata_properties(wvc),
                wvc.config.Property(
                    name="text",
                    data_type=wvc.config.DataType.TEXT,
                ),
            ],
        )

        _create_collection_if_missing(
            client,
            wvc,
            name=section_collection,
            properties=[
                wvc.config.Property(
                    name="section_id",
                    data_type=wvc.config.DataType.INT,
                ),
                wvc.config.Property(
                    name="document_id",
                    data_type=wvc.config.DataType.INT,
                ),
                wvc.config.Property(
                    name="section_index",
                    data_type=wvc.config.DataType.INT,
                ),
                wvc.config.Property(
                    name="source_row",
                    data_type=wvc.config.DataType.INT,
                ),
                *_common_metadata_properties(wvc),
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

        _create_collection_if_missing(
            client,
            wvc,
            name=chunk_collection,
            properties=[
                wvc.config.Property(
                    name="chunk_id",
                    data_type=wvc.config.DataType.INT,
                ),
                wvc.config.Property(
                    name="document_id",
                    data_type=wvc.config.DataType.INT,
                ),
                wvc.config.Property(
                    name="section_id",
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
                *_common_metadata_properties(wvc),
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

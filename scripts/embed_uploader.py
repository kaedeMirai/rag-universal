from __future__ import annotations

from datetime import datetime
from uuid import NAMESPACE_URL, uuid5

from sentence_transformers import SentenceTransformer

try:
    import torch
except Exception:
    torch = None

from settings import settings
from weaviate_store import (
    create_weaviate_client,
    get_chunk_collection_name,
    get_document_collection_name,
    get_section_collection_name,
)


class WeaviateBatchUploader:
    def __init__(self, *, batch_size: int):
        self.batch_size = batch_size
        self.actual_device = self._resolve_device()
        self.encoder = SentenceTransformer(
            settings.embedding_model_name,
            device=self.actual_device,
        )
        self.encoder.max_seq_length = settings.chunk_tokens
        self.client = create_weaviate_client()
        self.document_collection = self.client.collections.use(
            get_document_collection_name()
        )
        self.section_collection = self.client.collections.use(
            get_section_collection_name()
        )
        self.chunk_collection = self.client.collections.use(get_chunk_collection_name())
        self.pending_document_records: list[dict[str, object]] = []
        self.pending_section_records: list[dict[str, object]] = []
        self.pending_chunk_records: list[dict[str, object]] = []
        self.uploaded_chunks = 0

    @staticmethod
    def _normalize_acl_groups(value: object) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @staticmethod
    def _normalize_created_at(value: object) -> str | None:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return None

    def _resolve_device(self) -> str:
        requested_device = settings.embedding_device
        if requested_device.startswith("cuda"):
            if torch is None or not torch.cuda.is_available():
                return "cpu"
        return requested_device

    def add_document_record(self, record: dict[str, object]) -> None:
        self.pending_document_records.append(record)
        if len(self.pending_document_records) >= self.batch_size:
            self._flush_documents()

    def add_section_record(self, record: dict[str, object]) -> None:
        self.pending_section_records.append(record)
        if len(self.pending_section_records) >= self.batch_size:
            self._flush_sections()

    def add_chunk_record(self, record: dict[str, object]) -> None:
        self.pending_chunk_records.append(record)
        if len(self.pending_chunk_records) >= self.batch_size:
            self._flush_chunks()

    def _encode_texts(self, records: list[dict[str, object]]):
        batch_texts = [str(item["text"]) for item in records]
        return self.encoder.encode(
            batch_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

    def _base_properties(self, record: dict[str, object]) -> dict[str, object]:
        return {
            "source_row": int(record["source_row"]),
            "path": str(record["path"]),
            "title": str(record["title"]),
            "extension": str(record["extension"]),
            "domain": str(record.get("domain", "") or ""),
            "department": str(record.get("department", "") or ""),
            "doc_type": str(record.get("doc_type", "") or ""),
            "language": str(record.get("language", "") or ""),
            "acl_groups": self._normalize_acl_groups(record.get("acl_groups")),
            "created_at": self._normalize_created_at(record.get("created_at")),
        }

    def _flush_documents(self) -> None:
        if not self.pending_document_records:
            return

        batch_records = self.pending_document_records
        self.pending_document_records = []
        embeddings = self._encode_texts(batch_records)

        with self.document_collection.batch.fixed_size(batch_size=self.batch_size) as batch:
            for record, vector in zip(batch_records, embeddings):
                object_uuid = uuid5(
                    NAMESPACE_URL,
                    f"document::{record['path']}::{record['document_id']}",
                )
                batch.add_object(
                    properties={
                        "document_id": int(record["document_id"]),
                        **self._base_properties(record),
                        "text": str(record["text"]),
                    },
                    uuid=object_uuid,
                    vector=vector.tolist(),
                )

            if batch.number_errors > 0:
                failed_objects = getattr(batch, "failed_objects", None)
                if failed_objects is None:
                    failed_objects = getattr(
                        self.document_collection.batch, "failed_objects", []
                    )
                raise RuntimeError(
                    f"Weaviate document batch import failed for {len(failed_objects)} objects. "
                    f"First error: {failed_objects[0] if failed_objects else 'unknown'}"
                )

    def _flush_sections(self) -> None:
        if not self.pending_section_records:
            return

        batch_records = self.pending_section_records
        self.pending_section_records = []
        embeddings = self._encode_texts(batch_records)

        with self.section_collection.batch.fixed_size(batch_size=self.batch_size) as batch:
            for record, vector in zip(batch_records, embeddings):
                object_uuid = uuid5(
                    NAMESPACE_URL,
                    f"section::{record['path']}::{record['document_id']}::{record['section_id']}",
                )
                batch.add_object(
                    properties={
                        "section_id": int(record["section_id"]),
                        "document_id": int(record["document_id"]),
                        "section_index": int(record["section_index"]),
                        **self._base_properties(record),
                        "page_start": int(record["page_start"]),
                        "page_end": int(record["page_end"]),
                        "source_locator": str(record["source_locator"]),
                        "text": str(record["text"]),
                    },
                    uuid=object_uuid,
                    vector=vector.tolist(),
                )

            if batch.number_errors > 0:
                failed_objects = getattr(batch, "failed_objects", None)
                if failed_objects is None:
                    failed_objects = getattr(
                        self.section_collection.batch, "failed_objects", []
                    )
                raise RuntimeError(
                    f"Weaviate section batch import failed for {len(failed_objects)} objects. "
                    f"First error: {failed_objects[0] if failed_objects else 'unknown'}"
                )

    def _flush_chunks(self) -> None:
        if not self.pending_chunk_records:
            return

        batch_records = self.pending_chunk_records
        self.pending_chunk_records = []
        embeddings = self._encode_texts(batch_records)

        with self.chunk_collection.batch.fixed_size(batch_size=self.batch_size) as batch:
            for record, vector in zip(batch_records, embeddings):
                object_uuid = uuid5(
                    NAMESPACE_URL,
                    f"chunk::{record['path']}::{record['section_id']}::{record['chunk_index']}::{record['chunk_id']}",
                )
                batch.add_object(
                    properties={
                        "chunk_id": int(record["chunk_id"]),
                        "document_id": int(record["document_id"]),
                        "section_id": int(record["section_id"]),
                        "chunk_index": int(record["chunk_index"]),
                        **self._base_properties(record),
                        "page_start": int(record["page_start"]),
                        "page_end": int(record["page_end"]),
                        "source_locator": str(record["source_locator"]),
                        "text": str(record["text"]),
                    },
                    uuid=object_uuid,
                    vector=vector.tolist(),
                )

            if batch.number_errors > 0:
                failed_objects = getattr(batch, "failed_objects", None)
                if failed_objects is None:
                    failed_objects = getattr(
                        self.chunk_collection.batch, "failed_objects", []
                    )
                raise RuntimeError(
                    f"Weaviate chunk batch import failed for {len(failed_objects)} objects. "
                    f"First error: {failed_objects[0] if failed_objects else 'unknown'}"
                )

        self.uploaded_chunks += len(batch_records)

    def flush(self) -> None:
        self._flush_documents()
        self._flush_sections()
        self._flush_chunks()

    def close(self) -> None:
        self.flush()
        self.client.close()

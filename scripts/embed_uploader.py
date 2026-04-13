from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from sentence_transformers import SentenceTransformer

try:
    import torch
except Exception:
    torch = None

from settings import settings
from weaviate_store import create_weaviate_client


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
        self.collection = self.client.collections.use(settings.weaviate_collection)
        self.pending_records: list[dict[str, str | int]] = []
        self.uploaded_chunks = 0

    def _resolve_device(self) -> str:
        requested_device = settings.embedding_device
        if requested_device.startswith("cuda"):
            if torch is None or not torch.cuda.is_available():
                return "cpu"
        return requested_device

    def add_record(self, record: dict[str, str | int]) -> None:
        self.pending_records.append(record)
        if len(self.pending_records) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending_records:
            return

        batch_records = self.pending_records
        self.pending_records = []
        batch_texts = [str(item["text"]) for item in batch_records]
        embeddings = self.encoder.encode(
            batch_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        with self.collection.batch.fixed_size(batch_size=self.batch_size) as batch:
            for record, vector in zip(batch_records, embeddings):
                object_uuid = uuid5(
                    NAMESPACE_URL,
                    f"{record['path']}::{record['chunk_index']}::{record['chunk_id']}",
                )
                batch.add_object(
                    properties={
                        "chunk_id": int(record["chunk_id"]),
                        "chunk_index": int(record["chunk_index"]),
                        "source_row": int(record["source_row"]),
                        "path": str(record["path"]),
                        "title": str(record["title"]),
                        "extension": str(record["extension"]),
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
                    failed_objects = getattr(self.collection.batch, "failed_objects", [])
                raise RuntimeError(
                    f"Weaviate batch import failed for {len(failed_objects)} objects. "
                    f"First error: {failed_objects[0] if failed_objects else 'unknown'}"
                )

        self.uploaded_chunks += len(batch_records)

    def close(self) -> None:
        self.flush()
        self.client.close()

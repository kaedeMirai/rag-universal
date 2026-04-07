import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import torch
except Exception:
    torch = None

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from settings import settings
from weaviate_store import create_weaviate_client, ensure_weaviate_collection


logger = logging.getLogger("download_index")
CONTROL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
WHITESPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
DEFAULT_MAX_BLOCK_CHARS = max(settings.chunk_tokens * 8, 4096)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def configure_csv_field_limit() -> None:
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            return
        except OverflowError:
            max_int = max_int // 10


def humanize_bytes(size_bytes: int | float) -> str:
    if size_bytes <= 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    unit_index = 0

    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1

    return f"{value:.2f} {units[unit_index]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Потоковая индексация CSV в Weaviate без удержания всего датасета в памяти."
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Удалить и создать коллекцию заново перед импортом.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.embedding_batch_size,
        help="Размер батча для эмбеддингов и записи в Weaviate.",
    )
    parser.add_argument(
        "--progress-every-documents",
        type=int,
        default=100,
        help="Как часто печатать прогресс по документам.",
    )
    parser.add_argument(
        "--max-block-chars",
        type=int,
        default=DEFAULT_MAX_BLOCK_CHARS,
        help="Максимальный размер текстового блока до принудительного разрезания.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = CONTROL_CHARACTERS_RE.sub("", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines())
    cleaned = MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def resolve_text_column(fieldnames: list[str]) -> str:
    for candidate in ("clean_text", "content", "text"):
        if candidate in fieldnames:
            return candidate

    raise ValueError(
        f"CSV {settings.source_csv_path} не содержит текстовой колонки. "
        "Ожидалась одна из: clean_text, content, text."
    )


def split_long_piece(piece: str, *, max_chars: int) -> list[str]:
    if len(piece) <= max_chars:
        return [piece]

    chunks: list[str] = []
    start = 0
    piece_length = len(piece)
    while start < piece_length:
        end = min(start + max_chars, piece_length)
        chunks.append(piece[start:end])
        start = end
    return chunks


def iter_text_blocks(text: str, *, max_block_chars: int) -> list[str]:
    if not text:
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_blocks = [block.strip() for block in normalized.split("\n\n") if block.strip()]
    if not raw_blocks:
        raw_blocks = [normalized.strip()]

    blocks: list[str] = []
    for block in raw_blocks:
        if len(block) <= max_block_chars:
            blocks.append(block)
            continue

        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) > 1:
            for line in lines:
                blocks.extend(split_long_piece(line, max_chars=max_block_chars))
        else:
            blocks.extend(split_long_piece(block, max_chars=max_block_chars))

    return [block for block in blocks if block]


class StreamingChunker:
    def __init__(self, tokenizer, *, chunk_tokens: int, chunk_overlap: int):
        self.tokenizer = tokenizer
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap
        self.step_tokens = max(chunk_tokens - chunk_overlap, 1)

    def chunk_text(self, text: str, *, max_block_chars: int) -> list[str]:
        token_buffer: list[int] = []
        output_chunks: list[str] = []

        for block in iter_text_blocks(text, max_block_chars=max_block_chars):
            block_tokens = self.tokenizer.encode(block, add_special_tokens=False)
            if not block_tokens:
                continue

            token_buffer.extend(block_tokens)

            while len(token_buffer) >= self.chunk_tokens:
                chunk_tokens = token_buffer[: self.chunk_tokens]
                decoded = self.tokenizer.decode(chunk_tokens).strip()
                if decoded:
                    output_chunks.append(decoded)
                token_buffer = token_buffer[self.step_tokens :]

        if token_buffer:
            decoded_tail = self.tokenizer.decode(token_buffer).strip()
            if decoded_tail:
                output_chunks.append(decoded_tail)

        return output_chunks


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
        self.uploaded_text_bytes = 0

    def _resolve_device(self) -> str:
        requested_device = settings.embedding_device
        if requested_device.startswith("cuda"):
            if torch is None:
                logger.warning(
                    "Запрошен CUDA, но torch недоступен. Использую CPU для эмбеддингов."
                )
                return "cpu"
            if not torch.cuda.is_available():
                logger.warning(
                    "Запрошен CUDA, но torch.cuda.is_available() = False. "
                    "Использую CPU для эмбеддингов."
                )
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
        self.uploaded_text_bytes += sum(
            len(str(item["text"]).encode("utf-8", errors="ignore"))
            for item in batch_records
        )

    def close(self) -> None:
        self.flush()
        self.client.close()


def stream_index_csv(args: argparse.Namespace) -> None:
    source_path = settings.source_csv_path
    if not source_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_path}")

    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model_name)
    chunker = StreamingChunker(
        tokenizer,
        chunk_tokens=settings.chunk_tokens,
        chunk_overlap=settings.chunk_overlap,
    )
    uploader = WeaviateBatchUploader(batch_size=args.batch_size)

    csv_size_bytes = source_path.stat().st_size
    logger.info(f"Размер CSV: {humanize_bytes(csv_size_bytes)}")

    processed_documents = 0
    skipped_documents = 0
    total_chunks = 0
    total_text_bytes = 0
    max_chunks_in_document = 0
    next_chunk_id = 0

    with source_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError(f"CSV {source_path} не содержит заголовок.")

        text_column = resolve_text_column(reader.fieldnames)
        progress_bar = tqdm(desc="Indexing documents", unit="doc")

        try:
            for row_index, row in enumerate(reader):
                path = str(row.get("file_path", "") or "")
                title = str(row.get("file_name", "") or Path(path).name)
                extension = str(row.get("extension", "") or Path(path).suffix.lower())
                text = clean_text(str(row.get(text_column, "") or ""))
                progress_bar.update(1)

                if not text:
                    skipped_documents += 1
                    continue

                processed_documents += 1
                text_bytes = len(text.encode("utf-8", errors="ignore"))
                total_text_bytes += text_bytes
                document_chunks = chunker.chunk_text(
                    text,
                    max_block_chars=args.max_block_chars,
                )

                if not document_chunks:
                    skipped_documents += 1
                    continue

                max_chunks_in_document = max(max_chunks_in_document, len(document_chunks))

                for chunk_index, chunk in enumerate(document_chunks):
                    uploader.add_record(
                        {
                            "chunk_id": next_chunk_id,
                            "chunk_index": chunk_index,
                            "source_row": row_index,
                            "path": path,
                            "title": title,
                            "extension": extension,
                            "text": chunk,
                        }
                    )
                    next_chunk_id += 1
                    total_chunks += 1

                if (
                    args.progress_every_documents > 0
                    and processed_documents % args.progress_every_documents == 0
                ):
                    avg_chunks = total_chunks / max(processed_documents, 1)
                    avg_text_size = total_text_bytes / max(processed_documents, 1)
                    logger.info(
                        f"Прогресс индексации: документов обработано {processed_documents}, "
                        f"пропущено {skipped_documents}, чанков загружено {uploader.uploaded_chunks}, "
                        f"чанков подготовлено {total_chunks}, "
                        f"среднее чанков на документ {avg_chunks:.2f}, "
                        f"средний текст {humanize_bytes(avg_text_size)}, "
                        f"макс. чанков в документе {max_chunks_in_document}, "
                        f"загруженный текст {humanize_bytes(uploader.uploaded_text_bytes)}"
                    )
        finally:
            progress_bar.close()
            uploader.close()

    logger.info(
        f"Индексация завершена: документов обработано {processed_documents}, "
        f"пропущено {skipped_documents}, "
        f"всего чанков {total_chunks}, "
        f"загружено чанков {uploader.uploaded_chunks}, "
        f"суммарный очищенный текст {humanize_bytes(total_text_bytes)}, "
        f"загруженный текст {humanize_bytes(uploader.uploaded_text_bytes)}"
    )


def main() -> None:
    configure_logging()
    configure_csv_field_limit()
    args = parse_args()

    logger.info(f"Source CSV: {settings.source_csv_path}")
    logger.info(
        f"Weaviate endpoint: {settings.weaviate_http_host}:{settings.weaviate_http_port}"
    )
    logger.info(f"Weaviate collection: {settings.weaviate_collection}")
    logger.info(
        f"Параметры: batch_size={args.batch_size}, "
        f"chunk_tokens={settings.chunk_tokens}, "
        f"chunk_overlap={settings.chunk_overlap}, "
        f"embedding_device={settings.embedding_device}, "
        f"max_block_chars={args.max_block_chars}"
    )

    ensure_weaviate_collection(recreate=args.recreate_collection)
    if torch is not None:
        logger.info(
            f"CUDA status: available={torch.cuda.is_available()}, "
            f"device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        )
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    stream_index_csv(args)
    logger.info("DONE")


if __name__ == "__main__":
    main()

import argparse
import csv
import logging
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from document_processing import (
    ExtractedDocument,
    ExtractedSegment,
    StreamingChunker,
    clean_extracted_text,
    deserialize_segments,
)
from embed_uploader import WeaviateBatchUploader
from settings import settings
from weaviate_store import ensure_weaviate_collection


logger = logging.getLogger("download_index")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Прочитать CSV с документами, разбить на чанки и загрузить в Weaviate."
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
    parser.add_argument(
        "--max-chunks-per-document",
        type=int,
        default=0,
        help="Если > 0, ограничить число чанков на документ.",
    )
    return parser.parse_args()


def resolve_text_column(fieldnames: list[str]) -> str:
    for candidate in ("clean_text", "content", "text"):
        if candidate in fieldnames:
            return candidate

    raise ValueError(
        f"CSV {settings.source_csv_path} не содержит текстовой колонки. "
        "Ожидалась одна из: clean_text, content, text."
    )


def build_document_from_row(row: dict[str, str], text_column: str) -> ExtractedDocument:
    raw_text = clean_extracted_text(str(row.get(text_column, "") or ""))
    segments = deserialize_segments(str(row.get("segments_json", "") or ""))
    if segments:
        return ExtractedDocument(
            text="\n\n".join(segment.text for segment in segments),
            segments=segments,
        )
    if not raw_text:
        return ExtractedDocument(text="", segments=[])
    return ExtractedDocument(text=raw_text, segments=[ExtractedSegment(text=raw_text)])


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

    processed_documents = 0
    skipped_documents = 0
    total_chunks = 0
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
                document = build_document_from_row(row, text_column)
                progress_bar.update(1)

                if not document.text:
                    skipped_documents += 1
                    continue

                chunks = chunker.chunk_document(
                    document,
                    max_block_chars=args.max_block_chars,
                )
                if not chunks:
                    skipped_documents += 1
                    continue

                if args.max_chunks_per_document > 0:
                    chunks = chunks[: args.max_chunks_per_document]

                processed_documents += 1
                for chunk in chunks:
                    uploader.add_record(
                        {
                            "chunk_id": next_chunk_id,
                            "chunk_index": chunk.chunk_index,
                            "source_row": row_index,
                            "path": path,
                            "title": title,
                            "extension": extension,
                            "page_start": chunk.page_start or 0,
                            "page_end": chunk.page_end or 0,
                            "source_locator": chunk.source_locator,
                            "text": chunk.text,
                        }
                    )
                    next_chunk_id += 1
                    total_chunks += 1

                if (
                    args.progress_every_documents > 0
                    and processed_documents % args.progress_every_documents == 0
                ):
                    logger.info(
                        f"Прогресс: обработано {processed_documents}, "
                        f"пропущено {skipped_documents}, загружено чанков {uploader.uploaded_chunks}"
                    )
        finally:
            progress_bar.close()
            uploader.close()

    logger.info(
        f"Готово: обработано {processed_documents}, "
        f"пропущено {skipped_documents}, загружено чанков {total_chunks}"
    )


def main() -> None:
    configure_logging()
    configure_csv_field_limit()
    args = parse_args()
    ensure_weaviate_collection(recreate=args.recreate_collection)
    stream_index_csv(args)


if __name__ == "__main__":
    main()

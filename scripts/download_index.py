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
    format_source_locator,
)
from embed_uploader import WeaviateBatchUploader
from settings import settings
from weaviate_store import ensure_weaviate_collection


logger = logging.getLogger("download_index")
DEFAULT_MAX_BLOCK_CHARS = max(settings.chunk_tokens * 8, 4096)
OPTIONAL_METADATA_COLUMNS = (
    "domain",
    "department",
    "doc_type",
    "language",
    "acl_groups",
    "created_at",
)


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


def build_section_documents(document: ExtractedDocument) -> list[ExtractedDocument]:
    if not document.segments:
        if not document.text:
            return []
        return [ExtractedDocument(text=document.text, segments=[ExtractedSegment(text=document.text)])]

    output: list[ExtractedDocument] = []
    for segment in document.segments:
        if not segment.text.strip():
            continue
        output.append(
            ExtractedDocument(
                text=segment.text,
                segments=[segment],
            )
        )
    return output


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
    next_document_id = 0
    next_section_id = 0
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
                metadata = {key: row.get(key, "") for key in OPTIONAL_METADATA_COLUMNS}
                document = build_document_from_row(row, text_column)
                progress_bar.update(1)

                if not document.text:
                    skipped_documents += 1
                    continue

                sections = build_section_documents(document)
                if not sections:
                    skipped_documents += 1
                    continue

                metadata = {key: row.get(key, "") for key in OPTIONAL_METADATA_COLUMNS}
                document_id = next_document_id
                next_document_id += 1

                uploader.add_document_record(
                    {
                        "document_id": document_id,
                        "source_row": row_index,
                        "path": path,
                        "title": title,
                        "extension": extension,
                        "domain": metadata["domain"],
                        "department": metadata["department"],
                        "doc_type": metadata["doc_type"],
                        "language": metadata["language"],
                        "acl_groups": metadata["acl_groups"],
                        "created_at": metadata["created_at"],
                        "text": document.text,
                    }
                )

                document_chunk_count = 0
                for section_index, section_document in enumerate(sections):
                    section_id = next_section_id
                    next_section_id += 1
                    segment = section_document.segments[0]
                    section_page_start = segment.page_start or 0
                    section_page_end = segment.page_end or 0
                    uploader.add_section_record(
                        {
                            "section_id": section_id,
                            "document_id": document_id,
                            "section_index": section_index,
                            "source_row": row_index,
                            "path": path,
                            "title": title,
                            "extension": extension,
                            "domain": metadata["domain"],
                            "department": metadata["department"],
                            "doc_type": metadata["doc_type"],
                            "language": metadata["language"],
                            "acl_groups": metadata["acl_groups"],
                            "created_at": metadata["created_at"],
                            "page_start": section_page_start,
                            "page_end": section_page_end,
                            "source_locator": format_source_locator(
                                segment.page_start,
                                segment.page_end,
                            ),
                            "text": section_document.text,
                        }
                    )

                    chunks = chunker.chunk_document(
                        section_document,
                        max_block_chars=args.max_block_chars,
                    )
                    if args.max_chunks_per_document > 0:
                        remaining = args.max_chunks_per_document - document_chunk_count
                        if remaining <= 0:
                            break
                        chunks = chunks[:remaining]

                    for chunk in chunks:
                        uploader.add_chunk_record(
                            {
                                "chunk_id": next_chunk_id,
                                "document_id": document_id,
                                "section_id": section_id,
                                "chunk_index": document_chunk_count,
                                "source_row": row_index,
                                "path": path,
                                "title": title,
                                "extension": extension,
                                "domain": metadata["domain"],
                                "department": metadata["department"],
                                "doc_type": metadata["doc_type"],
                                "language": metadata["language"],
                                "acl_groups": metadata["acl_groups"],
                                "created_at": metadata["created_at"],
                                "page_start": chunk.page_start or 0,
                                "page_end": chunk.page_end or 0,
                                "source_locator": chunk.source_locator,
                                "text": chunk.text,
                            }
                        )
                        next_chunk_id += 1
                        total_chunks += 1
                        document_chunk_count += 1

                    if args.max_chunks_per_document > 0 and document_chunk_count >= args.max_chunks_per_document:
                        break

                if document_chunk_count == 0:
                    skipped_documents += 1
                    continue

                processed_documents += 1

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

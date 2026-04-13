import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from document_processing import StreamingChunker, extract_document_from_path
from settings import settings


DEFAULT_MAX_BLOCK_CHARS = max(settings.chunk_tokens * 8, 4096)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Локально распарсить один файл, разбить на чанки и сохранить результат рядом."
    )
    parser.add_argument("file", help="Путь к локальному файлу.")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Свое имя префикса для выходных файлов. По умолчанию используется имя исходного файла.",
    )
    parser.add_argument(
        "--max-block-chars",
        type=int,
        default=DEFAULT_MAX_BLOCK_CHARS,
        help="Максимальный размер текстового блока до принудительного разрезания.",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=settings.chunk_tokens,
        help="Размер чанка в токенах.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help="Перекрытие между чанками в токенах.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=settings.embedding_model_name,
        help="Модель токенизатора для preview.",
    )
    return parser.parse_args()


def build_output_paths(source_path: Path, output_prefix: str) -> tuple[Path, Path]:
    base_name = output_prefix.strip() or source_path.name
    extracted_path = source_path.parent / f"{base_name}.extracted.txt"
    chunks_path = source_path.parent / f"{base_name}.chunks.json"
    return extracted_path, chunks_path


def main() -> None:
    args = parse_args()
    source_path = Path(args.file).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Файл не найден: {source_path}")

    document = extract_document_from_path(source_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    chunker = StreamingChunker(
        tokenizer,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = chunker.chunk_document(document, max_block_chars=args.max_block_chars)
    extracted_path, chunks_path = build_output_paths(source_path, args.output_prefix)

    extracted_path.write_text(document.text, encoding="utf-8")
    chunks_payload = {
        "source_file": str(source_path),
        "tokenizer_model": args.tokenizer_model,
        "chunk_tokens": args.chunk_tokens,
        "chunk_overlap": args.chunk_overlap,
        "max_block_chars": args.max_block_chars,
        "segments": [
            {
                "page_start": segment.page_start,
                "page_end": segment.page_end,
                "text": segment.text,
            }
            for segment in document.segments
        ],
        "chunks": [
            {
                "chunk_index": chunk.chunk_index,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "source_locator": chunk.source_locator,
                "text": chunk.text,
            }
            for chunk in chunks
        ],
    }
    chunks_path.write_text(
        json.dumps(chunks_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Исходный файл: {source_path}")
    print(f"Извлеченный текст: {extracted_path}")
    print(f"Чанки JSON: {chunks_path}")
    print(f"Сегментов: {len(document.segments)}")
    print(f"Чанков: {len(chunks)}")


if __name__ == "__main__":
    main()

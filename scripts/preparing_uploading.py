import argparse
import csv
import logging
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from document_processing import extract_document_from_bytes, serialize_segments
from source_readers import (
    DEFAULT_SMB_DOMAIN,
    DEFAULT_SMB_PASS,
    DEFAULT_SMB_ROOT,
    DEFAULT_SMB_SERVER,
    DEFAULT_SMB_USER,
    SourceDocument,
    iter_local_documents,
    iter_smb_documents,
    mask_secret,
    register_smb_session,
    validate_smb_access,
)


logger = logging.getLogger("preparing_uploading")
OUTPUT_FIELDS = [
    "file_path",
    "file_name",
    "extension",
    "domain",
    "department",
    "doc_type",
    "language",
    "acl_groups",
    "created_at",
    "content",
    "segments_json",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Прочитать документы из источника и сохранить извлеченный текст в CSV."
    )
    parser.add_argument(
        "--source-type",
        choices=("smb", "local"),
        default="smb",
        help="Тип источника документов.",
    )
    parser.add_argument("--root-path", default=DEFAULT_SMB_ROOT)
    parser.add_argument("--server", default=DEFAULT_SMB_SERVER)
    parser.add_argument("--user", default=DEFAULT_SMB_USER)
    parser.add_argument("--password", default=DEFAULT_SMB_PASS)
    parser.add_argument("--domain", default=DEFAULT_SMB_DOMAIN)
    parser.add_argument(
        "--csv-path",
        default=str((Path(__file__).resolve().parent / "documents_data.csv")),
    )
    parser.add_argument(
        "--progress-every-files",
        type=int,
        default=100,
        help="Как часто печатать прогресс по документам.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Ограничить число документов для записи в CSV.",
    )
    return parser.parse_args()


def ensure_output_csv_schema(csv_path: Path) -> None:
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as existing_file:
        reader = csv.reader(existing_file)
        existing_header = next(reader, [])

    if existing_header == OUTPUT_FIELDS:
        return

    if not existing_header:
        return

    if not set(existing_header).issubset(OUTPUT_FIELDS):
        raise ValueError(
            f"CSV {csv_path} имеет несовместимую схему: {existing_header}. "
            f"Ожидалась подмножество {OUTPUT_FIELDS}."
        )

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=str(csv_path.parent),
    ) as temporary_file:
        temp_path = Path(temporary_file.name)
        writer = csv.DictWriter(temporary_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        with csv_path.open("r", encoding="utf-8", newline="") as source_file:
            reader = csv.DictReader(source_file)
            for row in reader:
                writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})

    temp_path.replace(csv_path)


def load_processed_files(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return {
            str(row.get("file_path", "") or "").strip()
            for row in reader
            if str(row.get("file_path", "") or "").strip()
        }


def iter_source_documents(args: argparse.Namespace):
    if args.source_type == "local":
        return iter_local_documents(args.root_path)

    register_smb_session(
        server=args.server,
        username=args.user,
        password=args.password,
        domain=args.domain,
    )
    validate_smb_access(args.root_path)
    return iter_smb_documents(args.root_path)


def write_document_row(writer: csv.DictWriter, document: SourceDocument) -> bool:
    extracted_document = extract_document_from_bytes(
        document.content_bytes,
        document.extension,
    )
    if not extracted_document.text:
        return False

    writer.writerow(
        {
            "file_path": document.path,
            "file_name": document.title,
            "extension": document.extension,
            "domain": "",
            "department": "",
            "doc_type": "",
            "language": "",
            "acl_groups": "",
            "created_at": "",
            "content": extracted_document.text,
            "segments_json": serialize_segments(extracted_document.segments),
        }
    )
    return True


def main() -> None:
    configure_logging()
    args = parse_args()
    csv_path = Path(args.csv_path).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_output_csv_schema(csv_path)
    processed_files = load_processed_files(csv_path)

    logger.info(f"Source type: {args.source_type}")
    logger.info(f"Root path: {args.root_path}")
    logger.info(f"CSV path: {csv_path}")
    if args.source_type == "smb":
        logger.info(
            "SMB params: "
            f"server={args.server}, domain={args.domain}, "
            f"user={args.user}, password={mask_secret(args.password)}"
        )

    documents_written = 0
    documents_seen = 0
    empty_documents = 0
    file_exists = csv_path.exists()

    with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDS)
        if not file_exists:
            writer.writeheader()

        for document in iter_source_documents(args):
            if document.path in processed_files:
                continue

            documents_seen += 1
            was_written = write_document_row(writer, document)
            if was_written:
                documents_written += 1
            else:
                empty_documents += 1

            if (
                args.progress_every_files > 0
                and documents_seen % args.progress_every_files == 0
            ):
                csv_file.flush()
                logger.info(
                    f"Прогресс: просмотрено {documents_seen}, "
                    f"записано {documents_written}, пустых {empty_documents}"
                )

            if args.max_documents is not None and documents_seen >= args.max_documents:
                break

    logger.info(
        f"Готово: просмотрено {documents_seen}, "
        f"записано {documents_written}, пустых {empty_documents}"
    )


if __name__ == "__main__":
    main()

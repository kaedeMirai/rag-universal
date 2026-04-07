import argparse
import csv
import io
import logging
import math
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

try:
    from loguru import logger
except Exception:
    logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xls", ".xlsx", ".txt"}
DEFAULT_AVG_CHARS_PER_TOKEN = 4.0
DEFAULT_EMBEDDING_DIMENSION = 1024
FLOAT32_SIZE_BYTES = 4
DEFAULT_SMB_SERVER = os.getenv("RAG_SMB_SERVER", "smb_server")
DEFAULT_SMB_USER = os.getenv("RAG_SMB_USER", "username")
DEFAULT_SMB_PASS = os.getenv("RAG_SMB_PASS", "password")
DEFAULT_SMB_DOMAIN = os.getenv("RAG_SMB_DOMAIN", "smb_domain")
DEFAULT_SMB_ROOT = os.getenv("RAG_SMB_ROOT", rf"\\{DEFAULT_SMB_SERVER}\share")
CONTROL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
WHITESPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def configure_logging(verbose: bool = True) -> None:
    if not verbose:
        return

    if hasattr(logger, "remove") and hasattr(logger, "add"):
        logger.remove()
        logger.add(
            sys.stderr,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def register_smb_session(
    *,
    server: str,
    username: str,
    password: str,
    domain: str | None = None,
) -> None:
    smbclient = _get_smbclient()
    full_username = f"{domain}\\{username}" if domain else username
    smbclient.register_session(
        server,
        username=full_username,
        password=password,
    )


def _get_smbclient():
    import smbclient

    return smbclient


def mask_secret(value: str | None, keep: int = 2) -> str:
    if not value:
        return "<empty>"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}***{value[-keep:]}"


def validate_smb_access(root_path: str) -> None:
    smbclient = _get_smbclient()
    try:
        entries = list(smbclient.listdir(root_path))
    except Exception as exc:
        raise RuntimeError(
            f"Не удалось открыть SMB-путь {root_path}. "
            f"Проверь server/user/password/domain/root-path. Ошибка: {exc}"
        ) from exc

    logger.info(
        f"SMB доступ подтвержден: {root_path} " f"(элементов в корне: {len(entries)})"
    )


def extract_text_from_pdf(file_obj: io.BytesIO) -> str:
    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(file_obj)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def extract_text_from_docx(file_obj: io.BytesIO) -> str:
    try:
        from docx import Document

        document = Document(file_obj)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except Exception:
        return ""


def _extract_xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []

    namespace = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    shared_strings: list[str] = []

    for string_item in root.findall("ns:si", namespace):
        fragments = [
            node.text or "" for node in string_item.findall(".//ns:t", namespace)
        ]
        shared_strings.append("".join(fragments).strip())

    return shared_strings


def _extract_xlsx_sheet_names(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    workbook_path = "xl/workbook.xml"
    relations_path = "xl/_rels/workbook.xml.rels"
    namespace = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
    }

    try:
        workbook_root = ET.fromstring(archive.read(workbook_path))
        relations_root = ET.fromstring(archive.read(relations_path))
    except KeyError:
        return []

    relation_targets: dict[str, str] = {}
    for relation in relations_root.findall("pkg:Relationship", namespace):
        relation_id = relation.attrib.get("Id")
        target = relation.attrib.get("Target")
        if not relation_id or not target:
            continue
        normalized_target = target.lstrip("/")
        if not normalized_target.startswith("xl/"):
            normalized_target = f"xl/{normalized_target}"
        relation_targets[relation_id] = normalized_target

    sheets: list[tuple[str, str]] = []
    for sheet in workbook_root.findall("main:sheets/main:sheet", namespace):
        sheet_name = (sheet.attrib.get("name") or "").strip() or "Sheet"
        relation_id = sheet.attrib.get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        )
        sheet_path = relation_targets.get(relation_id or "")
        if sheet_path:
            sheets.append((sheet_name, sheet_path))

    return sheets


def _extract_xlsx_cell_value(
    cell: ET.Element,
    *,
    shared_strings: list[str],
    namespace: dict[str, str],
) -> str:
    cell_type = cell.attrib.get("t")
    raw_value = cell.findtext("ns:v", default="", namespaces=namespace).strip()

    if cell_type == "s":
        if raw_value.isdigit():
            shared_string_index = int(raw_value)
            if 0 <= shared_string_index < len(shared_strings):
                return shared_strings[shared_string_index]
        return ""

    if cell_type == "inlineStr":
        inline_text = "".join(
            node.text or "" for node in cell.findall(".//ns:is//ns:t", namespace)
        )
        return inline_text.strip()

    if cell_type == "b":
        return "TRUE" if raw_value == "1" else "FALSE"

    if cell_type == "str":
        return raw_value

    formula = cell.findtext("ns:f", default="", namespaces=namespace).strip()
    if formula and raw_value:
        return f"{formula} = {raw_value}"

    return raw_value


def extract_text_from_xlsx_fallback(file_obj: io.BytesIO) -> str:
    namespace = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    try:
        with zipfile.ZipFile(file_obj) as archive:
            shared_strings = _extract_xlsx_shared_strings(archive)
            sheets = _extract_xlsx_sheet_names(archive)
            if not sheets:
                sheets = [
                    (Path(name).stem, name)
                    for name in sorted(archive.namelist())
                    if name.startswith("xl/worksheets/") and name.endswith(".xml")
                ]

            rendered_sheets: list[str] = []
            for sheet_name, sheet_path in sheets:
                try:
                    sheet_root = ET.fromstring(archive.read(sheet_path))
                except KeyError:
                    continue

                rendered_rows: list[str] = []
                for row in sheet_root.findall(".//ns:sheetData/ns:row", namespace):
                    row_values = []
                    for cell in row.findall("ns:c", namespace):
                        value = _extract_xlsx_cell_value(
                            cell,
                            shared_strings=shared_strings,
                            namespace=namespace,
                        )
                        if value:
                            row_values.append(value)

                    if row_values:
                        rendered_rows.append(" | ".join(row_values))

                if rendered_rows:
                    rendered_sheets.append(
                        f"Sheet: {sheet_name}\n" + "\n".join(rendered_rows)
                    )

            return "\n\n".join(rendered_sheets)
    except zipfile.BadZipFile:
        return ""
    except Exception:
        return ""


def extract_text_from_excel(file_obj: io.BytesIO, extension: str) -> str:
    try:
        dataframe_by_sheet = pd.read_excel(file_obj, sheet_name=None)
    except Exception:
        dataframe_by_sheet = None

    if dataframe_by_sheet is not None:
        output: list[str] = []
        for sheet_name, sheet in dataframe_by_sheet.items():
            cleaned_sheet = sheet.dropna(how="all").dropna(axis=1, how="all")
            if cleaned_sheet.empty:
                continue

            rendered_sheet = (
                cleaned_sheet.fillna("")
                .astype(str)
                .apply(
                    lambda row: " | ".join(
                        value.strip()
                        for value in row
                        if isinstance(value, str) and value.strip()
                    ),
                    axis=1,
                )
            )
            rendered_rows = [row for row in rendered_sheet.tolist() if row]
            if rendered_rows:
                output.append(f"Sheet: {sheet_name}\n" + "\n".join(rendered_rows))

        if output:
            return "\n\n".join(output)

    if extension == ".xlsx":
        file_obj.seek(0)
        return extract_text_from_xlsx_fallback(file_obj)

    return ""


def extract_text_from_txt(file_obj: io.BytesIO) -> str:
    try:
        return file_obj.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_text_from_bytes(content_bytes: bytes, extension: str) -> str:
    file_stream = io.BytesIO(content_bytes)
    try:
        if extension == ".pdf":
            return extract_text_from_pdf(file_stream)
        if extension == ".docx":
            return extract_text_from_docx(file_stream)
        if extension in {".xls", ".xlsx"}:
            return extract_text_from_excel(file_stream, extension)
        if extension == ".txt":
            return extract_text_from_txt(file_stream)
        return ""
    finally:
        file_stream.close()


def clean_extracted_text(text: str) -> str:
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = CONTROL_CHARACTERS_RE.sub("", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines())
    cleaned = MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


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


def estimate_chunks_from_token_count(
    token_count: int,
    *,
    chunk_tokens: int,
    chunk_overlap: int,
) -> int:
    if token_count <= 0:
        return 0

    if token_count <= chunk_tokens:
        return 1

    step = max(chunk_tokens - chunk_overlap, 1)
    return 1 + math.ceil((token_count - chunk_tokens) / step)


def load_processed_files(csv_path: Path | None) -> set[str]:
    if csv_path is None or not csv_path.exists():
        return set()

    try:
        existing_df = pd.read_csv(csv_path, usecols=["file_path"])
    except Exception:
        logger.warning(
            "Не удалось прочитать существующий CSV со списком обработанных файлов."
        )
        return set()

    return set(existing_df["file_path"].dropna().astype(str).tolist())


def get_top_level_directory(root_path: str, file_root: str) -> str:
    normalized_root = root_path.replace("\\", "/").rstrip("/")
    normalized_file_root = file_root.replace("\\", "/")

    if normalized_file_root.startswith(normalized_root):
        relative_root = normalized_file_root[len(normalized_root) :].strip("/")
    else:
        relative_root = normalized_file_root.strip("/")

    if not relative_root:
        return "."

    return relative_root.split("/", 1)[0]


def estimate_tokens_from_text(text: str, avg_chars_per_token: float) -> int:
    stripped_text = text.strip()
    if not stripped_text:
        return 0
    return max(1, math.ceil(len(stripped_text) / avg_chars_per_token))


def process_smb_garbage_collector(
    root_path: str,
    csv_path: Path,
    *,
    progress_every_files: int = 100,
) -> None:
    smbclient = _get_smbclient()
    processed_files = load_processed_files(csv_path)
    if processed_files:
        logger.info(f"Найдено уже обработанных файлов: {len(processed_files)}")

    file_exists = csv_path.exists()
    processed_now = 0
    seen_supported = 0
    empty_text_files = 0
    written_text_bytes = 0

    with csv_path.open(mode="a", encoding="utf-8", newline="") as csv_file:
        fieldnames = ["file_path", "file_name", "extension", "content"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for root, _, files in smbclient.walk(root_path):
            for file_name in files:
                extension = Path(file_name).suffix.lower()
                if extension not in SUPPORTED_EXTENSIONS:
                    continue

                seen_supported += 1
                file_path_raw = os.path.join(root, file_name)
                file_path_clean = file_path_raw.replace("\\", "/")

                if file_path_clean in processed_files:
                    continue

                try:
                    logger.info(f"Обработка: {file_name}")
                    with smbclient.open_file(file_path_raw, mode="rb") as file_obj:
                        content_bytes = file_obj.read()

                    raw_text = extract_text_from_bytes(content_bytes, extension)
                    text = clean_extracted_text(raw_text)
                    if not text:
                        empty_text_files += 1

                    writer.writerow(
                        {
                            "file_path": file_path_clean,
                            "file_name": file_name,
                            "extension": extension,
                            "content": text,
                        }
                    )
                    processed_now += 1
                    written_text_bytes += len(text.encode("utf-8", errors="ignore"))

                    if (
                        progress_every_files > 0
                        and processed_now % progress_every_files == 0
                    ):
                        csv_file.flush()
                        logger.info(
                            f"Прогресс выгрузки: записано {processed_now} новых файлов "
                            f"(просмотрено поддерживаемых: {seen_supported}, "
                            f"пустых текстов: {empty_text_files}, "
                            f"накопленный текст: {humanize_bytes(written_text_bytes)})"
                        )
                except Exception as exc:
                    logger.error(f"Ошибка доступа {file_name}: {exc}")

        csv_file.flush()
        logger.info(
            f"Выгрузка CSV завершена: записано {processed_now} файлов, "
            f"пустых текстов: {empty_text_files}, "
            f"общий размер текста: {humanize_bytes(written_text_bytes)}"
        )


def scan_smb_inventory(
    *,
    root_path: str,
    csv_path: Path | None = None,
    chunk_tokens: int = 512,
    chunk_overlap: int = 100,
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    avg_chars_per_token: float = DEFAULT_AVG_CHARS_PER_TOKEN,
    sample_limit_per_extension: int = 15,
    progress_every_files: int = 500,
) -> dict[str, Any]:
    smbclient = _get_smbclient()
    processed_files = load_processed_files(csv_path)
    scan_started_at = pd.Timestamp.utcnow()

    extension_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "files": 0,
            "size_bytes": 0,
            "sample_files": 0,
            "sample_input_bytes": 0,
            "sample_text_chars": 0,
            "sample_text_bytes": 0,
            "sample_tokens_estimated": 0,
            "sample_chunks_estimated": 0,
            "sample_embedding_bytes": 0,
        }
    )
    top_directory_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "size_bytes": 0}
    )
    errors: list[dict[str, str]] = []
    missing_files: list[dict[str, str]] = []

    total_files = 0
    total_size_bytes = 0
    total_supported_files = 0
    total_supported_size_bytes = 0
    total_processed_matches = 0
    unsupported_extensions = Counter()

    for root, _, files in smbclient.walk(root_path):
        top_directory = get_top_level_directory(root_path, root)

        for file_name in files:
            total_files += 1
            extension = Path(file_name).suffix.lower()
            file_path_raw = os.path.join(root, file_name)
            file_path_clean = file_path_raw.replace("\\", "/")

            try:
                stat_result = smbclient.stat(file_path_raw)
                file_size = int(stat_result.st_size)
            except Exception as exc:
                errors.append(
                    {
                        "file_path": file_path_clean,
                        "stage": "stat",
                        "error": str(exc),
                    }
                )
                continue

            total_size_bytes += file_size

            if progress_every_files > 0 and total_files % progress_every_files == 0:
                logger.info(
                    f"Прогресс инвентаризации: просмотрено {total_files} файлов, "
                    f"поддерживаемых найдено {total_supported_files}, "
                    f"текущий каталог {top_directory}"
                )

            if extension not in SUPPORTED_EXTENSIONS:
                unsupported_extensions[extension or "<no_extension>"] += 1
                continue

            total_supported_files += 1
            total_supported_size_bytes += file_size

            extension_bucket = extension_stats[extension]
            extension_bucket["files"] += 1
            extension_bucket["size_bytes"] += file_size

            directory_bucket = top_directory_stats[top_directory]
            directory_bucket["files"] += 1
            directory_bucket["size_bytes"] += file_size

            if file_path_clean in processed_files:
                total_processed_matches += 1
            else:
                missing_files.append(
                    {
                        "file_path": file_path_clean,
                        "extension": extension,
                        "top_directory": top_directory,
                    }
                )

            if extension_bucket["sample_files"] >= sample_limit_per_extension:
                continue

            try:
                with smbclient.open_file(file_path_raw, mode="rb") as file_obj:
                    content_bytes = file_obj.read()

                extracted_text = clean_extracted_text(
                    extract_text_from_bytes(content_bytes, extension)
                )
            except Exception as exc:
                errors.append(
                    {
                        "file_path": file_path_clean,
                        "stage": "sample_read",
                        "error": str(exc),
                    }
                )
                continue

            estimated_tokens = estimate_tokens_from_text(
                extracted_text,
                avg_chars_per_token=avg_chars_per_token,
            )
            estimated_chunks = estimate_chunks_from_token_count(
                estimated_tokens,
                chunk_tokens=chunk_tokens,
                chunk_overlap=chunk_overlap,
            )
            estimated_embedding_bytes = (
                estimated_chunks * embedding_dimension * FLOAT32_SIZE_BYTES
            )

            extension_bucket["sample_files"] += 1
            extension_bucket["sample_input_bytes"] += file_size
            extension_bucket["sample_text_chars"] += len(extracted_text)
            extension_bucket["sample_text_bytes"] += len(
                extracted_text.encode("utf-8", errors="ignore")
            )
            extension_bucket["sample_tokens_estimated"] += estimated_tokens
            extension_bucket["sample_chunks_estimated"] += estimated_chunks
            extension_bucket["sample_embedding_bytes"] += estimated_embedding_bytes

    extension_rows: list[dict[str, Any]] = []
    estimated_total_tokens = 0
    estimated_total_chunks = 0
    estimated_total_embedding_bytes = 0

    for extension, stats in sorted(extension_stats.items()):
        sample_files = stats["sample_files"]
        if sample_files > 0:
            avg_text_bytes_per_file = stats["sample_text_bytes"] / sample_files
            avg_tokens_per_file = stats["sample_tokens_estimated"] / sample_files
            avg_chunks_per_file = stats["sample_chunks_estimated"] / sample_files
            avg_embedding_bytes_per_file = (
                stats["sample_embedding_bytes"] / sample_files
            )
            avg_text_bytes_per_input_byte = stats["sample_text_bytes"] / max(
                stats["sample_input_bytes"],
                1,
            )
        else:
            avg_text_bytes_per_file = 0.0
            avg_tokens_per_file = 0.0
            avg_chunks_per_file = 0.0
            avg_embedding_bytes_per_file = 0.0
            avg_text_bytes_per_input_byte = 0.0

        extension_estimated_tokens = int(round(avg_tokens_per_file * stats["files"]))
        extension_estimated_chunks = int(round(avg_chunks_per_file * stats["files"]))
        extension_estimated_embedding_bytes = int(
            round(avg_embedding_bytes_per_file * stats["files"])
        )

        estimated_total_tokens += extension_estimated_tokens
        estimated_total_chunks += extension_estimated_chunks
        estimated_total_embedding_bytes += extension_estimated_embedding_bytes

        extension_rows.append(
            {
                "extension": extension,
                "files": stats["files"],
                "size_bytes": stats["size_bytes"],
                "size_human": humanize_bytes(stats["size_bytes"]),
                "sample_files": sample_files,
                "avg_text_bytes_per_file_from_sample": round(
                    avg_text_bytes_per_file, 2
                ),
                "avg_text_bytes_per_input_byte": round(
                    avg_text_bytes_per_input_byte, 4
                ),
                "avg_tokens_per_file_estimated": round(avg_tokens_per_file, 2),
                "avg_chunks_per_file_estimated": round(avg_chunks_per_file, 2),
                "estimated_tokens_total": extension_estimated_tokens,
                "estimated_chunks_total": extension_estimated_chunks,
                "estimated_embedding_size_bytes": extension_estimated_embedding_bytes,
                "estimated_embedding_size_human": humanize_bytes(
                    extension_estimated_embedding_bytes
                ),
            }
        )

    extensions_df = pd.DataFrame(extension_rows).sort_values(
        by=["files", "size_bytes"],
        ascending=[False, False],
    )
    if not extensions_df.empty:
        extensions_df = extensions_df.reset_index(drop=True)

    top_directories_df = pd.DataFrame(
        [
            {
                "top_directory": top_directory,
                "files": stats["files"],
                "size_bytes": stats["size_bytes"],
                "size_human": humanize_bytes(stats["size_bytes"]),
            }
            for top_directory, stats in sorted(
                top_directory_stats.items(),
                key=lambda item: (-item[1]["files"], -item[1]["size_bytes"]),
            )
        ]
    )
    if not top_directories_df.empty:
        top_directories_df = top_directories_df.reset_index(drop=True)

    missing_files_df = pd.DataFrame(missing_files)
    if not missing_files_df.empty:
        missing_files_df = missing_files_df.reset_index(drop=True)

    errors_df = pd.DataFrame(errors)
    if not errors_df.empty:
        errors_df = errors_df.reset_index(drop=True)

    unsupported_extensions_df = pd.DataFrame(
        [
            {"extension": extension, "files": count}
            for extension, count in unsupported_extensions.most_common()
        ]
    )

    summary = {
        "root_path": root_path,
        "scan_started_at": scan_started_at.isoformat(),
        "total_files_seen": total_files,
        "total_size_bytes_seen": total_size_bytes,
        "total_size_human_seen": humanize_bytes(total_size_bytes),
        "supported_files": total_supported_files,
        "supported_size_bytes": total_supported_size_bytes,
        "supported_size_human": humanize_bytes(total_supported_size_bytes),
        "processed_files_in_csv": len(processed_files),
        "matched_processed_files": total_processed_matches,
        "missing_supported_files": max(
            total_supported_files - total_processed_matches, 0
        ),
        "coverage_percent": round(
            (
                (total_processed_matches / total_supported_files * 100)
                if total_supported_files
                else 0.0
            ),
            2,
        ),
        "estimated_total_tokens": estimated_total_tokens,
        "estimated_total_chunks": estimated_total_chunks,
        "estimated_total_embedding_size_bytes": estimated_total_embedding_bytes,
        "estimated_total_embedding_size_human": humanize_bytes(
            estimated_total_embedding_bytes
        ),
        "chunk_tokens": chunk_tokens,
        "chunk_overlap": chunk_overlap,
        "embedding_dimension": embedding_dimension,
        "avg_chars_per_token_assumption": avg_chars_per_token,
        "sample_limit_per_extension": sample_limit_per_extension,
    }

    return {
        "summary": summary,
        "extensions_df": extensions_df,
        "top_directories_df": top_directories_df,
        "missing_files_df": missing_files_df,
        "errors_df": errors_df,
        "unsupported_extensions_df": unsupported_extensions_df,
    }


def print_inventory_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    logger.info(
        f"Поддерживаемые файлы: {summary['supported_files']} шт., "
        f"общий размер: {summary['supported_size_human']}"
    )
    logger.info(
        f"Оценка после чанкинга: {summary['estimated_total_chunks']} чанков, "
        f"размер эмбеддингов: {summary['estimated_total_embedding_size_human']}"
    )
    logger.info(
        f"Покрытие текущим CSV: {summary['coverage_percent']}% "
        f"({summary['matched_processed_files']}/{summary['supported_files']})"
    )


def print_dataframe_preview(
    title: str, dataframe: pd.DataFrame, limit: int = 20
) -> None:
    if dataframe.empty:
        logger.info(f"{title}: данных нет")
        return

    logger.info(f"{title}:\n{dataframe.head(limit).to_string(index=False)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Инвентаризация и выгрузка документов из SMB-шары."
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
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSION,
    )
    parser.add_argument("--sample-limit-per-extension", type=int, default=15)
    parser.add_argument("--progress-every-files", type=int, default=500)
    parser.add_argument(
        "--process",
        action="store_true",
        help="Вместо инвентаризации выполнить полную выгрузку содержимого файлов в CSV.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    csv_path = Path(args.csv_path).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Старт preparing_uploading.py")
    logger.info(
        f"Параметры SMB: server={args.server}, domain={args.domain}, "
        f"user={args.user}, password={mask_secret(args.password)}"
    )
    logger.info(f"Корневая SMB-директория: {args.root_path}")
    logger.info(f"CSV путь: {csv_path}")

    register_smb_session(
        server=args.server,
        username=args.user,
        password=args.password,
        domain=args.domain,
    )
    logger.info("SMB сессия зарегистрирована")
    validate_smb_access(args.root_path)

    if args.process:
        logger.info("Режим: полная выгрузка содержимого файлов в CSV")
        process_smb_garbage_collector(
            args.root_path,
            csv_path,
            progress_every_files=args.progress_every_files,
        )
        logger.info("Выгрузка завершена")
        return

    logger.info("Режим: только инвентаризация, без копирования содержимого в CSV")
    report = scan_smb_inventory(
        root_path=args.root_path,
        csv_path=csv_path,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        embedding_dimension=args.embedding_dimension,
        sample_limit_per_extension=args.sample_limit_per_extension,
        progress_every_files=args.progress_every_files,
    )
    print_inventory_report(report)
    print_dataframe_preview("Статистика по расширениям", report["extensions_df"])
    print_dataframe_preview("Топ директорий", report["top_directories_df"])
    print_dataframe_preview("Первые необработанные файлы", report["missing_files_df"])
    print_dataframe_preview("Ошибки", report["errors_df"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        configure_logging()
        logger.exception(f"Скрипт завершился с ошибкой: {exc}")
        raise

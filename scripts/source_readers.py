from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


SUPPORTED_EXTENSIONS = {
    item.strip().lower()
    for item in os.getenv("RAG_SUPPORTED_EXTENSIONS", ".pdf,.docx,.txt").split(",")
    if item.strip()
}

SKIP_NAME_PATTERNS = [
    re.compile(pattern.strip(), re.IGNORECASE)
    for pattern in os.getenv(
        "RAG_SKIP_NAME_PATTERNS",
        r"(^|[_\-\s])logs?\d*([_\-\s\.]|$),^git[_\-\s\.],(^|[_\-\s])git([_\-\s\.]|$)",
    ).split(",")
    if pattern.strip()
]

LOG_LIKE_PATTERN = re.compile(r"(^|[_\-\s])logs?\d*([_\-\s\.]|$)", re.IGNORECASE)

EXTENSION_SIZE_LIMITS = {
    extension.strip().lower(): int(size_bytes.strip())
    for item in os.getenv(
        "RAG_MAX_FILE_SIZE_BYTES_BY_EXTENSION",
        ".txt=52428800,.pdf=104857600,.docx=52428800",
    ).split(",")
    if item.strip() and "=" in item
    for extension, size_bytes in [item.split("=", 1)]
    if extension.strip() and size_bytes.strip()
}


DEFAULT_SMB_SERVER = os.getenv("RAG_SMB_SERVER", "fs")
DEFAULT_SMB_USER = os.getenv("RAG_SMB_USER", "v.yugov")
DEFAULT_SMB_PASS = os.getenv("RAG_SMB_PASS", "4b5RwV8M")
DEFAULT_SMB_DOMAIN = os.getenv("RAG_SMB_DOMAIN", "okb")
DEFAULT_SMB_ROOT = os.getenv("RAG_SMB_ROOT", rf"\\{DEFAULT_SMB_SERVER}\share\08 ОПО")


@dataclass(frozen=True)
class SourceDocument:
    path: str
    title: str
    extension: str
    content_bytes: bytes
    size_bytes: int


def _get_smbclient():
    import smbclient

    return smbclient


def register_smb_session(
    *,
    server: str,
    username: str,
    password: str,
    domain: str | None = None,
) -> None:
    smbclient = _get_smbclient()
    full_username = f"{domain}\\{username}" if domain else username
    smbclient.register_session(server, username=full_username, password=password)


def validate_smb_access(root_path: str) -> None:
    smbclient = _get_smbclient()
    list(smbclient.listdir(root_path))


def mask_secret(value: str | None, keep: int = 2) -> str:
    if not value:
        return "<empty>"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}***{value[-keep:]}"


def should_skip_file(file_name: str, file_path: str) -> bool:
    normalized_name = file_name.lower()
    normalized_path = file_path.lower()
    if LOG_LIKE_PATTERN.search(normalized_name) or LOG_LIKE_PATTERN.search(
        normalized_path
    ):
        return True
    return any(
        pattern.search(normalized_name) or pattern.search(normalized_path)
        for pattern in SKIP_NAME_PATTERNS
    )


def should_skip_by_size(extension: str, file_size: int) -> bool:
    size_limit = EXTENSION_SIZE_LIMITS.get(extension)
    return size_limit is not None and file_size > size_limit


def iter_smb_documents(root_path: str) -> Iterator[SourceDocument]:
    smbclient = _get_smbclient()

    for root, _, files in smbclient.walk(root_path):
        for file_name in files:
            extension = Path(file_name).suffix.lower()
            if extension not in SUPPORTED_EXTENSIONS:
                continue

            file_path_raw = os.path.join(root, file_name)
            file_path_clean = file_path_raw.replace("\\", "/")
            if should_skip_file(file_name, file_path_clean):
                continue

            stat_result = smbclient.stat(file_path_raw)
            file_size = int(stat_result.st_size)
            if should_skip_by_size(extension, file_size):
                continue

            with smbclient.open_file(file_path_raw, mode="rb") as file_obj:
                content_bytes = file_obj.read()

            yield SourceDocument(
                path=file_path_clean,
                title=file_name,
                extension=extension,
                content_bytes=content_bytes,
                size_bytes=file_size,
            )


def iter_local_documents(root_path: str | Path) -> Iterator[SourceDocument]:
    source_root = Path(root_path).expanduser().resolve()
    for file_path in source_root.rglob("*"):
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue

        normalized_path = str(file_path)
        if should_skip_file(file_path.name, normalized_path):
            continue

        size_bytes = file_path.stat().st_size
        if should_skip_by_size(extension, size_bytes):
            continue

        yield SourceDocument(
            path=normalized_path,
            title=file_path.name,
            extension=extension,
            content_bytes=file_path.read_bytes(),
            size_bytes=size_bytes,
        )

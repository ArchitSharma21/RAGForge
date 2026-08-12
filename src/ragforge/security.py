from __future__ import annotations

import ipaddress
import re
import socket
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from .config import get_settings

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".rst", ".docx", ".pptx",
    ".csv", ".xlsx", ".xls", ".json", ".html", ".htm",
    ".xml", ".yaml", ".yml", ".py", ".js", ".ts", ".java",
    ".c", ".cpp", ".sql", ".log", ".png", ".jpg", ".jpeg", ".webp",
}

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(the\s+)?system\s+prompt",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"developer\s+message",
    r"exfiltrat(e|ion)",
    r"do\s+not\s+follow\s+the\s+user",
    r"override\s+(all\s+)?instructions",
]

UNSAFE_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|export|import|pragma|install|load|call|vacuum)\b",
    re.IGNORECASE,
)


def sanitize_filename(name: str) -> str:
    name = Path(name).name
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name)[:180] or "upload"


def validate_upload(path: Path) -> None:
    settings = get_settings()
    if not path.exists() or not path.is_file():
        raise ValueError(f"File not found: {path}")
    if path.stat().st_size > settings.max_upload_mb * 1024 * 1024:
        raise ValueError(f"{path.name} exceeds the {settings.max_upload_mb} MB upload limit")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS and path.suffix.lower() != ".zip":
        raise ValueError(f"Unsupported file type: {path.suffix or '(none)'}")


def safe_extract_zip(zip_path: Path, destination: Path) -> list[Path]:
    settings = get_settings()
    validate_upload(zip_path)
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    total_size = 0

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) > settings.max_archive_files:
            raise ValueError(f"ZIP contains more than {settings.max_archive_files} files")
        for member in members:
            total_size += member.file_size
            if total_size > settings.max_archive_uncompressed_mb * 1024 * 1024:
                raise ValueError("ZIP expands beyond the configured uncompressed size limit")
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError("Unsafe archive path detected")
            if member_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            clean_name = sanitize_filename(member_path.name)
            target = destination / clean_name
            if target.exists():
                stem, suffix = target.stem, target.suffix
                n = 2
                while target.exists():
                    target = destination / f"{stem}_{n}{suffix}"
                    n += 1
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
            extracted.append(target)
    return extracted


def prompt_injection_score(text: str) -> float:
    lowered = text.lower()[:12000]
    hits = sum(bool(re.search(pattern, lowered, flags=re.I)) for pattern in INJECTION_PATTERNS)
    return min(1.0, hits / 2.0)


def redact_basic_pii(text: str) -> str:
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]", text)
    text = re.sub(r"(?<!\d)(?:\+?\d[\d ()-]{8,}\d)(?!\d)", "[REDACTED_PHONE]", text)
    return text


def validate_readonly_sql(sql: str) -> str:
    candidate = sql.strip().strip("`").strip()
    candidate = re.sub(r"^sql\s*", "", candidate, flags=re.I).strip()
    if ";" in candidate.rstrip(";"):
        raise ValueError("Only a single SQL statement is allowed")
    if not re.match(r"^(select|with)\b", candidate, flags=re.I):
        raise ValueError("Only SELECT/CTE queries are allowed")
    if UNSAFE_SQL.search(candidate):
        raise ValueError("Unsafe SQL keyword detected")
    if " limit " not in f" {candidate.lower()} ":
        candidate = candidate.rstrip(";") + " LIMIT 200"
    return candidate


def is_safe_public_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        host = parsed.hostname.lower()
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
            return False
        try:
            infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
        except socket.gaierror:
            return False
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        return True
    except Exception:
        return False

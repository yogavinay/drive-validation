from __future__ import annotations

import ast
import re
from typing import Iterable, Tuple
from urllib.parse import urlparse


DRIVE_HOSTS = {"drive.google.com", "docs.google.com"}


def validate_drive_url(url: str) -> Tuple[bool, str]:
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL format"
    if parsed.scheme not in {"http", "https"}:
        return False, "Drive URL must use http/https"
    if parsed.netloc not in DRIVE_HOSTS:
        return False, "Only Google Drive URLs are allowed"
    # Reject common *file* share links.
    if re.search(r"/spreadsheets/d/|/file/d/", url):
        return False, "Drive URL must point to a folder share link (not a file link)"
    # Accept folder-style links.
    if not re.search(r"/drive/folders/|/folders/", url) and not re.search(r"[?&]id=", url):
        return False, "URL must point to a Drive folder share link"
    m = re.search(r"/(?:drive/)?folders/([a-zA-Z0-9_-]+)", url)
    if m:
        fid = m.group(1)
        if fid.upper() in {
            "YOUR_FOLDER_ID",
            "YOURFOLDERID",
            "FOLDER_ID",
            "REPLACE_ME",
            "EXAMPLE",
        }:
            return (
                False,
                "Drive URL still contains a placeholder folder id. Copy the real link from Drive "
                "(Share → Anyone with the link) and paste the full URL.",
            )
    return True, ""


FORBIDDEN_NAMES = {
    "os",
    "sys",
    "subprocess",
    "importlib",
    "socket",
    "shutil",
    "__import__",
    "eval",
    "exec",
    "open",
}

FORBIDDEN_ATTRS = {"system", "popen", "remove", "rmtree", "unlink", "write", "chmod"}


def assert_safe_python(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements are not allowed in generated validators")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"Forbidden symbol in generated code: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            raise ValueError(f"Forbidden attribute in generated code: {node.attr}")


def extension_allowed(ext: str, allowed: Iterable[str]) -> bool:
    return ext.lower() in {item.lower() for item in allowed}

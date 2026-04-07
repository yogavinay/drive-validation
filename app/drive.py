from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import gdown
import requests

from app.models import FileManifestItem


# Google often returns sign-in / challenge HTML to bare Python clients; use a normal browser UA.
_DRIVE_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

SUPPORTED_EXTENSIONS = {".jsonl", ".json", ".txt", ".py", ".npy", ".pt"}


def extract_drive_folder_id(url: str) -> Optional[str]:
    # Canonical folder links:
    # - https://drive.google.com/drive/folders/<FOLDER_ID>?usp=sharing
    # - https://drive.google.com/open?id=<FOLDER_ID>
    m = re.search(r"/drive/folders/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    # open?id=... can be folder or file; gdown will decide.
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def check_drive_access(url: str, timeout: int = 20) -> Tuple[bool, str]:
    """
    Public folder pages still contain links to accounts.google.com in the HTML (nav sign-in).
    Only treat as blocked when the *response URL* is a Google auth wall, not when the body merely links to sign-in.
    """
    response = requests.get(
        url, timeout=timeout, allow_redirects=True, headers=_DRIVE_REQUEST_HEADERS
    )
    if response.status_code >= 400:
        return False, f"HTTP {response.status_code} while accessing Drive link"
    final = (response.url or "").lower()
    if "accounts.google.com" in final or "accounts.youtube.com" in final:
        return False, "Redirected to Google sign-in; folder is not reachable without login"
    if "consent.google.com" in final:
        return False, "Google consent page; folder may not be publicly accessible"
    body = response.text.lower()
    if "request access" in body or "you need access" in body:
        return False, "Drive link requires access permission"
    # Hard blocks (actual challenge pages), not normal Drive HTML that mentions captcha in scripts.
    if "unusual traffic" in body and "sorry" in body:
        return False, "Drive link is blocked by Google verification/captcha"
    if "verify you are a human" in body:
        return False, "Drive link is blocked by Google verification/captcha"
    return True, ""


def check_drive_folder_is_folder(folder_id: str, api_key: str) -> Tuple[str, str]:
    """
    Returns (kind, detail):
    - ("folder", mimeType) — confirmed folder
    - ("not_folder", mimeType) — confirmed not a folder
    - ("unknown", reason) — API unavailable (e.g. key not enabled); caller should not block
    """
    try:
        resp = requests.get(
            f"https://www.googleapis.com/drive/v3/files/{folder_id}",
            params={
                "key": api_key,
                "fields": "id,mimeType",
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            },
            timeout=60,
        )
        if resp.status_code >= 400:
            err = resp.text[:500]
            # Do not block pipeline when API key project has Drive API disabled / wrong key.
            if resp.status_code in (403, 401):
                return "unknown", f"Drive API error {resp.status_code}: {err}"
            return "unknown", f"Drive API error {resp.status_code}: {err}"
        data = resp.json() or {}
        mime_type = data.get("mimeType") or ""
        if mime_type == "application/vnd.google-apps.folder":
            return "folder", mime_type
        if mime_type:
            return "not_folder", mime_type
        return "unknown", "Empty mimeType from Drive API"
    except Exception as exc:
        return "unknown", str(exc)


def download_drive_folder(url: str, staging_root: str, job_id: str) -> Tuple[List[FileManifestItem], List[str]]:
    target = Path(staging_root) / job_id
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    folder_id = extract_drive_folder_id(url)
    last_exc: Optional[BaseException] = None
    downloaded = None
    for attempt in range(3):
        try:
            if folder_id:
                # Use parsed folder id to avoid URL-parsing failures in gdown.
                downloaded = gdown.download_folder(
                    id=folder_id,
                    output=str(target),
                    quiet=True,
                    remaining_ok=True,
                )
            else:
                downloaded = gdown.download_folder(
                    url=url,
                    output=str(target),
                    quiet=True,
                    remaining_ok=True,
                )
            break
        except Exception as exc:
            last_exc = exc
            time.sleep(2**attempt)
    if downloaded is None:
        # Fallback: use Google Drive API listing/downloading (works for public/shared items).
        # This avoids gdown's folder parsing/listing failures.
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"Failed to download Drive folder via gdown: {last_exc}. Also GOOGLE_API_KEY is not set."
            )
        try:
            manifest, skipped = _download_drive_folder_via_drive_api(
                folder_id=folder_id,
                api_key=api_key,
                root_dir=target,
                job_id=job_id,
            )
            return manifest, skipped
        except Exception as api_exc:
            raise RuntimeError(
                f"Failed to download Drive folder via gdown: {last_exc}. "
                f"Drive API fallback also failed: {api_exc}"
            )
    if downloaded is None:
        downloaded = []

    manifest: List[FileManifestItem] = []
    skipped: List[str] = []
    for path_str in downloaded:
        path = Path(path_str)
        if not path.exists():
            continue
        rel = path.relative_to(target).as_posix()
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            skipped.append(rel)
            manifest.append(
                FileManifestItem(
                    relative_path=rel,
                    absolute_path=str(path.resolve()),
                    size_bytes=path.stat().st_size,
                    extension=ext,
                    download_status="SKIPPED_UNSUPPORTED",
                    notes="Unsupported file extension",
                )
            )
            continue
        manifest.append(
            FileManifestItem(
                relative_path=rel,
                absolute_path=str(path.resolve()),
                size_bytes=path.stat().st_size,
                extension=ext,
                download_status="DOWNLOADED",
            )
        )
    return manifest, skipped


def _drive_api_get(
    url: str,
    api_key: str,
    params: Optional[Dict[str, str]] = None,
    max_attempts: int = 3,
) -> Dict:
    params = params or {}
    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            r = requests.get(url, params={**params, "key": api_key}, timeout=60)
            if r.status_code >= 400:
                raise RuntimeError(f"Drive API error {r.status_code}: {r.text[:300]}")
            return r.json()
        except Exception as exc:
            last_exc = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Drive API request failed after retries: {last_exc}")


def _drive_api_list_children(folder_id: str, api_key: str) -> List[Dict[str, str]]:
    # Lists immediate children (files + folders) within a folder.
    base = "https://www.googleapis.com/drive/v3/files"
    q = f"'{folder_id}' in parents and trashed=false"
    page_token: Optional[str] = None
    items: List[Dict[str, str]] = []
    while True:
        params: Dict[str, str] = {
            "q": q,
            "fields": "nextPageToken,files(id,name,mimeType,size)",
            "pageSize": "200",
            "includeItemsFromAllDrives": "true",
            "supportsAllDrives": "true",
            "orderBy": "folder,name",
        }
        if page_token:
            params["pageToken"] = page_token
        resp = _drive_api_get(base, api_key=api_key, params=params)
        items.extend(resp.get("files", []) or [])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def _drive_api_download_file(file_id: str, api_key: str, dest: Path) -> int:
    # Downloads file content streaming to avoid loading large files into memory.
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    params = {
        "alt": "media",
        "supportsAllDrives": "true",
    }
    last_exc: Optional[BaseException] = None
    for attempt in range(3):
        try:
            with requests.get(
                url, params={**params, "key": api_key}, stream=True, timeout=120
            ) as r:
                if r.status_code >= 400:
                    raise RuntimeError(f"Download failed {r.status_code}: {r.text[:300]}")
                total = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        total += len(chunk)
                return total
        except Exception as exc:
            last_exc = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Drive API download failed after retries: {last_exc}")


def _download_drive_folder_via_drive_api(
    folder_id: Optional[str],
    api_key: str,
    root_dir: Path,
    job_id: str,
) -> Tuple[List[FileManifestItem], List[str]]:
    if not folder_id:
        raise RuntimeError("Could not extract Drive folder id from the provided URL.")

    manifest: List[FileManifestItem] = []
    skipped: List[str] = []

    def is_folder(item: Dict[str, str]) -> bool:
        return item.get("mimeType") == "application/vnd.google-apps.folder"

    def sanitize_name(name: str) -> str:
        # Prevent path traversal / invalid filesystem names.
        return re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name).strip() or "unnamed"

    def walk(current_folder_id: str, rel_dir: str) -> None:
        children = _drive_api_list_children(current_folder_id, api_key=api_key)
        for child in children:
            child_name = sanitize_name(child.get("name") or "unnamed")
            child_rel = f"{rel_dir}/{child_name}".strip("/")
            if is_folder(child):
                walk(child["id"], child_rel)
                continue

            ext = Path(child_name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                skipped.append(child_rel)
                manifest.append(
                    FileManifestItem(
                        relative_path=child_rel,
                        absolute_path=str((root_dir / child_rel).resolve()),
                        size_bytes=int(child.get("size") or 0),
                        extension=ext,
                        download_status="SKIPPED_UNSUPPORTED",
                        notes="Unsupported file extension",
                    )
                )
                continue

            dest = root_dir / child_rel
            size = _drive_api_download_file(child["id"], api_key=api_key, dest=dest)
            manifest.append(
                FileManifestItem(
                    relative_path=child_rel,
                    absolute_path=str(dest.resolve()),
                    size_bytes=size,
                    extension=ext,
                    download_status="DOWNLOADED",
                )
            )

    walk(folder_id, rel_dir="")
    return manifest, skipped

from __future__ import annotations

import os
from typing import Any, Dict, List

import requests


def _firebase_db_url() -> str | None:
    url = os.getenv("FIREBASE_DATABASE_URL", "").strip()
    return url.rstrip("/") if url else None


def _enabled() -> bool:
    return bool(_firebase_db_url())


def _with_auth_params() -> Dict[str, str]:
    token = os.getenv("FIREBASE_DB_SECRET", "").strip()
    return {"auth": token} if token else {}


def save_report_to_firebase(job_id: str, report: Dict[str, Any], rule_sets: Dict[str, List[str]]) -> None:
    if not _enabled():
        return
    base = _firebase_db_url()
    if not base:
        return
    payload = {
        "job_id": job_id,
        "report_id": report.get("report_id"),
        "generated_at": report.get("generated_at"),
        "drive_url": report.get("drive_url"),
        "owner": "TEAM",
        "summary": report.get("summary", {}),
        "set_names": sorted(rule_sets.keys()),
        "report": report,
    }
    requests.put(
        f"{base}/reports/{job_id}.json",
        params=_with_auth_params(),
        json=payload,
        timeout=30,
    )


def list_reports_from_firebase(limit: int = 100) -> List[Dict[str, Any]]:
    if not _enabled():
        return []
    base = _firebase_db_url()
    if not base:
        return []
    resp = requests.get(
        f"{base}/reports.json",
        params=_with_auth_params(),
        timeout=30,
    )
    if resp.status_code >= 400:
        return []
    data = resp.json() or {}
    if not isinstance(data, dict):
        return []
    rows = list(data.values())
    rows.sort(key=lambda x: x.get("generated_at", ""), reverse=True)
    return rows[:limit]


def get_report_from_firebase(job_id: str) -> Dict[str, Any]:
    if not _enabled():
        return {}
    base = _firebase_db_url()
    if not base:
        return {}
    resp = requests.get(
        f"{base}/reports/{job_id}.json",
        params=_with_auth_params(),
        timeout=30,
    )
    if resp.status_code >= 400:
        return {}
    return resp.json() or {}


def delete_report_from_firebase(job_id: str) -> bool:
    if not _enabled():
        return False
    base = _firebase_db_url()
    if not base:
        return False
    resp = requests.delete(
        f"{base}/reports/{job_id}.json",
        params=_with_auth_params(),
        timeout=30,
    )
    return resp.status_code < 400

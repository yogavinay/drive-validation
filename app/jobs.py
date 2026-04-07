from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import orjson


def jobs_root() -> Path:
    root = Path(os.getenv("VALIDATION_JOBS_DIR", "/tmp/staging/jobs"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def job_dir(job_id: str) -> Path:
    path = jobs_root() / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_job_status(job_id: str, status: Dict[str, Any]) -> None:
    path = job_dir(job_id) / "status.json"
    with open(path, "wb") as handle:
        handle.write(orjson.dumps(status, option=orjson.OPT_INDENT_2))


def read_job_status(job_id: str) -> Dict[str, Any]:
    path = job_dir(job_id) / "status.json"
    if not path.exists():
        return {"job_id": job_id, "pipeline_status": "UNKNOWN"}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_job_report(job_id: str, report: Dict[str, Any]) -> None:
    path = job_dir(job_id) / "report.json"
    with open(path, "wb") as handle:
        handle.write(orjson.dumps(report, option=orjson.OPT_INDENT_2))


def read_job_report(job_id: str) -> Dict[str, Any]:
    path = job_dir(job_id) / "report.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def update_job_progress(job_id: str, current_agent: str, detail: str, pipeline_status: str = "RUNNING") -> None:
    """Emit sub-step progress so the UI does not look stuck during long LLM or I/O."""
    write_job_status(
        job_id,
        {
            "job_id": job_id,
            "current_agent": current_agent,
            "pipeline_status": pipeline_status,
            "detail": detail,
        },
    )
    append_job_log(job_id, f"{current_agent}: {detail}")


def append_job_log(job_id: str, message: str) -> None:
    path = job_dir(job_id) / "events.log"
    ts = datetime.utcnow().isoformat()
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{ts} {message}\n")


def read_job_log(job_id: str) -> str:
    path = job_dir(job_id) / "events.log"
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()

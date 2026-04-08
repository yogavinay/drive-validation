from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

# Load project-root .env before any code reads os.environ
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Deque, Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from app.firebase_store import (
    delete_report_from_firebase,
    get_report_from_firebase,
    list_reports_from_firebase,
    save_report_to_firebase,
)
from app.graph_pipeline import run_pipeline
from app.jobs import read_job_log, read_job_report, read_job_status, write_job_report, write_job_status
from app.models import ValidationRequest, ValidationState
from app.pdf_report import build_pdf


app = FastAPI(title="LangGraph Agentic Validation System", version="1.0.0")

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))


def _apply_fast_mode_policy(request: ValidationRequest) -> ValidationRequest:
    """
    Enforce fast validation on the server so old Streamlit/API clients cannot
    accidentally send fast_mode=false and trigger many slow LLM compilations.
    Set FORCE_FAST_MODE=false to allow full mode.
    """
    raw = os.getenv("FORCE_FAST_MODE", "true").strip().lower()
    if raw in ("0", "false", "no"):
        return request
    mrs = request.max_rules_per_set
    mrf = request.max_records_per_file
    if mrs is None:
        mrs = int(os.getenv("FAST_MAX_RULES_PER_SET", "2"))
    if mrf is None:
        mrf = int(os.getenv("FAST_MAX_RECORDS_PER_FILE", "5000"))
    return request.model_copy(
        update={
            "fast_mode": True,
            "max_rules_per_set": mrs,
            "max_records_per_file": mrf,
        },
    )


@app.on_event("startup")
async def startup_event():
    app.state.request_windows = defaultdict(deque)
    if not os.getenv("FIREBASE_DATABASE_URL", "").strip():
        print("WARN: FIREBASE_DATABASE_URL is not set; Firebase report history is disabled.")


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client = request.client.host if request.client else "unknown"
    now = datetime.utcnow()
    window = request.app.state.request_windows[client]
    while window and window[0] < now - timedelta(minutes=1):
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MIN:
        return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
    window.append(now)
    return await call_next(request)


async def _run_job(job_id: str, payload: ValidationRequest):
    state: ValidationState = {
        "drive_url": payload.drive_url,
        "access_status": "UNKNOWN",
        "downloaded_files": [],
        "rule_sets": payload.rule_sets,
        "file_rule_mapping": {},
        "generated_validators": {},
        "execution_results": {},
        "format_analysis": {},
        "final_report": {},
        "errors": [],
        "current_agent": "INIT",
        "pipeline_status": "RUNNING",
        "llm_provider": payload.llm_provider,
        "llm_model": payload.llm_model,
        "job_id": job_id,
        "fast_mode": payload.fast_mode,
        "max_rules_per_set": payload.max_rules_per_set,
        "max_records_per_file": payload.max_records_per_file,
    }
    write_job_status(job_id, {"job_id": job_id, "pipeline_status": "RUNNING", "current_agent": "INIT"})
    final_state = await asyncio.to_thread(run_pipeline, state)
    report = final_state.get("final_report", {})
    if not report:
        report = {
            "report_id": str(uuid4()),
            "generated_at": datetime.utcnow().isoformat(),
            "drive_url": payload.drive_url,
            "summary": {
                "total_files": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "overall_pass_rate": "0%",
                "final_verdict": "FAILED",
            },
            "files": [],
            "errors": [err.model_dump() if hasattr(err, "model_dump") else err for err in final_state.get("errors", [])],
        }
    write_job_report(job_id, report)
    try:
        save_report_to_firebase(job_id=job_id, report=report, rule_sets=payload.rule_sets)
    except Exception:
        # Non-fatal: local report is still persisted even if remote storage fails.
        pass
    write_job_status(
        job_id,
        {
            "job_id": job_id,
            "pipeline_status": final_state.get("pipeline_status", "FAILED"),
            "current_agent": final_state.get("current_agent", "DONE"),
        },
    )


@app.post("/validate")
async def validate(request: ValidationRequest):
    if not request.rule_sets:
        raise HTTPException(status_code=400, detail="At least one rule set is required")
    request = _apply_fast_mode_policy(request)
    job_id = str(uuid4())
    # Persist an immediate status so /status never appears as UNKNOWN.
    write_job_status(
        job_id,
        {"job_id": job_id, "pipeline_status": "RUNNING", "current_agent": "INIT"},
    )
    # Render can be flaky with response-bound background tasks; schedule directly.
    asyncio.create_task(_run_job(job_id, request))
    return {"job_id": job_id, "status": "RUNNING"}


@app.get("/status/{job_id}")
async def status(job_id: str):
    state = read_job_status(job_id)
    state["logs"] = read_job_log(job_id)
    return state


@app.get("/report/{job_id}")
async def report(job_id: str, format: str = "json"):
    data = read_job_report(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Report not ready")
    if format == "pdf":
        pdf = build_pdf(data)
        return Response(content=pdf, media_type="application/pdf")
    return data


@app.get("/reports/firebase")
async def list_firebase_reports(limit: int = 100):
    return {"items": list_reports_from_firebase(limit=limit)}


@app.get("/reports/firebase/{job_id}")
async def get_firebase_report(job_id: str):
    item = get_report_from_firebase(job_id)
    if not item:
        raise HTTPException(status_code=404, detail="Firebase report not found")
    return item


@app.delete("/reports/firebase/{job_id}")
async def delete_firebase_report(job_id: str):
    ok = delete_report_from_firebase(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Firebase report not found or delete failed")
    return {"status": "deleted", "job_id": job_id}

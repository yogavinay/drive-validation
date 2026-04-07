from __future__ import annotations

from datetime import datetime
from typing import Dict
from uuid import uuid4

from app.models import FileValidationResult


def build_report(
    drive_url: str,
    execution_results: Dict[str, FileValidationResult],
    errors: list,
) -> Dict:
    files = list(execution_results.values())
    passed = len([item for item in files if item.status == "PASSED"])
    failed = len([item for item in files if item.status in {"FAILED", "INVALID"}])
    skipped = len([item for item in files if item.status == "SKIPPED"])
    total = len(files)
    pass_rate = f"{(passed / total * 100):.2f}%" if total else "0%"
    verdict = "PASSED" if failed == 0 else "FAILED"
    return {
        "report_id": str(uuid4()),
        "generated_at": datetime.utcnow().isoformat(),
        "drive_url": drive_url,
        "summary": {
            "total_files": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "overall_pass_rate": pass_rate,
            "final_verdict": verdict,
        },
        "files": [item.model_dump() for item in files],
        "errors": [err.model_dump() if hasattr(err, "model_dump") else err for err in errors],
    }

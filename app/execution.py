from __future__ import annotations

import ast
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

from app.models import FileManifestItem, FileValidationResult, RuleEvaluation


def _rule_applicable(rule_text: str, extension: str) -> bool:
    r = rule_text.lower()
    ext = extension.lower()
    if "shape" in r and "1024" in r:
        return ext == ".npy"
    if "state dict" in r or "tensor" in r:
        return ext == ".pt"
    if "syntax" in r or "dangerous calls" in r or "ast" in r:
        return ext == ".py"
    return True


def _iter_records(item: FileManifestItem, max_records: int | None = None):
    path = Path(item.absolute_path)
    ext = item.extension.lower()
    if ext in {".txt", ".jsonl"}:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for idx, line in enumerate(handle, start=1):
                if max_records is not None and idx > max_records:
                    break
                yield idx, line.rstrip("\n")
    elif ext == ".json":
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            for idx, entry in enumerate(payload, start=1):
                if max_records is not None and idx > max_records:
                    break
                yield idx, entry
        else:
            yield 1, payload
    elif ext == ".npy":
        arr = np.load(path, mmap_mode="r")
        yield 1, {"shape": list(arr.shape), "dtype": str(arr.dtype), "has_nan": bool(np.isnan(arr).any())}
    elif ext == ".pt":
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            yield 1, {"keys": list(obj.keys())[:100], "type": "state_dict"}
        else:
            yield 1, {"type": str(type(obj))}
    elif ext == ".py":
        content = path.read_text(encoding="utf-8", errors="replace")
        ast.parse(content)
        yield 1, content
    else:
        return


def _run_rules(
    item: FileManifestItem,
    validators: List[Callable[..., Dict[str, Any]]],
    rule_texts: List[str],
    max_records: int | None,
) -> FileValidationResult:
    result = FileValidationResult(
        file_name=item.relative_path,
        file_type=item.extension.lstrip("."),
        matched_rule_set="UNASSIGNED",
        status="PASSED",
    )
    if item.size_bytes == 0:
        result.status = "FAILED"
        result.rule_evaluations.append(
            RuleEvaluation(
                rule="FILE NOT EMPTY",
                status="FAILED",
                failed_lines=[1],
                failure_count=1,
                total_checked=1,
                details="Empty file",
            )
        )
        return result

    for idx, validator in enumerate(validators):
        text = rule_texts[idx] if idx < len(rule_texts) else f"RULE_{idx+1}"
        if not _rule_applicable(text, item.extension):
            result.rule_evaluations.append(
                RuleEvaluation(
                    rule=text,
                    status="SKIPPED",
                    failed_lines=[],
                    failure_count=0,
                    total_checked=0,
                    details=f"Rule not applicable for file type {item.extension}",
                )
            )
            continue
        failed_lines: List[int] = []
        checked = 0
        details = ""
        try:
            for line_no, record in _iter_records(item, max_records=max_records):
                checked += 1
                outcome = validator(record, line_no, {"file_type": item.extension})
                if not outcome.get("passed", True):
                    failed_lines.extend(outcome.get("failed_lines") or [line_no])
                    details = outcome.get("details", details)
        except Exception as exc:
            result.status = "FAILED"
            result.rule_evaluations.append(
                RuleEvaluation(
                    rule=text,
                    status="FAILED",
                    failed_lines=[],
                    failure_count=1,
                    total_checked=checked,
                    details=f"Execution error: {exc}",
                )
            )
            continue

        status = "FAILED" if failed_lines else "PASSED"
        if status == "FAILED":
            result.status = "FAILED"
        result.rule_evaluations.append(
            RuleEvaluation(
                rule=text,
                status=status,
                failed_lines=sorted(set(failed_lines)),
                failure_count=len(failed_lines),
                total_checked=checked,
                details=details,
            )
        )
    return result


def execute_validations_parallel(
    files: List[FileManifestItem],
    mapping: Dict[str, str],
    validators_by_set: Dict[str, List[Callable[..., Dict[str, Any]]]],
    rules_by_set: Dict[str, List[str]],
    max_records_per_file: int | None = None,
    max_workers: int = 8,
) -> Dict[str, FileValidationResult]:
    output: Dict[str, FileValidationResult] = {}
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for item in files:
            rule_set = mapping.get(item.relative_path)
            if not rule_set or rule_set == "NO_RULE_ASSIGNED":
                output[item.relative_path] = FileValidationResult(
                    file_name=item.relative_path,
                    file_type=item.extension.lstrip("."),
                    matched_rule_set="NO_RULE_ASSIGNED",
                    status="SKIPPED",
                    rule_evaluations=[],
                )
                continue
            futures.append(
                pool.submit(
                    _run_rules,
                    item,
                    validators_by_set.get(rule_set, []),
                    rules_by_set.get(rule_set, []),
                    max_records_per_file,
                )
            )
        for future in as_completed(futures):
            result = future.result()
            output[result.file_name] = result
    for path, rule_set in mapping.items():
        if path in output:
            output[path].matched_rule_set = rule_set
    return output

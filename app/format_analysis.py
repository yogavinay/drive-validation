from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from app.models import FileManifestItem, FormatAnalysisResult


def analyze_file_format(item: FileManifestItem) -> FormatAnalysisResult:
    path = Path(item.absolute_path)
    ext = item.extension.lower()
    result = FormatAnalysisResult()
    if not path.exists():
        result.anomalies.append("File missing after download")
        return result
    if path.stat().st_size == 0:
        result.anomalies.append("Empty file")
        return result

    if ext in {".txt", ".jsonl", ".json", ".py"}:
        with open(path, "rb") as handle:
            raw = handle.read()
        result.has_bom = raw.startswith(b"\xef\xbb\xbf")
        text = raw.decode("utf-8", errors="replace")
        result.encoding = "UTF-8"
        if "\r\n" in text and "\n" in text:
            result.line_endings = "MIXED"
        elif "\r\n" in text:
            result.line_endings = "CRLF"
        else:
            result.line_endings = "LF"

    if ext == ".jsonl":
        keys = None
        total = 0
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        current = set(obj.keys())
                        if keys is None:
                            keys = current
                        elif current != keys:
                            result.schema_consistent = False
                except json.JSONDecodeError:
                    result.anomalies.append(f"Malformed JSON at line {total}")
        result.total_records = total
        if result.schema_consistent is None:
            result.schema_consistent = True
    elif ext == ".json":
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            payload = json.load(handle)
        result.total_records = len(payload) if isinstance(payload, list) else 1
        result.schema_consistent = True
    elif ext == ".npy":
        arr = np.load(path, mmap_mode="r")
        result.metadata["shape"] = list(arr.shape)
        result.metadata["dtype"] = str(arr.dtype)
        if np.isnan(arr).any():
            result.anomalies.append("Contains NaN values")
        if np.isinf(arr).any():
            result.anomalies.append("Contains Inf values")
    elif ext == ".pt":
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            result.metadata["state_dict_keys"] = list(obj.keys())[:100]
        result.metadata["object_type"] = str(type(obj))
    elif ext == ".py":
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            result.anomalies.append(
                f"Python syntax error at line {getattr(exc, 'lineno', '?')}: {exc.msg}"
            )
            result.metadata["syntax_ok"] = False
            return result
        imports: List[str] = []
        functions: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        result.metadata["imports"] = imports
        result.metadata["functions"] = functions
        result.metadata["syntax_ok"] = True
    return result


def analyze_all_formats(files: List[FileManifestItem]) -> Dict[str, FormatAnalysisResult]:
    return {item.relative_path: analyze_file_format(item) for item in files}

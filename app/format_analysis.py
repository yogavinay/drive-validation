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
    try:
        if not path.exists():
            result.anomalies.append("File missing after download")
            return result
        if path.stat().st_size == 0:
            result.anomalies.append("Empty file")
            return result

        # ── Text-based files: encoding / BOM / line-endings ──
        _text_exts = {
            ".txt", ".jsonl", ".json", ".py", ".csv", ".tsv", ".xml", ".yaml", ".yml",
            ".md", ".markdown", ".html", ".htm", ".css", ".scss", ".less",
            ".log", ".out", ".err", ".sh", ".bat", ".ps1", ".sql", ".lua", ".pl", ".php",
            ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h", ".cs", ".go",
            ".rs", ".rb", ".swift", ".kt", ".r", ".scala",
            ".toml", ".ini", ".cfg", ".env", ".conf", ".properties", ".svg",
        }
        if ext in _text_exts:
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

        # ── JSONL ──
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
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    payload = json.load(handle)
                result.total_records = len(payload) if isinstance(payload, list) else 1
                result.schema_consistent = True
            except json.JSONDecodeError as exc:
                result.schema_consistent = False
                result.anomalies.append(
                    f"Malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
                )
                return result

        elif ext in {".csv", ".tsv"}:
            import csv
            delimiter = "\t" if ext == ".tsv" else ","
            try:
                with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
                    reader = csv.reader(handle, delimiter=delimiter)
                    headers = None
                    total = 0
                    for row_idx, row in enumerate(reader, start=1):
                        if row_idx == 1:
                            headers = row
                            result.metadata["columns"] = headers
                            continue
                        total += 1
                        if headers and len(row) != len(headers):
                            result.anomalies.append(f"Row {row_idx} has {len(row)} columns, expected {len(headers)}")
                    result.total_records = total
                    result.schema_consistent = len(result.anomalies) == 0
            except Exception as exc:
                result.anomalies.append(f"CSV parse error: {exc}")

        elif ext in {".yaml", ".yml"}:
            try:
                import yaml
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    payload = yaml.safe_load(handle)
                result.total_records = len(payload) if isinstance(payload, list) else 1
                result.schema_consistent = True
                result.metadata["type"] = type(payload).__name__
            except Exception as exc:
                result.anomalies.append(f"YAML parse error: {exc}")
                result.schema_consistent = False

        elif ext == ".xml":
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(path)
                root = tree.getroot()
                result.metadata["root_tag"] = root.tag
                result.metadata["child_count"] = len(root)
                result.total_records = len(root)
                result.schema_consistent = True
            except Exception as exc:
                result.anomalies.append(f"XML parse error: {exc}")
                result.schema_consistent = False

        elif ext == ".parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(path)
                result.total_records = len(df)
                result.metadata["columns"] = list(df.columns)
                result.metadata["dtypes"] = {col: str(dt) for col, dt in df.dtypes.items()}
                result.schema_consistent = True
            except Exception as exc:
                result.anomalies.append(f"Parquet read error: {exc}")

        elif ext == ".npy":
            arr = np.load(path, mmap_mode="r")
            result.metadata["shape"] = list(arr.shape)
            result.metadata["dtype"] = str(arr.dtype)
            if np.isnan(arr).any():
                result.anomalies.append("Contains NaN values")
            if np.isinf(arr).any():
                result.anomalies.append("Contains Inf values")

        elif ext in {".pt", ".pth"}:
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

        elif ext in {".md", ".markdown", ".html", ".htm", ".log", ".out", ".err"}:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                content = handle.read()
            result.total_records = content.count("\n") + 1
            result.metadata["char_count"] = len(content)
            result.metadata["word_count"] = len(content.split())

        else:
            # Binary or unknown — just report size
            stat = path.stat()
            result.metadata["size_bytes"] = stat.st_size
            result.metadata["type"] = ext.lstrip(".")

        return result
    except Exception as exc:
        result.anomalies.append(f"Format analysis error: {exc}")
        return result


def analyze_all_formats(files: List[FileManifestItem]) -> Dict[str, FormatAnalysisResult]:
    return {item.relative_path: analyze_file_format(item) for item in files}

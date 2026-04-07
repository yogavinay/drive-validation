from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class FileManifestItem(BaseModel):
    relative_path: str
    absolute_path: str
    size_bytes: int
    extension: str
    file_type: Optional[str] = None
    download_status: Literal["DOWNLOADED", "FAILED", "SKIPPED_UNSUPPORTED"] = "DOWNLOADED"
    notes: Optional[str] = None


class RuleEvaluation(BaseModel):
    rule: str
    status: Literal["PASSED", "FAILED", "SKIPPED", "INVALID"] = "PASSED"
    failed_lines: List[int] = Field(default_factory=list)
    failure_count: int = 0
    total_checked: int = 0
    details: str = ""


class FormatAnalysisResult(BaseModel):
    schema_consistent: Optional[bool] = None
    encoding: Optional[str] = None
    line_endings: Optional[str] = None
    has_bom: Optional[bool] = None
    total_records: int = 0
    missing_keys: List[str] = Field(default_factory=list)
    anomalies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FileValidationResult(BaseModel):
    file_name: str
    file_type: str
    matched_rule_set: str
    status: Literal["PASSED", "FAILED", "SKIPPED", "INVALID"] = "SKIPPED"
    rule_evaluations: List[RuleEvaluation] = Field(default_factory=list)
    format_analysis: Optional[FormatAnalysisResult] = None


class ErrorEntry(BaseModel):
    agent: str
    message: str
    traceback: Optional[str] = None
    file: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationReport(BaseModel):
    report_id: str
    generated_at: str
    drive_url: str
    summary: Dict[str, Any]
    files: List[FileValidationResult]
    errors: List[Dict[str, Any]]


class ValidationRequest(BaseModel):
    drive_url: str
    rule_sets: Dict[str, List[str]]
    llm_provider: Literal["openai", "anthropic", "gemini", "nvidia"] = "openai"
    llm_model: str = "gpt-4o"
    # Fast mode: approximate / quick validation
    fast_mode: bool = False
    max_rules_per_set: Optional[int] = None
    max_records_per_file: Optional[int] = None


class ValidationState(TypedDict):
    drive_url: str
    access_status: str
    downloaded_files: List[FileManifestItem]
    rule_sets: Dict[str, List[str]]
    file_rule_mapping: Dict[str, str]
    generated_validators: Dict[str, List[Callable[..., Dict[str, Any]]]]
    execution_results: Dict[str, FileValidationResult]
    format_analysis: Dict[str, FormatAnalysisResult]
    final_report: Dict[str, Any]
    errors: List[ErrorEntry]
    current_agent: str
    pipeline_status: str
    llm_provider: str
    llm_model: str
    job_id: str
    fast_mode: bool
    max_rules_per_set: Optional[int]
    max_records_per_file: Optional[int]

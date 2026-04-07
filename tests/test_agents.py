from __future__ import annotations

from pathlib import Path

from app.models import FileManifestItem
from app.graph_pipeline import (
    access_validator_agent,
    code_generation_agent,
    drive_fetch_agent,
    execution_agent,
    file_classifier_agent,
    format_analysis_agent,
    report_generator_agent,
    rule_input_system,
    rule_matching_agent,
)


def base_state():
    return {
        "drive_url": "https://drive.google.com/drive/folders/abc",
        "access_status": "UNKNOWN",
        "downloaded_files": [],
        "rule_sets": {"A": ["No empty lines"]},
        "file_rule_mapping": {},
        "generated_validators": {},
        "execution_results": {},
        "format_analysis": {},
        "final_report": {},
        "errors": [],
        "current_agent": "INIT",
        "pipeline_status": "RUNNING",
        "llm_provider": "openai",
        "llm_model": "gpt-4o",
        "job_id": "test-job",
    }


def test_rule_input_system_passthrough():
    state = base_state()
    out = rule_input_system(state)
    assert out["rule_sets"]["A"] == ["No empty lines"]


def test_rule_matching_agent_no_files():
    state = base_state()
    out = rule_matching_agent(state)
    assert out["file_rule_mapping"] == {}


def test_access_validator_agent_blocked(monkeypatch):
    state = base_state()
    monkeypatch.setattr("app.graph_pipeline.validate_drive_url", lambda _: (True, ""))
    monkeypatch.setattr("app.graph_pipeline.check_drive_access", lambda _: (False, "blocked"))
    out = access_validator_agent(state)
    assert out["access_status"] == "BLOCKED"


def test_drive_fetch_agent(monkeypatch):
    state = base_state()
    fake_item = FileManifestItem(
        relative_path="A.jsonl",
        absolute_path="/tmp/staging/test/A.jsonl",
        size_bytes=12,
        extension=".jsonl",
    )
    monkeypatch.setattr("app.graph_pipeline.download_drive_folder", lambda *_: ([fake_item], []))
    out = drive_fetch_agent(state)
    assert len(out["downloaded_files"]) == 1


def test_file_classifier_agent():
    state = base_state()
    state["downloaded_files"] = [
        FileManifestItem(relative_path="A.jsonl", absolute_path="/tmp/A.jsonl", size_bytes=1, extension=".jsonl")
    ]
    out = file_classifier_agent(state)
    assert out["downloaded_files"][0].file_type == "jsonl"


def test_rule_matching_agent_match():
    state = base_state()
    state["downloaded_files"] = [
        FileManifestItem(relative_path="A_train.jsonl", absolute_path="/tmp/A_train.jsonl", size_bytes=1, extension=".jsonl")
    ]
    out = rule_matching_agent(state)
    assert out["file_rule_mapping"]["A_train.jsonl"] == "A"


def test_code_generation_agent(monkeypatch):
    state = base_state()
    monkeypatch.setattr(
        "app.graph_pipeline.generate_validators",
        lambda *_, **__: {"A": [lambda record, line_number, context: {"passed": True, "failed_lines": [], "details": ""}]},
    )
    out = code_generation_agent(state)
    assert "A" in out["generated_validators"]


def test_execution_agent():
    state = base_state()
    state["downloaded_files"] = [
        FileManifestItem(relative_path="A.jsonl", absolute_path="/tmp/A.jsonl", size_bytes=0, extension=".jsonl")
    ]
    state["file_rule_mapping"] = {"A.jsonl": "A"}
    state["generated_validators"] = {"A": [lambda *_: {"passed": True, "failed_lines": [], "details": ""}]}
    out = execution_agent(state)
    assert out["execution_results"]["A.jsonl"].status in {"FAILED", "INVALID", "PASSED"}


def test_format_analysis_agent():
    state = base_state()
    state["downloaded_files"] = [
        FileManifestItem(relative_path="missing.txt", absolute_path="/tmp/does_not_exist.txt", size_bytes=1, extension=".txt")
    ]
    out = format_analysis_agent(state)
    assert "missing.txt" in out["format_analysis"]


def test_report_generator_agent():
    state = base_state()
    out = report_generator_agent(state)
    assert out["pipeline_status"] == "COMPLETED"
    assert "summary" in out["final_report"]

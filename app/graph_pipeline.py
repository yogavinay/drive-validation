from __future__ import annotations

import re
import traceback
import os
from pathlib import Path
from typing import Callable, Dict, List

from langgraph.graph import END, START, StateGraph

from app.drive import SUPPORTED_EXTENSIONS, check_drive_access, download_drive_folder
from app.drive import check_drive_folder_is_folder, extract_drive_folder_id
from app.execution import execute_validations_parallel
from app.format_analysis import analyze_all_formats
from app.jobs import append_job_log, write_job_status
from app.logging_config import get_logger
from app.models import ErrorEntry, ValidationState
from app.reporting import build_report
from app.rules import generate_validators
from app.security import validate_drive_url


logger = get_logger()


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: Dict[str, Callable[[ValidationState], ValidationState]] = {}

    def node(self, name: str):
        def decorator(fn):
            self.nodes[name] = fn
            return fn

        return decorator


graph = GraphBuilder()


def _match_rule_set(relative_path: str, rule_keys: List[str]) -> str:
    p = Path(relative_path)
    parents = [part.upper() for part in p.parts[:-1]]
    stem = p.stem.upper()
    name = p.name.upper()
    suffix = p.suffix.lower()

    if suffix in {".md", ".markdown"} or name.startswith("README"):
        return "NO_RULE_ASSIGNED"

    # 1) Exact filename stem match (A.jsonl -> A)
    for key in rule_keys:
        ku = key.upper()
        if stem == ku:
            return key

    # 2) Parent folder name match (B/anything -> B)
    for key in rule_keys:
        ku = key.upper()
        if ku in parents:
            return key

    # 3) Filename prefix match (SFT_RULES_train.jsonl -> SFT_RULES)
    for key in rule_keys:
        ku = key.upper()
        if stem.startswith(ku + "_") or stem.startswith(ku + "-") or stem == ku:
            return key
    return "NO_RULE_ASSIGNED"


def _update(state: ValidationState, agent: str, status: str = "RUNNING") -> None:
    state["current_agent"] = agent
    state["pipeline_status"] = status
    write_job_status(
        state["job_id"],
        {
            "job_id": state["job_id"],
            "current_agent": agent,
            "pipeline_status": status,
        },
    )
    append_job_log(state["job_id"], f"{agent} {status}")


@graph.node("AccessValidatorAgent")
def access_validator_agent(state: ValidationState) -> ValidationState:
    _update(state, "AccessValidatorAgent")
    ok, reason = validate_drive_url(state["drive_url"])
    if not ok:
        state["access_status"] = "BLOCKED"
        state["pipeline_status"] = "FAILED"
        state["errors"].append(
            {
                "status": "BLOCKED",
                "reason": reason,
                "action": "Please share a Google Drive folder link with 'Anyone with the link' and retry",
                "agent": "AccessValidatorAgent",
            }
        )
        state["errors"].append(
            ErrorEntry(
                agent="AccessValidatorAgent",
                message=reason,
            )
        )
        return state
    ok_access, reason_access = check_drive_access(state["drive_url"])
    if not ok_access:
        state["access_status"] = "BLOCKED"
        state["pipeline_status"] = "FAILED"
        state["errors"].append(
            {
                "status": "BLOCKED",
                "reason": reason_access,
                "action": "Make sure the Drive folder is publicly accessible (no sign-in/captcha) and retry",
                "agent": "AccessValidatorAgent",
            }
        )
        state["errors"].append(
            ErrorEntry(
                agent="AccessValidatorAgent",
                message=reason_access,
            )
        )
        return state

    api_key = os.getenv("GOOGLE_API_KEY")
    folder_id = extract_drive_folder_id(state["drive_url"])
    if api_key and folder_id:
        kind, detail = check_drive_folder_is_folder(folder_id=folder_id, api_key=api_key)
        if kind == "not_folder":
            state["access_status"] = "BLOCKED"
            state["pipeline_status"] = "FAILED"
            state["errors"].append(
                {
                    "status": "BLOCKED",
                    "reason": f"URL does not appear to be a folder link (mimeType={detail}).",
                    "action": "Paste a Google Drive *folder* share link (not a file link) and retry",
                    "agent": "AccessValidatorAgent",
                }
            )
            state["errors"].append(
                ErrorEntry(agent="AccessValidatorAgent", message=str(detail))
            )
            return state
        # kind "folder" or "unknown" (API disabled / cannot verify) — proceed
    state["access_status"] = "GRANTED"
    return state


@graph.node("DriveFetchAgent")
def drive_fetch_agent(state: ValidationState) -> ValidationState:
    _update(state, "DriveFetchAgent")
    manifest, skipped = download_drive_folder(state["drive_url"], "/tmp/staging", state["job_id"])
    state["downloaded_files"] = manifest
    append_job_log(state["job_id"], f"Downloaded {len(manifest)} files; unsupported={len(skipped)}")
    return state


@graph.node("FileClassifierAgent")
def file_classifier_agent(state: ValidationState) -> ValidationState:
    _update(state, "FileClassifierAgent")
    for item in state["downloaded_files"]:
        ext = item.extension.lower()
        item.file_type = ext.lstrip(".") if ext in SUPPORTED_EXTENSIONS else "UNSUPPORTED"
    return state


@graph.node("RuleInputSystem")
def rule_input_system(state: ValidationState) -> ValidationState:
    _update(state, "RuleInputSystem")
    return state


@graph.node("RuleMatchingAgent")
def rule_matching_agent(state: ValidationState) -> ValidationState:
    _update(state, "RuleMatchingAgent")
    mapping: Dict[str, str] = {}
    keys = [k.strip() for k in state["rule_sets"].keys() if k.strip()]
    for item in state["downloaded_files"]:
        mapping[item.relative_path] = _match_rule_set(item.relative_path, keys)
    state["file_rule_mapping"] = mapping
    return state


@graph.node("CodeGenerationAgent")
def code_generation_agent(state: ValidationState) -> ValidationState:
    _update(state, "CodeGenerationAgent")
    state["generated_validators"] = generate_validators(
        state["rule_sets"],
        provider=state["llm_provider"],
        model=state["llm_model"],
        job_id=state["job_id"],
        fast_mode=state.get("fast_mode", False),
        max_rules_per_set=state.get("max_rules_per_set"),
    )
    return state


@graph.node("ExecutionAgent")
def execution_agent(state: ValidationState) -> ValidationState:
    _update(state, "ExecutionAgent")
    state["execution_results"] = execute_validations_parallel(
        files=state["downloaded_files"],
        mapping=state["file_rule_mapping"],
        validators_by_set=state["generated_validators"],
        rules_by_set=state["rule_sets"],
        max_records_per_file=state.get("max_records_per_file"),
    )
    return state


@graph.node("FileFormatAnalysisAgent")
def format_analysis_agent(state: ValidationState) -> ValidationState:
    _update(state, "FileFormatAnalysisAgent")
    try:
        state["format_analysis"] = analyze_all_formats(state["downloaded_files"])
        for rel, analysis in state["format_analysis"].items():
            if rel in state["execution_results"]:
                state["execution_results"][rel].format_analysis = analysis
                if analysis.anomalies and state["execution_results"][rel].status == "PASSED":
                    state["execution_results"][rel].status = "FAILED"
    except Exception as exc:
        # Non-fatal: report and continue to final report generation.
        state["errors"].append(
            ErrorEntry(
                agent="FileFormatAnalysisAgent",
                message=f"Format analysis partially failed: {exc}",
            )
        )
    return state


@graph.node("ReportGeneratorAgent")
def report_generator_agent(state: ValidationState) -> ValidationState:
    _update(state, "ReportGeneratorAgent")
    state["final_report"] = build_report(
        drive_url=state["drive_url"],
        execution_results=state["execution_results"],
        errors=state["errors"],
    )
    state["pipeline_status"] = "COMPLETED"
    _update(state, "ReportGeneratorAgent", "COMPLETED")
    return state


def _route_after_access(state: ValidationState) -> str:
    return "halt" if state.get("access_status") == "BLOCKED" else "ok"


def build_langgraph():
    sg = StateGraph(ValidationState)
    for name, fn in graph.nodes.items():
        sg.add_node(name, fn)
    sg.add_edge(START, "AccessValidatorAgent")
    sg.add_conditional_edges("AccessValidatorAgent", _route_after_access, {"halt": END, "ok": "DriveFetchAgent"})
    sg.add_edge("DriveFetchAgent", "FileClassifierAgent")
    sg.add_edge("FileClassifierAgent", "RuleInputSystem")
    sg.add_edge("RuleInputSystem", "RuleMatchingAgent")
    sg.add_edge("RuleMatchingAgent", "CodeGenerationAgent")
    sg.add_edge("CodeGenerationAgent", "ExecutionAgent")
    sg.add_edge("ExecutionAgent", "FileFormatAnalysisAgent")
    sg.add_edge("FileFormatAnalysisAgent", "ReportGeneratorAgent")
    sg.add_edge("ReportGeneratorAgent", END)
    return sg.compile()


def _failing_agent_from_traceback(tb: str) -> str | None:
    m = re.search(r"During task with name '([^']+)'", tb)
    return m.group(1) if m else None


def run_pipeline(state: ValidationState) -> ValidationState:
    app = build_langgraph()
    try:
        result = app.invoke(state)
        return result
    except Exception as exc:
        tb = traceback.format_exc()
        failing = _failing_agent_from_traceback(tb)
        if failing:
            state["current_agent"] = failing
        state["pipeline_status"] = "FAILED"
        state["errors"].append(
            ErrorEntry(
                agent=failing or state.get("current_agent") or "UNKNOWN",
                message=str(exc),
                traceback=tb,
            )
        )
        if state.get("job_id"):
            write_job_status(
                state["job_id"],
                {
                    "job_id": state["job_id"],
                    "pipeline_status": "FAILED",
                    "current_agent": failing or state.get("current_agent", "UNKNOWN"),
                    "detail": str(exc)[:500],
                },
            )
        return state

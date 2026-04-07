from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Agentic Validation System", layout="wide")
st.title("LangGraph Agentic Validation System")

if "rule_sets" not in st.session_state:
    st.session_state.rule_sets = {}

with st.sidebar:
    st.subheader("Input Section")
    drive_url = st.text_input("Google Drive Folder Link")
    if drive_url:
        is_google = "drive.google.com" in drive_url
        st.success("Drive URL format looks valid") if is_google else st.error("Not a Google Drive URL")
    rule_set_name = st.text_input("Rule Set Name", placeholder="A, SFT_RULES")
    rules_text = st.text_area("Rules (one per line)", height=180)
    if st.button("Add Rule Set"):
        rules = [line.strip() for line in rules_text.splitlines() if line.strip()]
        if rule_set_name and rules:
            st.session_state.rule_sets[rule_set_name] = rules
        else:
            st.warning("Both rule set name and rules are required.")
    st.subheader("Rule Set Preview")
    for key, rules in list(st.session_state.rule_sets.items()):
        st.write(f"**{key}**")
        st.code("\n".join(rules))
        col1, col2 = st.columns(2)
        if col1.button(f"Delete {key}"):
            del st.session_state.rule_sets[key]
            st.rerun()
        if col2.button(f"Edit {key}"):
            st.info(f"Re-enter values and click Add Rule Set to overwrite {key}.")

    st.subheader("LLM (code generation)")
    _prov_default = os.getenv("LLM_PROVIDER", "nvidia")
    _providers = ["openai", "nvidia", "anthropic", "gemini"]
    _idx = _providers.index(_prov_default) if _prov_default in _providers else 1
    llm_provider = st.selectbox(
        "Provider",
        _providers,
        index=_idx,
        help="Option B: choose NVIDIA and set NVIDIA_API_KEY + NVIDIA_MODEL in .env",
    )
    _model_default = (
        os.getenv("NVIDIA_MODEL", "google/gemma-4-31b-it")
        if llm_provider == "nvidia"
        else os.getenv("LLM_MODEL", "gpt-4o")
    )
    llm_model = st.text_input("Model name", value=_model_default)

    st.subheader("Fast mode (approximate)")
    fast_mode = st.checkbox(
        "Enable fast mode",
        value=bool(os.getenv("FAST_MODE_DEFAULT", "false").lower() in ("1", "true", "yes")),
        help="Compile fewer rules and sample fewer records per file for quicker feedback.",
    )
    max_rules_per_set = st.number_input(
        "Max rules per rule set (fast mode)",
        min_value=1,
        max_value=20,
        value=int(os.getenv("FAST_MAX_RULES_PER_SET", "2")),
        step=1,
    )
    max_records_per_file = st.number_input(
        "Max records per file (fast mode)",
        min_value=100,
        max_value=100000,
        value=int(os.getenv("FAST_MAX_RECORDS_PER_FILE", "5000")),
        step=100,
        help="For .txt/.jsonl/.json files, only this many records will be validated in fast mode.",
    )

col_run, col_reset = st.columns([1, 1])
if col_run.button("Run Validation", type="primary"):
    payload = {
        "drive_url": drive_url,
        "rule_sets": st.session_state.rule_sets,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "fast_mode": fast_mode,
        "max_rules_per_set": int(max_rules_per_set),
        "max_records_per_file": int(max_records_per_file) if fast_mode else None,
    }
    resp = requests.post(f"{API_BASE}/validate", json=payload, timeout=30)
    if resp.status_code == 200:
        st.session_state.job_id = resp.json()["job_id"]
    else:
        st.error(resp.text)

if col_reset.button("Reset"):
    st.session_state.rule_sets = {}
    st.session_state.pop("job_id", None)
    st.rerun()

if "job_id" in st.session_state:
    st.subheader("Output Section")
    st.write(f"Job ID: `{st.session_state.job_id}`")

    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    logs_placeholder = st.empty()

    status = {}
    poll_interval_sec = int(os.getenv("UI_POLL_INTERVAL_SEC", "5"))
    max_polls = int(os.getenv("UI_MAX_POLLS", "720"))
    started = time.monotonic()
    for _ in range(max_polls):
        try:
            # Allow long-running agents (e.g., large file analysis) without crashing the UI.
            resp = requests.get(
                f"{API_BASE}/status/{st.session_state.job_id}",
                timeout=int(os.getenv("UI_STATUS_TIMEOUT_SEC", "300")),
            )
            status = resp.json()
        except requests.exceptions.ReadTimeout:
            # Backend still working; show a soft warning and keep waiting.
            elapsed_min = (time.monotonic() - started) / 60.0
            status_placeholder.warning(
                f"Backend still busy (no response from /status yet). Elapsed: {elapsed_min:.1f} min"
            )
            time.sleep(poll_interval_sec)
            continue

        current_agent = status.get("current_agent", "INIT")
        pipeline = status.get("pipeline_status", "RUNNING")
        detail = status.get("detail") or ""
        elapsed_min = (time.monotonic() - started) / 60.0
        line = f"Current agent: {current_agent} | Status: {pipeline} | Elapsed: {elapsed_min:.1f} min"
        if detail:
            line += f" | {detail}"
        status_placeholder.info(line)
        order = [
            "AccessValidatorAgent",
            "DriveFetchAgent",
            "FileClassifierAgent",
            "RuleInputSystem",
            "RuleMatchingAgent",
            "CodeGenerationAgent",
            "ExecutionAgent",
            "FileFormatAnalysisAgent",
            "ReportGeneratorAgent",
        ]
        idx = order.index(current_agent) + 1 if current_agent in order else 0
        halted_text = f"Agent {idx}/9: {current_agent} halted ({pipeline})"
        running_text = f"Agent {idx}/9: {current_agent} running..."
        progress_placeholder.progress(min(idx / len(order), 1.0), text=halted_text if pipeline == "FAILED" else running_text)
        logs_placeholder.code(status.get("logs", ""), language="text")
        if pipeline in {"COMPLETED", "FAILED"}:
            break
        time.sleep(poll_interval_sec)

    pipeline_status = status.get("pipeline_status")
    if pipeline_status in {"COMPLETED", "FAILED"}:
        report = requests.get(f"{API_BASE}/report/{st.session_state.job_id}", timeout=30).json()
        if pipeline_status == "COMPLETED":
            st.success("Validation completed")
        else:
            st.error("Validation failed early. See error details below.")

        summary = report.get("summary", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Files", summary.get("total_files", 0))
        c2.metric("Passed", summary.get("passed", 0))
        c3.metric("Failed", summary.get("failed", 0))
        c4.metric("Skipped", summary.get("skipped", 0))
        c5.metric("Pass Rate", summary.get("overall_pass_rate", "0%"))

        if report.get("errors"):
            st.subheader("Errors")
            st.json(report.get("errors", []))

        for file_result in report.get("files", []):
            badge = file_result.get("status", "SKIPPED")
            color = {"PASSED": "green", "FAILED": "red", "SKIPPED": "gray", "INVALID": "red"}.get(badge, "gray")
            with st.expander(f"{file_result.get('file_name')} [{badge}]"):
                st.markdown(f"**Status:** :{color}[{badge}]")
                st.json(file_result.get("rule_evaluations", []))
                st.json(file_result.get("format_analysis", {}))

        json_bytes = json.dumps(report, indent=2).encode("utf-8")
        pdf_bytes = requests.get(f"{API_BASE}/report/{st.session_state.job_id}?format=pdf", timeout=60).content
        b1, b2 = st.columns(2)
        b1.download_button("Download JSON", data=json_bytes, file_name="validation_report.json", mime="application/json")
        b2.download_button("Download PDF", data=pdf_bytes, file_name="validation_report.pdf", mime="application/pdf")

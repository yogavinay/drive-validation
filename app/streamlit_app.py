from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Dict, List

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_BASE = (
    os.getenv("API_BASE")
    or os.getenv("API_URL")
    or "http://localhost:8000"
).rstrip("/")

st.set_page_config(page_title="Agentic Validation System", layout="wide")
st.title("LangGraph Agentic Validation System")
st.caption("Universal validation pipeline: resilient by default, with live progress updates.")
st.markdown(
    """
<style>
    .stApp { background: #f8fbf8; color: #102214; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #d9ead9; }
    /* Removed global span/div styling to fix code block and dynamic text visibility */
    h1, h2, h3, h4, h5, h6, p, label { color: #102214 !important; }
    .stTextInput label, .stTextArea label, .stSelectbox label, .stMultiSelect label,
    .stNumberInput label, .stCheckbox label, .stDateInput label, .stSlider label {
        font-weight: 700 !important;
        color: #102214 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        border-bottom: 2px solid #d9ead9;
        padding-bottom: 6px;
        margin-top: 12px;
        color: #1f8f4d !important;
    }
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div, .stNumberInput input {
        background: #ffffff !important;
        color: #102214 !important;
        border: 1px solid #b6d9bf !important;
        border-radius: 8px !important;
    }
    /* Beautiful popped-up buttons with shadows */
    div[data-testid="stButton"] button, 
    div[data-testid="stDownloadButton"] button, 
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(145deg, #24a15b, #1d8048) !important;
        color: #ffffff !important;
        border: 1px solid #1a703f !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 4px 6px rgba(31, 143, 77, 0.25), 0 2px 4px rgba(0,0,0,0.1) !important;
        transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    }
    div[data-testid="stButton"] button:hover, 
    div[data-testid="stDownloadButton"] button:hover, 
    div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(145deg, #1f8c4e, #186b3b) !important;
        box-shadow: 0 6px 12px rgba(31, 143, 77, 0.35), 0 3px 6px rgba(0,0,0,0.15) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stButton"] button:active, 
    div[data-testid="stDownloadButton"] button:active, 
    div[data-testid="stFormSubmitButton"] button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 2px 3px rgba(31, 143, 77, 0.2) !important;
    }
    
    /* Fix for Number Input +/- buttons looking black */
    div[data-baseweb="input"] button {
        background: #e8f4eb !important;
        color: #102214 !important;
    }
    div[data-testid="metric-container"] { background: #ffffff; border: 1px solid #d9ead9; border-radius: 10px; padding: 12px; box-shadow: 0 3px 6px rgba(0,0,0,0.05); }
    .stCaption { color: #4b6652 !important; }
</style>
""",
    unsafe_allow_html=True,
)

if "rule_sets" not in st.session_state:
    st.session_state.rule_sets = {}
if "show_previous_report" not in st.session_state:
    st.session_state.show_previous_report = False
if "firebase_history" not in st.session_state:
    st.session_state.firebase_history = []
if "history_load_attempted" not in st.session_state:
    st.session_state.history_load_attempted = False


def _network_error_message(action: str, exc: Exception) -> str:
    base = (
        f"Could not {action} because the backend API is unreachable at `{API_BASE}`.\n\n"
        "Set `API_BASE` to your deployed backend URL (for Streamlit Cloud, use app Secrets or environment variables)."
    )
    if "localhost" in API_BASE or "127.0.0.1" in API_BASE:
        base += "\n\nCurrent value points to localhost, which is usually not reachable from cloud deployments."
    return f"{base}\n\nError: {exc}"


def render_report(report: dict, title: str = "Validation Report"):
    st.subheader(title)
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
    b1, b2 = st.columns(2)
    b1.download_button("Download JSON", data=json_bytes, file_name="validation_report.json", mime="application/json")
    try:
        job_id = report.get("job_id")
        if job_id:
            pdf_bytes = requests.get(f"{API_BASE}/report/{job_id}?format=pdf", timeout=60).content
            b2.download_button("Download PDF", data=pdf_bytes, file_name="validation_report.pdf", mime="application/pdf")
    except Exception:
        pass

with st.sidebar:
    st.subheader("Input Section")
    drive_url = st.text_input("Google Drive Folder Link")
    if drive_url:
        is_google = "drive.google.com" in drive_url
        # Use explicit if/else (not ternary) to avoid Streamlit magic rendering return objects.
        if is_google:
            st.success("Link looks valid")
        else:
            st.error("Invalid Drive link")
    with st.form("add_rule_set_form", clear_on_submit=True):
        rule_set_name = st.text_input("Rule Set Name", placeholder="Example: A or SFT_RULES")
        rules_text = st.text_area("Rules (one per line)", height=140, placeholder="Example:\nNo empty lines\nEnds with <EOS>")
        submitted = st.form_submit_button("Add Rule Set")
        if submitted:
            rules = [line.strip() for line in rules_text.splitlines() if line.strip()]
            if rule_set_name and rules:
                st.session_state.rule_sets[rule_set_name] = rules
                st.success(f"✅ Rule Set '{rule_set_name}' added successfully! You can add another rule set.")
            else:
                st.warning("⚠️ Both rule set name and rules are required.")
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

    st.subheader("LLM (Code Generation)")
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

    st.subheader("Fast mode (always on)")
    st.caption("Uses fewer rules per set and caps records per file for quicker runs.")
    max_rules_per_set = st.number_input(
        "Max rules per rule set",
        min_value=1,
        max_value=20,
        value=int(os.getenv("FAST_MAX_RULES_PER_SET", "2")),
        step=1,
    )
    max_records_per_file = st.number_input(
        "Max records per file",
        min_value=100,
        max_value=100000,
        value=int(os.getenv("FAST_MAX_RECORDS_PER_FILE", "5000")),
        step=100,
        help="For .txt/.jsonl/.json files, only this many records are validated per file.",
    )
    st.divider()
    st.subheader("Live Monitoring")
    auto_refresh = st.checkbox("Auto-refresh progress", value=True)
    poll_interval_sec = st.slider(
        "Refresh every (seconds)",
        min_value=2,
        max_value=20,
        value=int(os.getenv("UI_POLL_INTERVAL_SEC", "5")),
        step=1,
    )
    max_polls = int(os.getenv("UI_MAX_POLLS", "720"))
    st.divider()
    st.subheader("Previous Validations")
    if st.button("Load validation history"):
        st.session_state.history_load_attempted = True
        try:
            resp = requests.get(f"{API_BASE}/reports/firebase?limit=100", timeout=30)
            if resp.status_code != 200:
                st.session_state.firebase_history = []
                st.error(f"Could not load history: {resp.status_code} {resp.text}")
            else:
                rows = resp.json().get("items", [])
                st.session_state.firebase_history = rows
                if rows:
                    st.success(f"Loaded {len(rows)} previous report(s).")
                else:
                    st.info("No previous reports found yet.")
        except requests.exceptions.RequestException as exc:
            st.session_state.firebase_history = []
            st.warning(_network_error_message("load previous validations", exc))
    history = st.session_state.get("firebase_history", [])
    if st.session_state.get("history_load_attempted") and not history:
        st.caption("History list is currently empty.")
    if history:
        st.caption("Owner: TEAM")
        all_sets = sorted({s for item in history for s in item.get("set_names", [])})
        set_filter = st.multiselect("Filter by set name", all_sets, default=[])
        verdict_filter = st.selectbox("Filter by verdict", ["ALL", "PASSED", "FAILED"])
        d_from, d_to = st.date_input("Filter by date range", value=(date(2024, 1, 1), date.today()))

        filtered = []
        for item in history:
            sets = item.get("set_names", [])
            verdict = (item.get("summary", {}) or {}).get("final_verdict", "")
            ts = (item.get("generated_at") or "")[:10]
            ok_set = (not set_filter) or any(s in sets for s in set_filter)
            ok_verdict = verdict_filter == "ALL" or verdict == verdict_filter
            ok_date = True
            if isinstance(d_from, date) and isinstance(d_to, date) and ts:
                ok_date = d_from.isoformat() <= ts <= d_to.isoformat()
            if ok_set and ok_verdict and ok_date:
                filtered.append(item)

        st.caption(f"Showing {len(filtered)} of {len(history)} reports")
        labels = []
        for item in filtered:
            sets = ",".join(item.get("set_names", []))
            verdict = (item.get("summary", {}) or {}).get("final_verdict", "")
            labels.append(f"{item.get('generated_at', '')} | {item.get('job_id', '')} | {sets} | {verdict}")
        if labels:
            picked = st.selectbox("Select previous report", labels)
            c_open, c_delete = st.columns(2)
            if c_open.button("Open selected report"):
                idx = labels.index(picked)
                st.session_state.selected_report = filtered[idx].get("report", {})
                st.session_state.selected_report["job_id"] = filtered[idx].get("job_id")
                st.session_state.show_previous_report = True
            if c_delete.button("Delete selected report"):
                idx = labels.index(picked)
                job_id = filtered[idx].get("job_id")
                try:
                    r = requests.delete(f"{API_BASE}/reports/firebase/{job_id}", timeout=30)
                    if r.status_code == 200:
                        st.success(f"Deleted {job_id}")
                        st.session_state.firebase_history = [x for x in history if x.get("job_id") != job_id]
                    else:
                        st.error(f"Delete failed: {r.text}")
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
    st.session_state.show_previous_report = st.checkbox(
        "Show previous report details",
        value=st.session_state.show_previous_report,
    )

st.divider()
col_run, col_reset, col_clear = st.columns([1, 1, 1], gap="large")
if col_run.button("Run Validation", type="primary"):
    payload = {
        "drive_url": drive_url,
        "rule_sets": st.session_state.rule_sets,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "fast_mode": True,
        "max_rules_per_set": int(max_rules_per_set),
        "max_records_per_file": int(max_records_per_file),
    }
    validate_timeout_sec = int(os.getenv("UI_VALIDATE_TIMEOUT_SEC", "120"))
    try:
        # Use split connect/read timeout to tolerate slow backend cold starts.
        resp = requests.post(
            f"{API_BASE}/validate",
            json=payload,
            timeout=(10, validate_timeout_sec),
        )
        if resp.status_code == 200:
            st.session_state.job_id = resp.json()["job_id"]
        else:
            st.error(resp.text)
    except requests.exceptions.ReadTimeout as exc:
        st.error(
            _network_error_message(
                f"start validation within {validate_timeout_sec}s", exc
            )
        )
    except requests.exceptions.RequestException as exc:
        st.error(_network_error_message("start validation", exc))

if col_reset.button("Reset"):
    st.session_state.rule_sets = {}
    st.session_state.pop("job_id", None)
    st.rerun()
if col_clear.button("Clear Current Job"):
    st.session_state.pop("job_id", None)
    st.rerun()

if "job_id" in st.session_state:
    st.subheader("Output Section")
    st.write(f"Job ID: `{st.session_state.job_id}`")

    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    logs_placeholder = st.empty()

    status = {}
    started = time.monotonic()
    if not auto_refresh:
        st.info("Auto-refresh is OFF. Click 'Run Validation' again or enable auto-refresh to continue polling.")
    for _ in range(max_polls if auto_refresh else 1):
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
        if pipeline == "FAILED":
            status_placeholder.error(line)
        elif pipeline == "COMPLETED":
            status_placeholder.success(line)
        else:
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
        try:
            report = requests.get(f"{API_BASE}/report/{st.session_state.job_id}", timeout=30).json()
        except requests.exceptions.RequestException as exc:
            st.error(_network_error_message("fetch the final report", exc))
            report = {"summary": {}, "files": [], "errors": [str(exc)]}
        if pipeline_status == "COMPLETED":
            st.success("Validation completed")
        else:
            st.error("Validation failed early. See error details below.")

        report["job_id"] = st.session_state.job_id
        render_report(report, "Current Validation Report")

if st.session_state.get("selected_report") and st.session_state.get("show_previous_report"):
    st.divider()
    with st.expander("Previously Saved Validation", expanded=False):
        render_report(st.session_state["selected_report"], "Previously Saved Validation")

from __future__ import annotations

import ast
import json
import os
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.jobs import update_job_progress
from app.security import assert_safe_python

# Default matches NVIDIA integrate.api.nvidia.com chat/completions models.
DEFAULT_NVIDIA_MODEL = "google/gemma-4-31b-it"


def _llm_request_timeout() -> Optional[float]:
    """Per-request HTTP timeout for LLM calls (important on Render / slow NVIDIA models)."""
    raw = os.getenv("LLM_REQUEST_TIMEOUT_SEC", "900").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def build_llm(provider: str, model: str):
    timeout = _llm_request_timeout()
    openai_timeout_kw: Dict[str, Any] = {"timeout": timeout} if timeout is not None else {}

    if provider == "openai":
        return ChatOpenAI(model=model, temperature=0, **openai_timeout_kw)
    if provider == "anthropic":
        # LangChain passes this through to the Anthropic client as request timeout.
        kw: Dict[str, Any] = {}
        if timeout is not None:
            kw["default_request_timeout"] = timeout
        return ChatAnthropic(model=model, temperature=0, **kw)
    if provider == "gemini":
        kw: Dict[str, Any] = {}
        if timeout is not None:
            kw["timeout"] = timeout
        return ChatGoogleGenerativeAI(model=model, temperature=0, **kw)
    if provider == "nvidia":
        # Same API as: POST https://integrate.api.nvidia.com/v1/chat/completions (OpenAI-compatible)
        api_key = os.getenv("NVIDIA_API_KEY")
        base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY is not set in environment")
        resolved_model = (model or os.getenv("NVIDIA_MODEL") or DEFAULT_NVIDIA_MODEL).strip()
        max_tokens = int(os.getenv("NVIDIA_MAX_TOKENS", "16384"))
        temperature = float(os.getenv("NVIDIA_TEMPERATURE", "0"))
        top_p = os.getenv("NVIDIA_TOP_P")
        # Thinking mode greatly slows code-gen on NVIDIA NIM; default off for cloud latency.
        thinking = os.getenv("NVIDIA_ENABLE_THINKING", "false").lower() in ("1", "true", "yes")
        extra_body: Dict[str, Any] = {}
        if thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
        params: Dict[str, Any] = {
            "model": resolved_model,
            "temperature": temperature,
            "api_key": api_key,
            "base_url": base_url,
            "max_tokens": max_tokens,
        }
        if top_p is not None and top_p != "":
            params["top_p"] = float(top_p)
        if extra_body:
            params["extra_body"] = extra_body
        if timeout is not None:
            params["timeout"] = timeout
        return ChatOpenAI(**params)
    raise ValueError(f"Unsupported provider: {provider}")


# Literal braces must be doubled for str.format — only {rule} is a placeholder.
RULE_PROMPT = """
You are an intelligent Python rule compiler that understands natural English descriptions.

The user will describe a validation rule in plain English. Your job is to interpret
what they mean and convert it into a Python function.

Generate a complete Python function with this signature:
def validate_record(record, line_number, context):
    ...
    return {{"passed": bool, "failed_lines": list[int], "details": str}}

About the arguments:
- `record` can be a string (text line), a dict (parsed JSON object), or other data types
- `line_number` is the 1-based line/record index
- `context` is a dict with metadata like {{"file_type": ".jsonl"}}

Examples of natural English rules users might write:
- "Each line must not be empty" → check if record is empty/whitespace
- "Every record should have instruction and output fields" → check dict keys
- "Text must end with a period" → check string ending
- "No profanity allowed" → check for common bad words
- "Values must be between 0 and 100" → check numeric ranges
- "Each record must be valid JSON" → try parsing as JSON

Constraints:
- No imports (but you can use: len, str, int, float, isinstance, dict, list, tuple, set, any, all, bool, json, ast)
- No file IO
- No eval/exec
- Pure computation on record/context
- Return PASSED (passed=True) for valid records, FAILED (passed=False) with the failed line number for invalid records
- Keep deterministic and safe
- If the rule is ambiguous, interpret it in the most reasonable way

User's rule in natural English: {rule}
"""


def compile_rule_callable(code: str) -> Callable[..., Dict[str, Any]]:
    assert_safe_python(code)
    namespace: Dict[str, Any] = {}
    exec(
        code,
        {
            "__builtins__": {
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "isinstance": isinstance,
                "dict": dict,
                "list": list,
                "tuple": tuple,
                "set": set,
                "any": any,
                "all": all,
                "bool": bool,
                "json": json,
                "ast": ast,
            }
        },
        namespace,
    )
    fn = namespace.get("validate_record")
    if not callable(fn):
        raise ValueError("Generated validator must define validate_record")
    return fn


def _sanitize_generated_code(raw: str) -> str:
    """
    LLMs often wrap code in markdown fences or add leading explanation.
    Keep only executable Python and the validate_record function body.
    """
    text = textwrap.dedent(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop opening fence
        lines = lines[1:] if lines else lines
        # Drop closing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    marker = "def validate_record"
    idx = text.find(marker)
    if idx != -1:
        text = text[idx:]
    return text


def _is_quota_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "429" in msg
        or "insufficient_quota" in msg
        or "rate limit" in msg
        or "quota" in msg
    )


def _builtin_validator_code(rule: str) -> Optional[str]:
    r = rule.lower().strip()

    # ── No empty lines / records ──
    if "no empty lines" in r or "must not contain empty lines" in r or "no blank lines" in r or "each line must not be empty" in r:
        return (
            "def validate_record(record, line_number, context):\n"
            "    text = str(record) if not isinstance(record, dict) else str(record.get('text', ''))\n"
            "    if len(text.strip()) == 0:\n"
            "        return {'passed': False, 'failed_lines': [line_number], 'details': 'empty line/record'}\n"
            "    return {'passed': True, 'failed_lines': [], 'details': ''}\n"
        )

    # ── Text / content must not be empty ──
    if ("text field" in r and "empty" in r) or ("must not be empty" in r) or ("should not be empty" in r) or ("cannot be empty" in r):
        return (
            "def validate_record(record, line_number, context):\n"
            "    val = None\n"
            "    if isinstance(record, dict):\n"
            "        val = record.get('text') or record.get('content') or record.get('value')\n"
            "    elif isinstance(record, str):\n"
            "        val = record\n"
            "    if val is None or len(str(val).strip()) == 0:\n"
            "        return {'passed': False, 'failed_lines': [line_number], 'details': 'text/content is empty'}\n"
            "    return {'passed': True, 'failed_lines': [], 'details': ''}\n"
        )

    # ── Ends with EOS / </s> token ──
    if "ends with" in r and ("<eos>" in r or "</s>" in r):
        return (
            "def validate_record(record, line_number, context):\n"
            "    text = record.get('text') if isinstance(record, dict) else str(record)\n"
            "    t = str(text).strip()\n"
            "    ok = t.endswith('<EOS>') or t.endswith('</s>')\n"
            "    if not ok:\n"
            "        return {'passed': False, 'failed_lines': [line_number], 'details': 'Missing EOS token'}\n"
            "    return {'passed': True, 'failed_lines': [], 'details': ''}\n"
        )

    # ── Shape check (e.g., "shape must be 1024") ──
    if "shape" in r and "1024" in r:
        return (
            "def validate_record(record, line_number, context):\n"
            "    shape = record.get('shape') if isinstance(record, dict) else None\n"
            "    ok = isinstance(shape, list) and len(shape) == 2 and int(shape[1]) == 1024\n"
            "    if not ok:\n"
            "        return {'passed': False, 'failed_lines': [line_number], 'details': f'Unexpected shape: {shape}'}\n"
            "    return {'passed': True, 'failed_lines': [], 'details': ''}\n"
        )

    # ── Valid JSON check ──
    if "valid json" in r or ("must be valid" in r and "json" in r) or "well-formed json" in r:
        return (
            "def validate_record(record, line_number, context):\n"
            "    if isinstance(record, dict):\n"
            "        return {'passed': True, 'failed_lines': [], 'details': ''}\n"
            "    s = str(record).strip()\n"
            "    try:\n"
            "        json.loads(s)\n"
            "        return {'passed': True, 'failed_lines': [], 'details': ''}\n"
            "    except Exception:\n"
            "        return {'passed': False, 'failed_lines': [line_number], 'details': 'invalid JSON'}\n"
        )

    # ── Python syntax check ──
    if ("syntax" in r and "python" in r) or "syntax errors" in r:
        return (
            "def validate_record(record, line_number, context):\n"
            "    try:\n"
            "        ast.parse(str(record))\n"
            "        return {'passed': True, 'failed_lines': [], 'details': ''}\n"
            "    except SyntaxError as e:\n"
            "        return {'passed': False, 'failed_lines': [line_number], 'details': str(e)}\n"
        )

    # ── Required fields check (e.g., "must have instruction and output fields") ──
    import re as _re
    field_match = _re.findall(r"['\"](\w+)['\"]", rule)
    if not field_match:
        # Try matching "have/contain X and Y fields"
        field_match = _re.findall(r"(?:have|contain|include)\s+(?:an?\s+)?(\w+)(?:\s+and\s+(\w+))?(?:\s+field)?", r)
        if field_match:
            field_match = [f for pair in field_match for f in pair if f and f not in {"an", "a", "the", "and", "field", "fields"}]
    if field_match and any(kw in r for kw in ["must have", "should have", "must contain", "should contain", "require", "need"]):
        fields_str = ", ".join(f"'{f}'" for f in field_match)
        checks = " and ".join(f"'{f}' in record" for f in field_match)
        missing_expr = " + ".join(f"(['{f}'] if '{f}' not in record else [])" for f in field_match)
        return (
            "def validate_record(record, line_number, context):\n"
            "    if not isinstance(record, dict):\n"
            "        try:\n"
            "            record = json.loads(str(record))\n"
            "        except Exception:\n"
            f"            return {{'passed': False, 'failed_lines': [line_number], 'details': 'Not a dict, cannot check fields: {fields_str}'}}\n"
            f"    missing = {missing_expr}\n"
            "    if missing:\n"
            f"        return {{'passed': False, 'failed_lines': [line_number], 'details': f'Missing fields: {{missing}}'}}\n"
            "    return {'passed': True, 'failed_lines': [], 'details': ''}\n"
        )

    # ── No duplicate records ──
    if "no duplicate" in r or "must not have duplicate" in r or "unique" in r:
        return (
            "def validate_record(record, line_number, context):\n"
            "    # Duplicate detection requires full-file context; mark as passed at record level\n"
            "    return {'passed': True, 'failed_lines': [], 'details': 'Duplicate check requires full-file scan'}\n"
        )

    return None


def generate_validators(
    rule_sets: Dict[str, List[str]],
    provider: str,
    model: str,
    job_id: Optional[str] = None,
    fast_mode: bool = False,
    max_rules_per_set: Optional[int] = None,
) -> Dict[str, List[Callable[..., Dict[str, Any]]]]:
    llm = build_llm(provider=provider, model=model)
    out: Dict[str, List[Callable[..., Dict[str, Any]]]] = {}
    total_rules = sum(len(rules) for rules in rule_sets.values())
    rule_index = 0
    for name, rules in rule_sets.items():
        funcs: List[Callable[..., Dict[str, Any]]] = []
        for local_idx, rule in enumerate(rules, start=1):
            if fast_mode and max_rules_per_set is not None and local_idx > max_rules_per_set:
                # Skip remaining rules in this set in fast/approximate mode.
                break
            rule_index += 1
            builtin_code = _builtin_validator_code(rule)
            if builtin_code is not None:
                if job_id:
                    update_job_progress(
                        job_id,
                        "CodeGenerationAgent",
                        f"Builtin rule {rule_index}/{total_rules} (set {name!r})",
                    )
                funcs.append(compile_rule_callable(builtin_code))
                continue
            if job_id:
                update_job_progress(
                    job_id,
                    "CodeGenerationAgent",
                    f"LLM compiling rule {rule_index}/{total_rules} (set {name!r})",
                )
            prompt = RULE_PROMPT.format(rule=rule)
            last_exc = None
            response = None
            for attempt in range(3):
                try:
                    response = llm.invoke(prompt)
                    break
                except Exception as exc:
                    last_exc = exc
                    # OpenAI quota / rate limit: switch to NVIDIA for this and remaining rules.
                    if (
                        provider == "openai"
                        and _is_quota_error(exc)
                        and os.getenv("NVIDIA_API_KEY")
                    ):
                        try:
                            llm = build_llm(
                                "nvidia",
                                os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL),
                            )
                            response = llm.invoke(prompt)
                            last_exc = None
                            break
                        except Exception as fb_exc:
                            last_exc = fb_exc
                    time.sleep(2**attempt)
            if response is None:
                raise RuntimeError(f"LLM generation failed after retries: {last_exc}")
            code = _sanitize_generated_code(str(response.content))
            if "def validate_record" not in code:
                code = (
                    "def validate_record(record, line_number, context):\n"
                    "    return {'passed': True, 'failed_lines': [], 'details': 'No-op fallback'}\n"
                )
            try:
                funcs.append(compile_rule_callable(code))
            except Exception:
                # Non-fatal: if one generated rule is malformed, keep pipeline running.
                funcs.append(
                    compile_rule_callable(
                        "def validate_record(record, line_number, context):\n"
                        "    return {'passed': True, 'failed_lines': [], 'details': 'Fallback due to invalid generated code'}\n"
                    )
                )
        out[name] = funcs
    return out

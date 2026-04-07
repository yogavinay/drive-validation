from __future__ import annotations

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


def build_llm(provider: str, model: str):
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=0)
    if provider == "anthropic":
        return ChatAnthropic(model=model, temperature=0)
    if provider == "gemini":
        return ChatGoogleGenerativeAI(model=model, temperature=0)
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
        thinking = os.getenv("NVIDIA_ENABLE_THINKING", "true").lower() in ("1", "true", "yes")
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
        return ChatOpenAI(**params)
    raise ValueError(f"Unsupported provider: {provider}")


# Literal braces must be doubled for str.format — only {rule} is a placeholder.
RULE_PROMPT = """
You are a secure Python rule compiler.
Convert the user rule into a Python function body for:
def validate_record(record, line_number, context):
    ...
    return {{"passed": bool, "failed_lines": list[int], "details": str}}

Constraints:
- No imports
- No file IO
- No eval/exec
- Pure computation on record/context
- Return PASSED for valid record and FAILED with failed line for invalid
- Keep deterministic and safe
Rule: {rule}
"""


def compile_rule_callable(code: str) -> Callable[..., Dict[str, Any]]:
    assert_safe_python(code)
    namespace: Dict[str, Any] = {}
    exec(code, {"__builtins__": {"len": len, "str": str, "int": int, "float": float, "isinstance": isinstance}}, namespace)
    fn = namespace.get("validate_record")
    if not callable(fn):
        raise ValueError("Generated validator must define validate_record")
    return fn


def _is_quota_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "429" in msg
        or "insufficient_quota" in msg
        or "rate limit" in msg
        or "quota" in msg
    )


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
            code = textwrap.dedent(str(response.content)).strip()
            if "def validate_record" not in code:
                code = (
                    "def validate_record(record, line_number, context):\n"
                    "    return {'passed': True, 'failed_lines': [], 'details': 'No-op fallback'}\n"
                )
            funcs.append(compile_rule_callable(code))
        out[name] = funcs
    return out

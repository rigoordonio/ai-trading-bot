"""Optional LLM explanation layer: TEXT IN, TEXT OUT.

The LLM receives an already-made decision (facts from the audit trail) and
writes a plain-English explanation. It has no access to brokers, orders,
the risk engine or configuration, and its output is never parsed for
actions. This module deliberately imports nothing from src.execution,
src.risk or src.portfolio (enforced by tests/test_ai_guardrails.py).

Without an LLM (LLM_PROVIDER=none, the default) a deterministic template is
used, so explanations work offline.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import requests

SYSTEM_PROMPT = (
    "You explain decisions already made by a deterministic quantitative trading system. "
    "You cannot place, modify or recommend orders. Do not give financial advice. "
    "Explain only the facts provided, in 2-4 plain sentences, and mention the risk status."
)


@dataclass(frozen=True)
class DecisionFacts:
    symbol: str
    bot_id: str
    action: str
    score: float | None
    ml_probability: float | None
    components: Mapping[str, float | None]
    risk_status: str | None
    reason: str | None
    rule_triggered: str | None = None

    @classmethod
    def from_audit(cls, rec: Mapping) -> "DecisionFacts":
        comps = rec.get("components")
        if isinstance(comps, str):
            try:
                comps = json.loads(comps.replace("'", '"').replace("None", "null").replace("nan", "null"))
            except json.JSONDecodeError:
                comps = {}
        if not isinstance(comps, Mapping):   # NaN / None / garbage from a CSV round trip
            comps = {}
        return cls(str(rec.get("symbol")), str(rec.get("bot_id")), str(rec.get("final_action") or rec.get("action")),
                   _num(rec.get("score")), _num(rec.get("ml_probability")), MappingProxyType(dict(comps)),
                   rec.get("risk_status"), rec.get("reason"), rec.get("rule_triggered"))


def _num(x):
    try:
        x = float(x)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def template_explanation(f: DecisionFacts) -> str:
    comps = {k: v for k, v in f.components.items() if v is not None and k not in ("bot_id", "account")}
    strong = [k.replace("_", " ") for k, v in sorted(comps.items(), key=lambda kv: -kv[1]) if v >= 0.6][:3]
    weak = [k.replace("_", " ") for k, v in sorted(comps.items(), key=lambda kv: kv[1]) if v <= 0.3][:2]
    parts = []
    if f.action in ("BUY", "SELL"):
        verb = "opened/added a position" if f.action == "BUY" else "reduced or exited a position"
        parts.append(f"{f.bot_id} {verb} in {f.symbol}")
    else:
        parts.append(f"{f.bot_id} did not trade {f.symbol}")
    if f.score is not None:
        parts[-1] += f" (combined score {f.score:.2f})"
    s = parts[0] + "."
    if strong:
        s += f" Supportive signals: {', '.join(strong)}."
    if weak:
        s += f" Weak signals: {', '.join(weak)}."
    if f.ml_probability is not None:
        s += f" The model estimated a {f.ml_probability:.0%} probability that the forward return beats its target."
    if f.risk_status:
        s += f" Risk check: {f.risk_status}" + (f" ({f.rule_triggered})" if f.rule_triggered else "") + "."
    if f.reason and f.action not in ("BUY",):
        s += f" Reason: {f.reason}."
    return s


def _llm_call(prompt: str, timeout: float = 20.0) -> str | None:
    provider = os.getenv("LLM_PROVIDER", "none").lower()
    model = os.getenv("LLM_MODEL", "")
    try:
        if provider == "ollama":
            url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/generate"
            r = requests.post(url, json={"model": model or "qwen2.5:7b", "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                                         "stream": False}, timeout=timeout)
            r.raise_for_status()
            return r.json().get("response", "").strip() or None
        if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=timeout,
                              headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                                       "content-type": "application/json"},
                              json={"model": model or "claude-haiku-4-5", "max_tokens": 300, "system": SYSTEM_PROMPT,
                                    "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json().get("content", [])).strip() or None
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            r = requests.post("https://api.openai.com/v1/chat/completions", timeout=timeout,
                              headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                              json={"model": model or "gpt-4o-mini", "max_tokens": 300,
                                    "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                                 {"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip() or None
    except (requests.RequestException, KeyError, ValueError):
        return None   # explanation is non-critical: fall back to the template
    return None


def explain(facts: DecisionFacts, use_llm: bool = True) -> str:
    """Returns a string. Never returns or triggers anything executable."""
    base = template_explanation(facts)
    if not use_llm or os.getenv("LLM_PROVIDER", "none").lower() == "none":
        return base
    prompt = "Facts (JSON):\n" + json.dumps({**facts.__dict__, "components": dict(facts.components)}, default=str)
    text = _llm_call(prompt)
    return text if isinstance(text, str) and text else base

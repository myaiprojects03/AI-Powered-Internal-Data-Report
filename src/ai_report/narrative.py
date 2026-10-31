"""Call the model, validate, repair, and fall back if needed. The report never ships unchecked text."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import SETTINGS
from .llm import LLMClient, LLMError, LLMOutputError, OfflineClient
from .metrics import MetricSet, detect_signals
from .prompt import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt, select_facts
from .validation import order_by_severity, tidy_wording, validate

log = logging.getLogger(__name__)


@dataclass
class Narrative:
    data: dict
    source: str                 # e.g. "groq:openai/gpt-oss-120b" or "offline:rule-based"
    attempts: int
    passed_checks: bool
    fell_back: bool = False
    issues: list[str] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)


def generate(ms: MetricSet, client: LLMClient, context: str = "", max_attempts: int | None = None) -> Narrative:
    signals = detect_signals(ms)
    facts = select_facts(ms, signals)
    user = build_user_prompt(ms, facts, signals, context)
    max_attempts = max_attempts or SETTINGS.llm_max_attempts

    issues: list[str] = []
    prompt = user
    for attempt in range(1, max_attempts + 1):
        try:
            raw = client.complete(SYSTEM_PROMPT, prompt)
        except LLMOutputError as e:
            issues.append(f"Attempt {attempt}: {e}")
            log.warning(issues[-1])
            prompt = user                  # a shorter prompt gives the model more room to finish
            continue
        except LLMError as e:
            issues.append(f"Attempt {attempt}: model call failed: {e}")
            log.warning(issues[-1])
            break                          # connection/auth errors will not fix themselves
        result = validate(raw, facts, signals)
        if result.ok:
            return Narrative(tidy_wording(order_by_severity(result.data, signals)), client.label, attempt, True, issues=issues, facts=facts, signals=signals)
        issues.append(f"Attempt {attempt}: {len(result.errors)} check(s) failed: " + "; ".join(result.errors[:5]))
        log.warning(issues[-1])
        prompt = build_repair_prompt(user, raw, result.errors)

    # Fallback: deterministic writer, still validated, clearly labelled in the report.
    fallback = OfflineClient()
    result = validate(fallback.complete(SYSTEM_PROMPT, user), facts, signals)
    if not result.ok:
        raise RuntimeError(f"Fallback writer failed validation: {result.errors}")
    return Narrative(tidy_wording(order_by_severity(result.data, signals)), fallback.label, 0, True, fell_back=not isinstance(client, OfflineClient),
                     issues=issues, facts=facts, signals=signals)

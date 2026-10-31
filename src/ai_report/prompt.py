"""Prompt construction: fixed instructions, figures as structured data, a set output format."""
from __future__ import annotations

import json
import re

from .metrics import MetricSet

FACTS_START, FACTS_END = "<<<FACTS_JSON", "FACTS_JSON>>>"

SYSTEM_PROMPT = """You write short business reports for non-technical managers.

You receive pre-calculated FACTS (each with an id, label and display value) and SIGNALS
(issues already detected by code). Your job is to explain them in plain language and
recommend actions. You never calculate anything.

Rules:
1. Use only numbers that appear in FACTS, written exactly as their "display" value.
   Do not round, add, subtract, compare as ratios ("2x", "double") or invent figures.
   Do not set numeric targets ("bring returns below 10%", "restore on-time to 70%"): a target is a
   new number that is not in FACTS. Describe the goal in words instead ("back to last month's level").
   Describe a change with a direction word that matches its sign: a negative change is
   "down", "fell" or "dropped"; a positive change is "up", "rose" or "grew".
2. Every summary, insight and action must list in "evidence" the fact ids it relies on.
   Every number you write must come from a fact listed in that SAME item's evidence. This applies to
   the summary too: if the summary mentions a figure, its fact id goes in the summary's evidence.
3. Plain English, short sentences, no jargon, no buzzwords. Write for someone with 30 seconds.
4. Each insight covers one SIGNAL and cites that signal's facts. Cover "high" signals first,
   then "medium", then "positive". Do not write an insight that only repeats the summary.
5. Each action addresses a SIGNAL and cites at least one of its facts. Actions must be concrete
   and owned: start with a specific verb and name the exact data to compare or the exact change to
   make, using the area and figure from the signal (patterns: "Compare <measure> for <area> week by
   week", "Set a limit on <lever> for <segment>"). Not acceptable: "Investigate", "Address",
   "Review the process", "Monitor", "Track".
   Say who should do it (a role, not a name) and by when (e.g. "this week").
6. Do not name a cause the data does not show. For example, a high return rate alone does not prove
   a quality problem; say what to check to find the cause instead.
7. The summary is 2 to 3 sentences: the overall result in one sentence, then the most urgent
   issues (every "high" signal) by name. Use at most 5 figures in the summary.
8. Give only actions someone can start on. "Monitor" or "track" is not an action. At most one action
   may be about growth alone. Three strong actions are better than five weak ones.
9. If a signal has a "caution", follow it. Never recommend growing, promoting or investing in an
   area that also has a high-severity problem; the problem comes first.

Return only one JSON object, no markdown, in exactly this shape:
{
  "summary": {"text": "2 to 3 sentences", "evidence": ["fact_id", ...]},
  "insights": [{"title": "max 8 words", "text": "1 to 2 sentences", "evidence": ["fact_id", ...]}],
  "actions": [{"action": "imperative sentence", "owner": "role", "timing": "when",
               "priority": "high|medium|low", "why": "1 sentence", "evidence": ["fact_id", ...]}]
}
Give 3 to 5 insights and 3 to 5 actions."""


def select_facts(ms: MetricSet, signals: list[dict]) -> list[dict]:
    """Send the model only what it needs: headline figures, signal figures, region revenue.

    Fewer facts means less room for confusion, especially with small local models.
    """
    wanted: list[str] = [f.id for f in ms.facts if f.group == "overall"]
    for s in signals:
        wanted += s["facts"]
    wanted += [f.id for f in ms.facts if f.group == "region" and f.id.endswith((".revenue", ".revenue_change"))]
    by_id = ms.by_id()
    seen, out = set(), []
    for fid in wanted:
        if fid not in seen:
            seen.add(fid)
            out.append(by_id[fid].to_dict())
    return out


def build_user_prompt(ms: MetricSet, facts: list[dict], signals: list[dict], context: str = "") -> str:
    payload = {
        "report_period": ms.current_period,
        "compared_with": ms.previous_period,
        "facts": [{"id": f["id"], "label": f["label"], "display": f["display"]} for f in facts],
        "signals": signals,
    }
    ctx = f"Business context: {context.strip()}\n\n" if context.strip() else ""
    high = [sig for sig in signals if sig["severity"] == "high"]
    required = ""
    if high:
        lines = "\n".join(f"- {sig['type']}: cite one of {sig['facts'][:2]}" for sig in high)
        required = f"The summary must name each of these urgent issues and cite its fact in the summary's evidence:\n{lines}\n\n"
    return (
        f"{ctx}Write the report for {ms.current_period} compared with {ms.previous_period}.\n\n"
        f"{FACTS_START}\n{json.dumps(payload, indent=1)}\n{FACTS_END}\n\n"
        f"{required}Return the JSON object now."
    )


def build_repair_prompt(original_user: str, bad_output: str, errors: list[str]) -> str:
    issues = "\n".join(f"- {e}" for e in errors[:15])
    return (
        f"{original_user}\n\nYour previous answer failed these checks:\n{issues}\n\n"
        f"Previous answer:\n{bad_output[:3000]}\n\n"
        "Fix every issue. Use only display values from FACTS and cite them in evidence. "
        "Return the corrected JSON object only."
    )


def extract_payload(user_prompt: str) -> dict:
    m = re.search(re.escape(FACTS_START) + r"\s*(.*?)\s*" + re.escape(FACTS_END), user_prompt, re.S)
    if not m:
        raise ValueError("No facts block in prompt")
    return json.loads(m.group(1))

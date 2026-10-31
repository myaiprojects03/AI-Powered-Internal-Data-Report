"""Checks on the model's output before anything reaches a reader.

1. Structure: valid JSON in the required shape, 3-5 insights and actions.
2. Evidence: every item cites at least one fact id that was actually provided.
3. Numbers: every number in the text matches a fact cited by that same item,
   allowing only for the rounding the writer visibly applied.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

NUM_RE = re.compile(
    r"(?<![\w.])(?P<sign>[-+−])?(?P<cur>[$£€])?(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?"
    r"\s*(?P<suffix>%|\s?(?:pts|points|percentage points|days|k|K|m|M|million|thousand)\b)?"
)
# Bare small integers ("3 regions", "top 5") and years are allowed without evidence.
FREE_INT_MAX = 10


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    data: dict | None = None


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found")
    return json.loads(text[start:end + 1])


def numbers_in(text: str) -> list[tuple[str, float, float]]:
    """Return (raw, value, tolerance) for each number written in the text."""
    out = []
    for m in NUM_RE.finditer(text):
        raw = m.group(0).strip()
        value = float(m.group("num").replace(",", "") + (m.group("dec") or ""))
        decimals = len(m.group("dec")) - 1 if m.group("dec") else 0
        tol = 0.5 * 10 ** -decimals
        suffix = (m.group("suffix") or "").strip().lower()
        if suffix in {"k", "thousand"}:
            value, tol = value * 1e3, tol * 1e3
        elif suffix in {"m", "million"}:
            value, tol = value * 1e6, tol * 1e6
        out.append((raw, value, tol + 1e-9))
    return out


def _is_free(raw: str, value: float) -> bool:
    plain = re.fullmatch(r"\d+", raw)
    return bool(plain) and (value <= FREE_INT_MAX or 1900 <= value <= 2100)


UP_WORDS = {"up", "rose", "rise", "risen", "rising", "increased", "increase", "increasing", "grew", "grow",
            "grown", "growth", "gained", "gain", "higher", "jumped", "climbed", "improved"}
DOWN_WORDS = {"down", "fell", "fall", "fallen", "falling", "dropped", "drop", "decreased", "decrease",
              "declined", "decline", "lower", "slipped", "shrank", "reduced", "worsened", "slowed"}
WORD_RE = re.compile(r"[a-z]+")


def _direction(text_before: str) -> int:
    """+1 / -1 for the direction word closest to the number in the same clause, 0 if none."""
    clause = re.split(r"[,;:.()]", text_before)[-1]
    for word in reversed(WORD_RE.findall(clause.lower())):
        if word in UP_WORDS:
            return 1
        if word in DOWN_WORDS:
            return -1
    return 0


def _is_change(fact: dict) -> bool:
    return fact["id"].endswith("_change")


def check_numbers(text: str, cited: list[dict], all_facts: list[dict]) -> list[str]:
    errors = []
    label_numbers = {v for f in cited for _, v, _ in numbers_in(f["label"])}
    matches = list(NUM_RE.finditer(text))
    parsed = numbers_in(text)
    prev_end = 0
    for m, (raw, value, tol) in zip(matches, parsed):
        window, prev_end = text[prev_end:m.start()], m.end()
        if _is_free(raw, value) or value in label_numbers:
            continue
        hits = [f for f in cited if abs(abs(f["value"]) - value) <= tol]
        if not hits:
            elsewhere = [f["id"] for f in all_facts if abs(abs(f["value"]) - value) <= tol]
            if elsewhere:
                errors.append(f"'{raw}' matches fact {elsewhere[0]} but that fact is not in this item's evidence")
            else:
                errors.append(f"'{raw}' does not match any provided fact")
            continue

        # Direction check: a change figure must be described in the right direction.
        changes = [f for f in hits if _is_change(f) and f["value"] != 0]
        if not changes:
            continue
        sign = m.group("sign")
        stated = {"+": 1, "-": -1, "−": -1}.get(sign, 0) or _direction(window)
        if stated and not any((f["value"] > 0) == (stated > 0) for f in changes):
            actual = "an increase" if changes[0]["value"] > 0 else "a decrease"
            errors.append(f"'{raw}' is described as {'an increase' if stated > 0 else 'a decrease'} "
                          f"but fact {changes[0]['id']} is {actual} ({changes[0]['display']})")
    return errors


# Below this many signals, the model may need non-signal figures to reach 3 insights.
MIN_SIGNALS_FOR_ANCHORING = 3
SUMMARY_MAX_NUMBERS = 5        # a summary is a takeaway, not a list of figures
MAX_GROWTH_ONLY_ACTIONS = 1    # "keep an eye on growth" actions add little; one is enough


def validate(raw_output: str, facts: list[dict], signals: list[dict] | None = None) -> ValidationResult:
    """`facts` are the dicts sent to the model (id, label, value, display); `signals` the flagged issues."""
    try:
        data = parse_json(raw_output)
    except (ValueError, json.JSONDecodeError) as e:
        return ValidationResult(False, [f"Output is not valid JSON: {e}"])

    errors: list[str] = []
    by_id = {f["id"]: f for f in facts}

    def check_item(where: str, item: dict, text_fields: tuple[str, ...]) -> None:
        if not isinstance(item, dict):
            errors.append(f"{where}: must be an object")
            return
        ev = item.get("evidence")
        if not isinstance(ev, list) or not ev:
            errors.append(f"{where}: 'evidence' must be a non-empty list of fact ids")
            ev = []
        unknown = [e for e in ev if e not in by_id]
        if unknown:
            errors.append(f"{where}: unknown fact ids {unknown}")
        cited = [by_id[e] for e in ev if e in by_id]
        for tf in text_fields:
            val = item.get(tf)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"{where}: '{tf}' is missing or empty")
                continue
            errors.extend(f"{where}.{tf}: {e}" for e in check_numbers(val, cited, facts))

    check_item("summary", data.get("summary"), ("text",))

    signal_facts = {fid for sig in (signals or []) for fid in sig["facts"]}
    anchored = len(signals or []) >= MIN_SIGNALS_FOR_ANCHORING

    summary = data.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("text"), str):
        counted = [raw for raw, value, _ in numbers_in(summary["text"]) if not _is_free(raw, value)]
        if len(counted) > SUMMARY_MAX_NUMBERS:
            errors.append(f"summary: uses {len(counted)} figures; use at most {SUMMARY_MAX_NUMBERS}. "
                          "Give the overall result, then name the most urgent issues")
        cited = set(summary.get("evidence") or [])
        for sig in (signals or []):
            if sig["severity"] == "high" and not cited & set(sig["facts"]):
                errors.append(f"summary: must mention the high-severity issue '{sig['type']}' "
                              f"(cite one of {sig['facts'][:2]})")

    growth_facts = {fid for sig in (signals or []) if sig["severity"] == "positive" for fid in sig["facts"]}
    problem_facts = signal_facts - growth_facts
    actions = data.get("actions") if isinstance(data.get("actions"), list) else []
    growth_only = [i for i, a in enumerate(actions) if isinstance(a, dict)
                   and set(a.get("evidence") or []) & growth_facts and not set(a.get("evidence") or []) & problem_facts]
    if len(growth_only) > MAX_GROWTH_ONLY_ACTIONS:
        errors.append(f"actions{growth_only}: at most {MAX_GROWTH_ONLY_ACTIONS} action may be about growth alone; "
                      "drop or merge the others. Three strong actions are better than five weak ones")

    for key, lo, hi, fields_ in (("insights", 3, 5, ("title", "text")),
                                 ("actions", 3, 5, ("action", "owner", "timing", "why"))):
        items = data.get(key)
        if not isinstance(items, list):
            errors.append(f"'{key}' must be a list")
            continue
        if not lo <= len(items) <= hi:
            errors.append(f"'{key}' must have {lo} to {hi} items, got {len(items)}")
        for i, it in enumerate(items):
            check_item(f"{key}[{i}]", it, fields_)
            if anchored and isinstance(it, dict) and not set(it.get("evidence") or []) & signal_facts:
                errors.append(f"{key}[{i}]: must cite at least one fact from a flagged signal "
                              f"(it repeats general figures instead of addressing an issue)")
            if key == "actions" and isinstance(it, dict) and it.get("priority") not in {"high", "medium", "low"}:
                errors.append(f"actions[{i}]: priority must be high, medium or low")

    return ValidationResult(not errors, errors, data if not errors else None)


SEVERITY_RANK = {"high": 0, "medium": 1, "positive": 2}
PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}


def order_by_severity(data: dict, signals: list[dict]) -> dict:
    """Put the most urgent insights and actions first, regardless of the order the model used."""
    rank_of = {}
    for sig in signals:
        for fid in sig["facts"]:
            rank_of[fid] = min(rank_of.get(fid, 9), SEVERITY_RANK[sig["severity"]])

    def insight_rank(item: dict) -> int:
        return min((rank_of.get(e, 9) for e in item["evidence"]), default=9)

    data = dict(data)
    data["insights"] = sorted(data["insights"], key=insight_rank)          # stable sort
    data["actions"] = sorted(data["actions"], key=lambda a: (PRIORITY_RANK[a["priority"]], insight_rank(a)))
    return data


_SIGNED_AFTER_DIRECTION = re.compile(
    r"\b(" + "|".join(sorted(UP_WORDS | DOWN_WORDS, key=len, reverse=True)) + r")(\s+(?:by\s+)?)[+\-−](?=\d)",
    re.IGNORECASE,
)


def tidy_wording(data: dict) -> dict:
    """Drop a sign that repeats a direction word: "fell -24.0 pts" -> "fell 24.0 pts".

    Runs only on validated output, where the direction check has already confirmed the word
    and the sign agree, so this changes wording, never meaning.
    """
    def fix(text):
        return _SIGNED_AFTER_DIRECTION.sub(r"\1\2", text) if isinstance(text, str) else text

    out = dict(data)
    out["summary"] = {**data["summary"], "text": fix(data["summary"]["text"])}
    out["insights"] = [{**i, "title": fix(i["title"]), "text": fix(i["text"])} for i in data["insights"]]
    out["actions"] = [{**a, "action": fix(a["action"]), "why": fix(a["why"])} for a in data["actions"]]
    return out

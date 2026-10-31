import json
from pathlib import Path

import pandas as pd
import pytest

from ai_report.data_loader import DataError, clean, load
from ai_report.llm import GeminiClient, GroqClient, LLMClient, OllamaClient, get_client
from ai_report.metrics import compute, detect_signals
from ai_report.narrative import generate
from ai_report.prompt import select_facts
from ai_report.validation import numbers_in, validate

SAMPLE = Path(__file__).parents[1] / "data" / "sample" / "orders.csv"


@pytest.fixture(scope="module")
def metrics():
    df, _ = load(SAMPLE)
    return compute(df)


@pytest.fixture(scope="module")
def facts(metrics):
    return select_facts(metrics, detect_signals(metrics))


def _good_output(facts):
    f = {x["id"]: x for x in facts}
    anchor = "region.east.on_time"   
    item = lambda fid, text: {"title": "t", "text": text, "evidence": [fid, anchor]}  
    act = lambda fid: {"action": "Do it.", "owner": "Manager", "timing": "this week", "priority": "high", 
                       "why": f"Because of {f[fid]['display']}.", "evidence": [fid, anchor]}
    return {
        "summary": {"text": f"Revenue was {f['revenue']['display']}.",
                    "evidence": ["revenue", "region.east.on_time", "hotspot.south.electronics.return_rate"]},
        "insights": [item("margin", f"Margin was {f['margin']['display']}."),
                     item("orders", f"There were {f['orders']['display']} orders."),
                     item("on_time", f"On time: {f['on_time']['display']}.")],
        "actions": [act("margin"), act("on_time"), act("return_rate")],
    }


# ---------- data ----------
def test_cleaning_removes_duplicates_and_invalid_rows():
    raw = pd.read_csv(SAMPLE)
    df, log = clean(raw)
    assert df["order_id"].is_unique
    assert (df["units"] > 0).all()
    assert set(df["region"].unique()) == {"North", "South", "East", "West", "Central"}
    assert log.rows_out < log.rows_in and log.notes


def test_missing_columns_raise():
    with pytest.raises(DataError):
        clean(pd.DataFrame({"order_id": [1]}))


# ---------- metrics ----------
def test_revenue_matches_independent_calculation(metrics):
    df, _ = load(SAMPLE)
    cur = df[df["month"] == df["month"].max()]
    expected = (cur["units"] * cur["unit_price"] * (1 - cur["discount_pct"] / 100)).sum()
    assert metrics.get("revenue").value == pytest.approx(expected)


def test_planted_issues_are_detected(metrics):
    types = {(s["type"], s["facts"][0]) for s in detect_signals(metrics)}
    assert ("return_hotspot", "hotspot.south.electronics.return_rate") in types
    assert ("delivery_slowdown", "region.east.on_time_change") in types
    assert any(t == "margin_erosion" for t, _ in types)


# ---------- validation ----------
def test_number_parser():
    vals = {round(v, 2) for _, v, _ in numbers_in("Revenue $1,122,285, up 8.8%, about $1.1M, 18 pts")}
    assert {1122285.0, 8.8, 1100000.0, 18.0} <= vals


def test_valid_output_passes(facts):
    assert validate(json.dumps(_good_output(facts)), facts).ok


def test_invented_number_fails(facts):
    out = _good_output(facts)
    out["summary"]["text"] = "Revenue was $9,999,999."
    res = validate(json.dumps(out), facts)
    assert not res.ok and any("does not match" in e for e in res.errors)


def test_uncited_number_fails(facts):
    out = _good_output(facts)
    margin = {x["id"]: x for x in facts}["margin"]["display"]
    out["summary"]["text"] = f"Margin was {margin}."      # evidence still only cites revenue
    res = validate(json.dumps(out), facts)
    assert not res.ok and any("not in this item's evidence" in e for e in res.errors)


def test_unknown_evidence_and_bad_shape_fail(facts):
    out = _good_output(facts)
    out["insights"][0]["evidence"] = ["made_up_id"]
    out["actions"] = out["actions"][:1]
    errors = validate(json.dumps(out), facts).errors
    assert any("unknown fact ids" in e for e in errors)
    assert any("3 to 5" in e for e in errors)


def test_markdown_wrapped_json_is_accepted(facts):
    assert validate("```json\n" + json.dumps(_good_output(facts)) + "\n```", facts).ok


# ---------- LLM module and fallback ----------
class ScriptedClient(LLMClient):
    provider, default_model = "scripted", "test"

    def __init__(self, replies):
        super().__init__()
        self.replies, self.calls = list(replies), []

    def complete(self, system, user):
        self.calls.append(user)
        return self.replies.pop(0)


def test_repair_loop_recovers(metrics, facts):
    good = json.dumps(_good_output(facts))
    client = ScriptedClient(['{"summary": {"text": "Revenue was $5.", "evidence": ["revenue"]}}', good])
    n = generate(metrics, client)
    assert n.passed_checks and not n.fell_back and n.attempts == 2
    assert "failed these checks" in client.calls[1]


def test_falls_back_when_model_keeps_failing(metrics):
    n = generate(metrics, ScriptedClient(["not json"] * 3))
    assert n.fell_back and n.passed_checks and n.source.startswith("offline")


def test_offline_writer_passes_its_own_checks(metrics):
    n = generate(metrics, get_client("offline"))
    assert n.passed_checks and n.attempts == 1 and 3 <= len(n.data["actions"]) <= 5


@pytest.mark.parametrize("cls,env,response", [
    (OllamaClient, {}, {"message": {"content": "{}"}}),
    (GeminiClient, {"GEMINI_API_KEY": "k"}, {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}),
    (GroqClient, {"GROQ_API_KEY": "k"}, {"choices": [{"message": {"content": "{}"}}]}),
])
def test_providers_parse_responses(monkeypatch, cls, env, response):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sent = {}

    class Resp:
        status_code, headers = 200, {}
        def json(self):  
            return response

    def fake_post(url, json, headers, timeout):
        sent.update(url=url, body=json, headers=headers)
        return Resp()

    monkeypatch.setattr("ai_report.llm.requests.post", fake_post)
    assert cls().complete("sys", "user") == "{}"
    assert "sys" in str(sent["body"]) and "user" in str(sent["body"])


# ---------- direction check and .env loading ----------
def _with_summary(facts, text, evidence):
    out = _good_output(facts)
    out["summary"] = {"text": text, "evidence": evidence}
    return validate(json.dumps(out), facts)


def test_wrong_direction_fails(facts):
    f = {x["id"]: x for x in facts}
    assert f["margin_change"]["value"] < 0                     
    pts = f["margin_change"]["display"].lstrip("+-").replace(" pts", "")
    res = _with_summary(facts, f"Gross margin rose {pts} points.", ["margin_change"])
    assert not res.ok and any("described as an increase" in e for e in res.errors)
    assert _with_summary(facts, f"Gross margin fell {pts} points.", ["margin_change"]).ok


def test_wrong_explicit_sign_fails(facts):
    f = {x["id"]: x for x in facts}
    pct = f["revenue_change"]["display"]                    
    assert not _with_summary(facts, f"Revenue changed by -{pct}.", ["revenue_change"]).ok
    assert _with_summary(facts, f"Revenue went up {pct}.", ["revenue_change"]).ok


def test_dotenv_loader(tmp_path, monkeypatch):
    from ai_report.config import load_dotenv
    env = tmp_path / ".env"
    env.write_text("# comment\nAI_REPORT_TEST_A=hello\nexport AI_REPORT_TEST_B='quoted'\nAI_REPORT_TEST_C=\n")
    monkeypatch.delenv("AI_REPORT_TEST_A", raising=False)
    monkeypatch.delenv("AI_REPORT_TEST_B", raising=False)
    monkeypatch.setenv("AI_REPORT_TEST_KEEP", "real")
    env.write_text(env.read_text() + "AI_REPORT_TEST_KEEP=from_file\n")
    load_dotenv(env)
    import os
    assert os.environ["AI_REPORT_TEST_A"] == "hello"
    assert os.environ["AI_REPORT_TEST_B"] == "quoted"
    assert "AI_REPORT_TEST_C" not in os.environ
    assert os.environ["AI_REPORT_TEST_KEEP"] == "real"         
    for k in ("AI_REPORT_TEST_A", "AI_REPORT_TEST_B"):
        monkeypatch.delenv(k)


# ---------- signal anchoring, ordering, cautions ----------
def test_unanchored_insight_fails_when_signals_exist(metrics, facts):
    signals = detect_signals(metrics)
    out = _good_output(facts)
    out["insights"][1]["evidence"] = ["orders"]                  
    res = validate(json.dumps(out), facts, signals)
    assert not res.ok and any("flagged signal" in e for e in res.errors)
    assert validate(json.dumps(_good_output(facts)), facts, signals).ok


def test_output_is_ordered_by_severity(metrics, facts):
    from ai_report.validation import order_by_severity
    signals = detect_signals(metrics)
    growth = next(s for s in signals if s["severity"] == "positive")
    high = next(s for s in signals if s["severity"] == "high")
    data = {"summary": {}, "insights": [{"evidence": growth["facts"]}, {"evidence": high["facts"]}],
            "actions": [{"priority": "low", "evidence": growth["facts"]},
                        {"priority": "high", "evidence": high["facts"]}]}
    ordered = order_by_severity(data, signals)
    assert ordered["insights"][0]["evidence"] == high["facts"]
    assert ordered["actions"][0]["priority"] == "high"


def test_growth_in_problem_area_carries_caution(metrics):
    signals = detect_signals(metrics)
    electronics = next(s for s in signals if s["facts"][0] == "category.electronics.revenue_change")
    assert "caution" in electronics and "hotspot.south.electronics.return_rate" in electronics["caution"]


def test_prompt_sends_cautions(metrics):
    from ai_report.prompt import build_user_prompt, extract_payload
    signals = detect_signals(metrics)
    payload = extract_payload(build_user_prompt(metrics, select_facts(metrics, signals), signals))
    assert any("caution" in s for s in payload["signals"])


# ---------- token budget, reasoning effort, retryable generation failures ----------
def _fake_post(monkeypatch, status, body, sent):
    class Resp:
        status_code, headers, text = status, {}, json.dumps(body)
        def json(self):  # noqa: E301
            return body

    def fake_post(url, json, headers, timeout):
        sent.update(body=json)
        return Resp()
    monkeypatch.setattr("ai_report.llm.requests.post", fake_post)


def test_groq_sends_token_budget_and_reasoning_effort(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    sent = {}
    _fake_post(monkeypatch, 200, {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}, sent)
    GroqClient(model="openai/gpt-oss-120b").complete("s", "u")
    assert sent["body"]["max_completion_tokens"] > 0
    assert sent["body"]["reasoning_effort"] in {"low", "medium", "high"}
    GroqClient(model="some-other-model").complete("s", "u")
    assert "reasoning_effort" not in sent["body"]           # only sent to reasoning models


@pytest.mark.parametrize("status,body", [
    (400, {"error": {"code": "json_validate_failed", "failed_generation": "max completion tokens reached"}}),
    (200, {"choices": [{"message": {"content": "{\"summary\""}, "finish_reason": "length"}]}),
])
def test_truncated_generation_is_retryable(monkeypatch, status, body):
    from ai_report.llm import LLMOutputError
    monkeypatch.setenv("GROQ_API_KEY", "k")
    _fake_post(monkeypatch, status, body, {})
    with pytest.raises(LLMOutputError):
        GroqClient().complete("s", "u")


def test_pipeline_retries_after_generation_failure(metrics, facts):
    from ai_report.llm import LLMOutputError

    class FlakyClient(ScriptedClient):
        def complete(self, system, user):
            self.calls.append(user)
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

    client = FlakyClient([LLMOutputError("cut off"), json.dumps(_good_output(facts))])
    n = generate(metrics, client)
    assert n.passed_checks and not n.fell_back and n.attempts == 2


# ---------- summary and action quality rules ----------
def test_summary_must_name_urgent_issues_and_stay_short(metrics, facts):
    signals = detect_signals(metrics)
    f = {x["id"]: x for x in facts}
    out = _good_output(facts)
    out["summary"] = {"text": f"Revenue was {f['revenue']['display']}.", "evidence": ["revenue"]}
    errors = validate(json.dumps(out), facts, signals).errors
    assert any("high-severity issue 'delivery_slowdown'" in e for e in errors)
    ids = ["revenue", "revenue_change", "orders", "orders_change", "aov", "margin"]
    out["summary"] = {"text": " ".join(f[i]["display"] for i in ids),
                      "evidence": ids + ["region.east.on_time", "hotspot.south.electronics.return_rate"]}
    errors = validate(json.dumps(out), facts, signals).errors
    assert any("use at most" in e for e in errors)


def test_at_most_one_growth_only_action(metrics, facts):
    signals = detect_signals(metrics)
    growth = [s for s in signals if s["severity"] == "positive"]
    out = _good_output(facts)
    for i, sig in enumerate(growth[:2]):
        out["actions"][i] = {"action": "Track it.", "owner": "Manager", "timing": "weekly", "priority": "low",
                             "why": "Growth.", "evidence": [sig["facts"][0]]}
    errors = validate(json.dumps(out), facts, signals).errors
    assert any("growth alone" in e for e in errors)


# ---------- slow answers and explicit summary requirements ----------
def test_read_timeout_is_retryable(monkeypatch):
    import requests
    from ai_report.llm import LLMOutputError
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    calls = []

    def slow_post(url, json, headers, timeout):
        calls.append(1)
        raise requests.ReadTimeout("read timed out")
    monkeypatch.setattr("ai_report.llm.requests.post", slow_post)
    with pytest.raises(LLMOutputError):
        GeminiClient().complete("s", "u")
    assert len(calls) == 1         


def test_user_prompt_lists_required_summary_facts(metrics):
    from ai_report.prompt import build_user_prompt
    signals = detect_signals(metrics)
    prompt = build_user_prompt(metrics, select_facts(metrics, signals), signals)
    assert "region.east.on_time_change" in prompt.split("FACTS_JSON>>>")[1]
    assert "hotspot.south.electronics.return_rate" in prompt.split("FACTS_JSON>>>")[1]


def test_model_setting_does_not_leak_across_providers(monkeypatch):
    import dataclasses
    import ai_report.llm as llm
    fake = dataclasses.replace(llm.SETTINGS, llm_provider="groq", llm_model="openai/gpt-oss-120b")
    monkeypatch.setattr(llm, "SETTINGS", fake)
    assert llm.get_client("groq").model == "openai/gpt-oss-120b"
    assert llm.get_client("gemini").model == llm.GeminiClient.default_model
    assert llm.get_client("gemini", "gemini-x").model == "gemini-x"


def test_tidy_wording_removes_redundant_sign():
    from ai_report.validation import tidy_wording
    item = lambda t: {"title": t, "text": t, "action": t, "why": t, "evidence": []} 
    data = {"summary": {"text": "Margin dropped -4.2 pts; revenue rose by +8.8%.", "evidence": []},
            "insights": [item("On-time fell -24.0 pts to 55.4%.")], "actions": [item("Change of -5.7 pts.")]}
    out = tidy_wording(data)
    assert out["summary"]["text"] == "Margin dropped 4.2 pts; revenue rose by 8.8%."
    assert out["insights"][0]["text"] == "On-time fell 24.0 pts to 55.4%."
    assert out["actions"][0]["why"] == "Change of -5.7 pts."       


def test_prompt_forbids_numeric_targets():
    from ai_report.prompt import SYSTEM_PROMPT
    assert "numeric targets" in SYSTEM_PROMPT


# ---------- pre-release checks ----------
def test_excel_upload_gives_same_figures(tmp_path, metrics):
    xlsx = tmp_path / "orders.xlsx"
    pd.read_csv(SAMPLE).to_excel(xlsx, index=False)
    df, _ = load(xlsx)
    assert compute(df).get("revenue").value == pytest.approx(metrics.get("revenue").value)


def test_one_month_of_data_gives_clear_error():
    raw = pd.read_csv(SAMPLE)
    raw = raw[raw["order_date"].str.startswith("2026-10")]
    with pytest.raises(DataError, match="two months"):
        clean(raw)


def test_footer_says_synthetic_only_for_sample(tmp_path):
    from ai_report.pipeline import build_report
    assert "synthetic" in build_report(SAMPLE, get_client("offline")).html
    copy = tmp_path / "real_export.csv"
    copy.write_bytes(SAMPLE.read_bytes())
    assert "synthetic" not in build_report(copy, get_client("offline")).html


def test_app_loads_secrets_before_settings():
    src = (Path(__file__).parents[1] / "app.py").read_text()
    assert src.index("st.secrets") < src.index("from ai_report")
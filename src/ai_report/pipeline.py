"""End-to-end: load -> clean -> calculate -> write (LLM) -> validate -> render."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import SAMPLE_DATA
from .data_loader import CleaningLog, load
from .llm import LLMClient, get_client
from .metrics import MetricSet, compute
from .narrative import Narrative, generate
from .render import render_html, to_pdf


@dataclass
class ReportResult:
    metrics: MetricSet
    narrative: Narrative
    log: CleaningLog
    html: str


def build_report(data_path: str | Path, client: LLMClient | None = None, context: str = "") -> ReportResult:
    df, log = load(data_path)
    ms = compute(df)
    narrative = generate(ms, client or get_client(), context=context)
    is_sample = Path(data_path).resolve() == SAMPLE_DATA.resolve()
    return ReportResult(ms, narrative, log, render_html(ms, narrative, log, is_sample=is_sample))


def save(result: ReportResult, out_dir: str | Path, stem: str = "report", pdf: bool = True) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"html": out / f"{stem}.html", "json": out / f"{stem}.json"}
    paths["html"].write_text(result.html, encoding="utf-8")
    paths["json"].write_text(json.dumps({
        "period": result.metrics.current_period,
        "source": result.narrative.source,
        "fell_back": result.narrative.fell_back,
        "attempts": result.narrative.attempts,
        "issues": result.narrative.issues,
        "narrative": result.narrative.data,
        "facts": result.narrative.facts,
        "signals": result.narrative.signals,
    }, indent=2), encoding="utf-8")
    if pdf:
        paths["pdf"] = to_pdf(result.html, out / f"{stem}.pdf")
    return paths

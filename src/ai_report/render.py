"""Turn figures + validated narrative into HTML, and HTML into PDF."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .charts import all_charts
from .config import SETTINGS
from .data_loader import CleaningLog
from .metrics import MetricSet
from .narrative import Narrative

TEMPLATES = Path(__file__).parent / "templates"

# PDF and font libraries log harmless rendering notes; show only real errors (CLI and Streamlit).
for _noisy in ("fontTools", "weasyprint", "matplotlib"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# (fact id, change fact id, label, higher is better)
KPIS = [
    ("revenue", "revenue_change", "Net revenue", True),
    ("orders", "orders_change", "Orders", True),
    ("aov", "aov_change", "Average order value", True),
    ("margin", "margin_change", "Gross margin", True),
    ("return_rate", "return_rate_change", "Return rate", False),
    ("on_time", "on_time_change", None, True),
]


def _kpis(ms: MetricSet) -> list[dict]:
    f = ms.by_id()
    out = []
    for fid, cid, label, higher_better in KPIS:
        ch = f[cid]
        flat = abs(ch.value) < 0.05
        up = ch.value > 0
        tone = "flat" if flat else ("good" if up == higher_better else "bad")
        change = ch.display if ch.unit == "pts" else f"{ch.value:+.1f}%"
        out.append({"label": label or f[fid].label, "value": f[fid].display, "change": change,
                    "tone": tone, "arrow": "=" if flat else ("▲" if up else "▼")})
    return out


def render_html(ms: MetricSet, narrative: Narrative, log: CleaningLog, is_sample: bool = False) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]))
    facts = {f["id"]: f for f in narrative.facts}
    n = narrative.data
    items = [n["summary"], *n["insights"], *n["actions"]]
    used_ids = list(dict.fromkeys(e for it in items for e in it["evidence"]))

    if narrative.fell_back:
        notice = ("The language model did not return a report that passed the accuracy checks, so this text "
                  "was produced by the built-in rule-based writer. Figures are unaffected.")
        source_label = f"{narrative.source} (fallback)"
    elif narrative.source.startswith("offline"):
        notice = "Demo mode: text produced by the built-in rule-based writer, not a language model."
        source_label = "Rule-based writer (no model)"
    else:
        notice, source_label = "", narrative.source

    return env.get_template("report.html.j2").render(
        css=(TEMPLATES / "report.css").read_text(),
        title=SETTINGS.report_title, company=SETTINGS.company_label,
        period=ms.current_period, previous_period=ms.previous_period,
        generated=datetime.now().strftime("%d %b %Y, %H:%M"),
        rows=f"{log.rows_out:,}", source_label=source_label, notice=notice,
        n=n, facts=facts, kpis=_kpis(ms), charts=all_charts(ms),
        alert_pts=f"{SETTINGS.on_time_drop_alert_pts:g}",
        passed=narrative.passed_checks, log=log, is_sample=is_sample,
        cited_count=sum(1 for it in items if it["evidence"]), statement_count=len(items),
        used_facts=[facts[i] for i in used_ids],
    )


def to_pdf(html: str, out_path: str | Path) -> Path:
    from weasyprint import HTML  # imported lazily: needs system Pango libraries

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(TEMPLATES)).write_pdf(out_path)
    return out_path

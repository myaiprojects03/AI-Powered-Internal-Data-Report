"""Charts as inline SVG (crisp in both the HTML report and the PDF)."""
from __future__ import annotations

import io
import itertools
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .config import SETTINGS  # noqa: E402
from .metrics import MetricSet  # noqa: E402

CURRENT = "#2a78d6"     # categorical slot 1
PRIOR = "#b9b8b2"       # muted neutral for the comparison period
INK = "#2b2a27"
MUTED = "#6f6e69"
GRID = "#e6e5e1"
ALERT = "#e34948"

# Use the first font installed on this machine (Inter on Linux, Helvetica/Arial on macOS).
_INSTALLED = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Inter", "Helvetica Neue", "Helvetica", "Arial") if f in _INSTALLED), "DejaVu Sans")

plt.rcParams.update({
    "font.family": FONT, "font.size": 9, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "svg.fonttype": "none",
})


def _money(x: float, _=None) -> str:
    s = SETTINGS.currency_symbol
    return f"{s}{x / 1e6:.1f}M" if abs(x) >= 1e6 else f"{s}{x / 1e3:.0f}k"


_chart_ids = itertools.count(1)


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    svg = buf.getvalue()
    svg = svg[svg.find("<svg"):]
    # Several charts share one HTML page: give each chart's element ids a unique prefix.
    p = f"c{next(_chart_ids)}_"
    svg = re.sub(r'\bid="', f'id="{p}', svg)
    svg = svg.replace('href="#', f'href="#{p}').replace("url(#", f"url(#{p}")
    return svg


def _frame(title: str):
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold", y=1.1)
    return fig, ax


def weekly_revenue(ms: MetricSet) -> str:
    w = ms.tables["weekly"].iloc[1:-1] if len(ms.tables["weekly"]) > 4 else ms.tables["weekly"]  # drop partial weeks
    fig, ax = _frame("Weekly net revenue")
    ax.plot(w["order_date"], w["revenue"], color=CURRENT, linewidth=2, marker="o", markersize=4)
    ax.yaxis.set_major_formatter(FuncFormatter(_money))
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=8)
    fig.autofmt_xdate(rotation=0, ha="center")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    return _svg(fig)


def _grouped(ms: MetricSet, dim: str, cur_col: str, prev_col: str, title: str, fmt, highlight=None) -> str:
    t = ms.tables["breakdown"]
    t = t[t["dimension"] == dim].sort_values(cur_col, ascending=False)
    fig, ax = _frame(title)
    x = range(len(t))
    w = 0.36
    ax.bar([i - w / 2 - 0.01 for i in x], t[prev_col], w, color=PRIOR, label=ms.previous_period.split()[0])
    bars = ax.bar([i + w / 2 + 0.01 for i in x], t[cur_col], w, color=CURRENT, label=ms.current_period.split()[0])
    if highlight is not None:
        for b, name in zip(bars, t["name"]):
            if name == highlight:
                b.set_color(ALERT)
    ax.set_xticks(list(x), t["name"])
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncols=2, bbox_to_anchor=(-0.02, 0.98),
              handlelength=1, handleheight=0.8, borderaxespad=0)
    return _svg(fig)


def region_revenue(ms: MetricSet) -> str:
    return _grouped(ms, "region", "revenue", "revenue_prev", "Net revenue by region", _money)


def region_on_time(ms: MetricSet, worst: str | None) -> str:
    return _grouped(ms, "region", "on_time", "on_time_prev", "On-time delivery by region",
                    lambda v, _=None: f"{v:.0f}%", highlight=worst)


def category_returns(ms: MetricSet) -> str:
    t = ms.tables["breakdown"]
    t = t[t["dimension"] == "category"].sort_values("return_rate")
    overall = ms.get("return_rate").value
    fig, ax = _frame(f"Return rate by category, {ms.current_period.split()[0]}")
    ax.barh(t["name"], t["return_rate"], height=0.5, color=CURRENT)
    for i, v in enumerate(t["return_rate"]):
        ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=8, color=INK)
    ax.axvline(overall, color=MUTED, linestyle="--", linewidth=1)
    ax.text(overall, -0.65, f" all categories {overall:.1f}%", fontsize=8, color=MUTED, va="top")
    ax.set_ylim(-0.9, len(t) - 0.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _=None: f"{v:.0f}%"))
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(t["return_rate"].max(), overall) * 1.25)
    return _svg(fig)


def all_charts(ms: MetricSet) -> list[str]:
    reg = ms.tables["breakdown"]
    reg = reg[reg["dimension"] == "region"]
    worst = reg.assign(d=reg["on_time"] - reg["on_time_prev"]).sort_values("d").iloc[0]
    worst_name = worst["name"] if worst["d"] <= -SETTINGS.on_time_drop_alert_pts else None
    return [weekly_revenue(ms), region_revenue(ms), region_on_time(ms, worst_name), category_returns(ms)]

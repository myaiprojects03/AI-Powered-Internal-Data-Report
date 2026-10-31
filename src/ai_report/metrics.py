"""All figures are calculated here, in code. The LLM never computes a number.

Output is a flat list of `Fact` objects, each with a stable ID. The LLM must cite
these IDs as evidence, and the validator checks every number it writes against them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from .config import SETTINGS

UNIT_DECIMALS = {"currency": 0, "pct": 1, "pts": 1, "days": 1, "count": 0}


@dataclass(frozen=True)
class Fact:
    id: str
    label: str
    value: float
    unit: str                      # currency | pct | pts | days | count
    group: str = "overall"         # overall | region | category | segment | signal

    @property
    def decimals(self) -> int:
        return UNIT_DECIMALS[self.unit]

    @property
    def display(self) -> str:
        return format_value(self.value, self.unit)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["value"] = round(self.value, self.decimals)
        d["display"] = self.display
        return d


def format_value(value: float, unit: str) -> str:
    if unit == "currency":
        sign = "-" if value < 0 else ""
        return f"{sign}{SETTINGS.currency_symbol}{abs(value):,.0f}"
    if unit == "pct":
        return f"{value:.1f}%"
    if unit == "pts":
        return f"{value:+.1f} pts"
    if unit == "days":
        return f"{value:.1f} days"
    return f"{value:,.0f}"


@dataclass
class MetricSet:
    current_period: str
    previous_period: str
    facts: list[Fact] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def by_id(self) -> dict[str, Fact]:
        return {f.id: f for f in self.facts}

    def get(self, fact_id: str) -> Fact:
        return self.by_id()[fact_id]


def _pct_change(cur: float, prev: float) -> float:
    return 0.0 if prev == 0 else (cur - prev) / prev * 100


def _summ(d: pd.DataFrame) -> dict[str, float]:
    revenue = d["revenue"].sum()
    delivered = d["delivery_days"].notna()
    return {
        "revenue": revenue,
        "orders": float(len(d)),
        "aov": revenue / len(d) if len(d) else 0.0,
        "margin": (revenue - d["cost"].sum()) / revenue * 100 if revenue else 0.0,
        "discount": (d["gross_revenue"] - d["revenue"]).sum() / d["gross_revenue"].sum() * 100,
        "return_rate": d["returned"].mean() * 100,
        "delivery": d.loc[delivered, "delivery_days"].mean(),
        "on_time": d.loc[delivered, "on_time"].mean() * 100,
    }


def _slug(text: str) -> str:
    return text.lower().replace(" ", "_")


def compute(df: pd.DataFrame) -> MetricSet:
    months = sorted(df["month"].unique())
    cur_m, prev_m = months[-1], months[-2]
    cur, prev = df[df["month"] == cur_m], df[df["month"] == prev_m]
    ms = MetricSet(current_period=cur_m.strftime("%B %Y"), previous_period=prev_m.strftime("%B %Y"))
    add = ms.facts.append

    c, p = _summ(cur), _summ(prev)
    add(Fact("revenue", "Net revenue", c["revenue"], "currency"))
    add(Fact("revenue_prev", "Net revenue, previous month", p["revenue"], "currency"))
    add(Fact("revenue_change", "Net revenue change vs previous month", _pct_change(c["revenue"], p["revenue"]), "pct"))
    add(Fact("orders", "Orders", c["orders"], "count"))
    add(Fact("orders_change", "Orders change vs previous month", _pct_change(c["orders"], p["orders"]), "pct"))
    add(Fact("aov", "Average order value", c["aov"], "currency"))
    add(Fact("aov_change", "Average order value change", _pct_change(c["aov"], p["aov"]), "pct"))
    add(Fact("margin", "Gross margin", c["margin"], "pct"))
    add(Fact("margin_prev", "Gross margin, previous month", p["margin"], "pct"))
    add(Fact("margin_change", "Gross margin change", c["margin"] - p["margin"], "pts"))
    add(Fact("discount", "Average discount given", c["discount"], "pct"))
    add(Fact("discount_change", "Average discount change", c["discount"] - p["discount"], "pts"))
    add(Fact("return_rate", "Return rate", c["return_rate"], "pct"))
    add(Fact("return_rate_change", "Return rate change", c["return_rate"] - p["return_rate"], "pts"))
    add(Fact("delivery_days", "Average delivery time", c["delivery"], "days"))
    add(Fact("on_time", f"On-time delivery (within {SETTINGS.on_time_threshold_days:g} days)", c["on_time"], "pct"))
    add(Fact("on_time_change", "On-time delivery change", c["on_time"] - p["on_time"], "pts"))

    rows = []
    for dim in ("region", "category", "segment"):
        for name in sorted(df[dim].unique()):
            cs, ps = _summ(cur[cur[dim] == name]), _summ(prev[prev[dim] == name])
            key, lab = f"{dim}.{_slug(name)}", f"{name} ({dim})"
            add(Fact(f"{key}.revenue", f"Net revenue, {lab}", cs["revenue"], "currency", dim))
            add(Fact(f"{key}.revenue_change", f"Net revenue change, {lab}", _pct_change(cs["revenue"], ps["revenue"]), "pct", dim))
            add(Fact(f"{key}.margin", f"Gross margin, {lab}", cs["margin"], "pct", dim))
            add(Fact(f"{key}.margin_change", f"Gross margin change, {lab}", cs["margin"] - ps["margin"], "pts", dim))
            add(Fact(f"{key}.return_rate", f"Return rate, {lab}", cs["return_rate"], "pct", dim))
            add(Fact(f"{key}.on_time", f"On-time delivery, {lab}", cs["on_time"], "pct", dim))
            add(Fact(f"{key}.on_time_change", f"On-time delivery change, {lab}", cs["on_time"] - ps["on_time"], "pts", dim))
            if dim == "segment":
                add(Fact(f"{key}.discount", f"Average discount, {lab}", cs["discount"], "pct", dim))
                add(Fact(f"{key}.discount_change", f"Average discount change, {lab}", cs["discount"] - ps["discount"], "pts", dim))
            rows.append({"dimension": dim, "name": name, "revenue": cs["revenue"], "revenue_prev": ps["revenue"],
                         "margin": cs["margin"], "return_rate": cs["return_rate"],
                         "on_time": cs["on_time"], "on_time_prev": ps["on_time"]})
    ms.tables["breakdown"] = pd.DataFrame(rows)

    # Region x category hot spots for returns: the level where problems usually hide.
    hot = (cur.groupby(["region", "category"]).agg(orders=("order_id", "size"), return_rate=("returned", "mean"))
           .reset_index())
    hot["return_rate"] *= 100
    hot = hot[hot["orders"] >= SETTINGS.hotspot_min_orders]
    for _, r in hot.iterrows():
        if r["return_rate"] >= SETTINGS.return_rate_alert_multiple * c["return_rate"]:
            add(Fact(f"hotspot.{_slug(r['region'])}.{_slug(r['category'])}.return_rate",
                     f"Return rate, {r['category']} in {r['region']}", r["return_rate"], "pct", "signal"))
            add(Fact(f"hotspot.{_slug(r['region'])}.{_slug(r['category'])}.orders",
                     f"Orders, {r['category']} in {r['region']}", float(r["orders"]), "count", "signal"))

    weekly = (df.set_index("order_date").resample("W-SUN")["revenue"].sum().reset_index())
    ms.tables["weekly"] = weekly
    ms.tables["hotspots"] = hot.sort_values("return_rate", ascending=False)
    return ms


def detect_signals(ms: MetricSet) -> list[dict]:
    """Rule-based flags computed before the LLM call. They steer attention, not wording."""
    f = ms.by_id()
    s = SETTINGS
    signals: list[dict] = []
    for fid, fact in f.items():
        if fid.startswith("hotspot.") and fid.endswith(".return_rate"):
            signals.append({"type": "return_hotspot", "severity": "high", "facts": [fid, fid.replace(".return_rate", ".orders"), "return_rate"]})
        if fact.group in {"region", "category", "segment"}:
            if fid.endswith(".on_time_change") and fact.value <= -s.on_time_drop_alert_pts:
                signals.append({"type": "delivery_slowdown", "severity": "high", "facts": [fid, fid.replace("_change", "")]})
            # Only flag a slice whose margin fell clearly more than the business overall.
            if fid.endswith(".margin_change") and fact.value <= f["margin_change"].value - s.margin_drop_alert_pts:
                signals.append({"type": "margin_erosion", "severity": "medium", "facts": [fid, fid.replace("_change", "")]})
            if fid.endswith(".revenue_change") and fact.value >= s.growth_alert_pct:
                signals.append({"type": "growth", "severity": "positive", "facts": [fid, fid.replace("_change", "")]})
    if f["margin_change"].value <= -s.margin_drop_alert_pts:
        # Attribute the overall drop to the segment with the largest margin fall.
        segs = [x for x in f.values() if x.group == "segment" and x.id.endswith(".margin_change")]
        worst = min(segs, key=lambda x: x.value).id.rsplit(".", 1)[0]
        signals.append({"type": "margin_erosion", "severity": "medium",
                        "facts": ["margin_change", "margin", "discount_change",
                                  f"{worst}.margin_change", f"{worst}.discount_change"]})
    # Growth in an area that also has an urgent problem must not be read as "push harder here".
    high_areas = [(sig, _areas(sig["facts"][0])) for sig in signals if sig["severity"] == "high"]
    for sig in signals:
        if sig["type"] == "growth":
            overlaps = [h["facts"][0] for h, areas in high_areas if areas & _areas(sig["facts"][0])]
            if overlaps:
                sig["caution"] = ("This area also has an unresolved high-severity issue "
                                  f"({', '.join(overlaps)}). Do not recommend growing or promoting it.")
    order = {"high": 0, "medium": 1, "positive": 2}
    return sorted(signals, key=lambda x: (order[x["severity"]], -abs(f[x["facts"][0]].value)))


def _areas(fact_id: str) -> set[str]:
    """Business areas named in a fact id, e.g. hotspot.south.electronics.return_rate -> {south, electronics}."""
    parts = fact_id.split(".")
    return set(parts[1:-1]) if len(parts) > 2 else set()

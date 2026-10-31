"""Rule-based report writer used when no LLM is available.

It consumes exactly the same prompt payload a model would, and returns the same JSON
shape, so it exercises the full pipeline (including validation) without a model.
"""
from __future__ import annotations

import json

from .prompt import extract_payload


def write_from_prompt(user_prompt: str) -> str:
    p = extract_payload(user_prompt)
    f = {x["id"]: x for x in p["facts"]}
    d = lambda i: f[i]["display"]  # noqa: E731
    signals = p["signals"]
    period = p["report_period"]

    def direction(fid: str, up: str, down: str) -> str:
        return down if d(fid).startswith("-") else up

    def strip_sign(fid: str) -> str:
        return d(fid).lstrip("+-").replace(" pts", "")

    def name_of(fid: str) -> str:
        return f[fid]["label"].split(", ", 1)[1].split(" (")[0]

    insights, actions = [], []
    for s in signals:
        ids = s["facts"]
        if s["type"] == "return_hotspot":
            what = f[ids[0]]["label"].split(", ", 1)[1]
            insights.append({
                "title": f"Returns are high for {what}",
                "text": f"{d(ids[0])} of these orders were returned ({d(ids[1])} orders), "
                        f"against {d(ids[2])} across the business.",
                "evidence": ids})
            actions.append({
                "action": f"Review the returned orders for {what} and record the return reason for each.",
                "owner": "Quality or customer service lead", "timing": "this week", "priority": "high",
                "why": f"The return rate here is {d(ids[0])}, far above the {d(ids[2])} average.",
                "evidence": [ids[0], ids[2]]})
        elif s["type"] == "delivery_slowdown":
            region = f[ids[0]]["label"].split(", ", 1)[1].replace(" (region)", "")
            insights.append({
                "title": f"Deliveries slowed sharply in {region}",
                "text": f"Only {d(ids[1])} of {region} orders arrived on time, "
                        f"down {strip_sign(ids[0])} percentage points on last month.",
                "evidence": ids})
            actions.append({
                "action": f"Check carrier performance and warehouse dispatch times for the {region} region.",
                "owner": "Logistics manager", "timing": "this week", "priority": "high",
                "why": f"On-time delivery in {region} is down to {d(ids[1])}.",
                "evidence": [ids[1]]})
        elif s["type"] == "margin_erosion" and ids[0] == "margin_change":
            seg = f[ids[3]]["label"].split(", ", 1)[1].replace(" (segment)", "")
            insights.append({
                "title": "Discounts are eating into margin",
                "text": f"Gross margin fell to {d('margin')}, down {strip_sign('margin_change')} points, while average discount rose "
                        f"{strip_sign('discount_change')} percentage points. The biggest rise was in {seg}, "
                        f"where discount went up {strip_sign(ids[4])} points.",
                "evidence": ["margin", "margin_change", "discount_change", ids[4]]})
            actions.append({
                "action": f"Set a maximum discount level for {seg} deals and require approval above it.",
                "owner": "Sales director", "timing": "before next month's report", "priority": "medium",
                "why": f"{seg} margin fell {strip_sign(ids[3])} points as discounts grew.",
                "evidence": [ids[3]]})
        elif s["type"] == "growth" and len([i for i in insights if i["title"].startswith("Strong growth")]) < 1:
            name = name_of(ids[0])
            insights.append({
                "title": f"Strong growth in {name}",
                "text": f"Revenue in {name} grew {d(ids[0])} to {d(ids[1])}.",
                "evidence": ids})
            actions.append({
                "action": f"Confirm stock levels can support continued demand in {name}.",
                "owner": "Inventory planner", "timing": "next two weeks", "priority": "low",
                "why": f"Revenue in {name} grew {d(ids[0])}.",
                "evidence": [ids[0]]})

    if len(insights) < 3:
        insights.append({
            "title": "Orders and order value",
            "text": f"Orders changed by {d('orders_change')} and average order value is {d('aov')}.",
            "evidence": ["orders_change", "aov"]})

    urgent, urgent_ids = [], []
    for s in signals:
        if s["severity"] != "high":
            continue
        ids = s["facts"]
        if s["type"] == "delivery_slowdown":
            urgent.append(f"on-time delivery in {name_of(ids[1])} fell to {d(ids[1])}")
            urgent_ids.append(ids[1])
        elif s["type"] == "return_hotspot":
            category, region = f[ids[0]]["label"].split(", ", 1)[1].split(" in ", 1)
            urgent.append(f"{d(ids[0])} of {category} orders in {region} were returned")
            urgent_ids.append(ids[0])
    urgent = urgent[:2]
    summary = (
        f"Revenue for {period} was {d('revenue')}, {direction('revenue_change', 'up', 'down')} "
        f"{strip_sign('revenue_change')} on last month, but gross margin fell to {d('margin')}."
    )
    if urgent:
        summary += f" {'Two problems need' if len(urgent) > 1 else 'One problem needs'} action this week: " \
                   + " and ".join(urgent) + "."
    out = {
        "summary": {"text": summary, "evidence": ["revenue", "revenue_change", "margin", *urgent_ids[:2]]},
        "insights": insights[:5],
        "actions": actions[:5],
    }
    return json.dumps(out)

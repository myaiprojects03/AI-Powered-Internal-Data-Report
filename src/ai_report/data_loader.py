"""Load and clean the raw export. Every change is logged so it can be shown in the report."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import SETTINGS


class DataError(ValueError):
    """Raised when the input cannot be used to build a report."""


@dataclass
class CleaningLog:
    rows_in: int = 0
    rows_out: int = 0
    notes: list[str] = field(default_factory=list)

    def add(self, note: str) -> None:
        self.notes.append(note)


def read_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise DataError(f"Unsupported file type: {path.suffix} (use .csv or .xlsx)")


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    log = CleaningLog(rows_in=len(raw))
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in SETTINGS.required_columns if c not in df.columns]
    if missing:
        raise DataError(f"Missing required columns: {', '.join(missing)}")

    before = len(df)
    df = df.drop_duplicates()
    if (n := before - len(df)):
        log.add(f"Removed {n} exact duplicate rows.")

    before = len(df)
    df = df.drop_duplicates(subset="order_id", keep="first")
    if (n := before - len(df)):
        log.add(f"Removed {n} rows with a repeated order ID.")

    for col in ("region", "category", "segment"):
        normalised = df[col].astype(str).str.strip().str.title()
        if (n := int((normalised != df[col]).sum())):
            log.add(f"Standardised {n} '{col}' labels (spacing/capitalisation).")
        df[col] = normalised

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    for col in ("units", "unit_price", "unit_cost", "discount_pct", "delivery_days"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    flag = df["returned"].astype(str).str.strip().str.upper()
    if (n := int((~flag.isin(["Y", "N"])).sum())):
        log.add(f"Normalised {n} non-standard return flags (e.g. 'yes').")
    df["returned"] = flag.isin(["Y", "YES", "TRUE", "1"])

    bad = df["order_date"].isna() | df["units"].isna() | (df["units"] <= 0) | df["unit_price"].isna()
    if (n := int(bad.sum())):
        log.add(f"Excluded {n} rows with missing dates or invalid quantities/prices.")
    df = df[~bad]

    if (n := int(df["delivery_days"].isna().sum())):
        log.add(f"{n} orders have no delivery time; they are excluded from delivery figures only.")

    df["month"] = df["order_date"].dt.to_period("M")
    df["gross_revenue"] = df["units"] * df["unit_price"]
    df["revenue"] = df["gross_revenue"] * (1 - df["discount_pct"] / 100)
    df["cost"] = df["units"] * df["unit_cost"]
    df["on_time"] = df["delivery_days"] <= SETTINGS.on_time_threshold_days

    if df["month"].nunique() < 2:
        raise DataError("At least two months of data are needed to compare periods.")

    log.rows_out = len(df)
    return df.reset_index(drop=True), log


def load(path: str | Path) -> tuple[pd.DataFrame, CleaningLog]:
    return clean(read_any(path))

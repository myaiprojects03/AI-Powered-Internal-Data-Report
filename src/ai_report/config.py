"""Central configuration. Everything tunable lives here or in environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file. Real environment variables always win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.removeprefix("export ").strip(), value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key, value)


load_dotenv(PROJECT_ROOT / ".env")

SAMPLE_DATA = PROJECT_ROOT / "data" / "sample" / "orders.csv"


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    # Report
    report_title: str = _env("REPORT_TITLE", "Monthly Operations Report")
    company_label: str = _env("COMPANY_LABEL", "Sample Distribution Co. (anonymised)")
    currency_symbol: str = _env("CURRENCY_SYMBOL", "$")
    on_time_threshold_days: float = float(_env("ON_TIME_DAYS", "5"))

    # Signal thresholds used to flag issues before the LLM sees anything
    return_rate_alert_multiple: float = 1.8   # segment return rate vs overall
    on_time_drop_alert_pts: float = 10.0      # month-on-month drop in on-time rate
    margin_drop_alert_pts: float = 2.0        # month-on-month drop in gross margin
    growth_alert_pct: float = 15.0            # standout revenue growth
    hotspot_min_orders: int = 50              # ignore tiny slices (noise)

    # LLM
    llm_provider: str = _env("LLM_PROVIDER", "offline")    # offline | ollama | gemini | groq
    llm_model: str = _env("LLM_MODEL", "")                 # blank = provider default
    llm_temperature: float = float(_env("LLM_TEMPERATURE", "0.2"))
    llm_timeout_s: int = int(_env("LLM_TIMEOUT_S", "180"))                 # per call; reasoning models are slower
    llm_max_attempts: int = int(_env("LLM_MAX_ATTEMPTS", "3"))
    llm_max_tokens: int = int(_env("LLM_MAX_TOKENS", "8192"))              # output budget incl. reasoning
    llm_reasoning_effort: str = _env("LLM_REASONING_EFFORT", "medium")     # low | medium | high (reasoning models)
    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")

    required_columns: tuple[str, ...] = field(default=(
        "order_id", "order_date", "region", "category", "segment",
        "units", "unit_price", "unit_cost", "discount_pct", "returned", "delivery_days",
    ))


SETTINGS = Settings()

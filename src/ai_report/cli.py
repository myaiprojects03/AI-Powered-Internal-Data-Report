"""Command line entry point.

    python -m ai_report.cli --data data/sample/orders.csv --provider groq --out output
"""
from __future__ import annotations

import argparse
import logging
import sys

from .data_loader import DataError
from .llm import PROVIDERS, LLMError, get_client
from .pipeline import build_report, save


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the AI-written internal data report.")
    ap.add_argument("--data", required=True, help="CSV or Excel export")
    ap.add_argument("--provider", choices=list(PROVIDERS), help="Overrides LLM_PROVIDER")
    ap.add_argument("--model", help="Overrides LLM_MODEL")
    ap.add_argument("--context", default="", help="One line of business context for the writer")
    ap.add_argument("--out", default="output")
    ap.add_argument("--stem", default="report")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        client = get_client(args.provider, args.model)
        result = build_report(args.data, client, args.context)
        paths = save(result, args.out, args.stem, pdf=not args.no_pdf)
    except (DataError, LLMError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    n = result.narrative
    print(f"Narrative: {n.source}{' (fallback)' if n.fell_back else ''}, checks passed: {n.passed_checks}")
    for issue in n.issues:
        print(f"  note: {issue}")
    for kind, p in paths.items():
        print(f"{kind.upper():5} {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

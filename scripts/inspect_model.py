#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hecras_review.inspector import compare_plans, inspect_zip, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="HEC-RAS Review Tool v0.4 - read-only model review engine")
    parser.add_argument("zip_model", help="Path to HEC-RAS model ZIP")
    parser.add_argument("--existing", help="Existing/baseline plan code, e.g. p03")
    parser.add_argument("--revised", help="Revised/proposed plan code, e.g. p05")
    parser.add_argument("--output", default="hecras_review_report.json", help="Output JSON path")
    args = parser.parse_args()

    if args.existing and args.revised:
        report = compare_plans(args.zip_model, args.existing, args.revised)
    else:
        report = inspect_zip(args.zip_model)
    save_json(report, args.output)
    print(json.dumps(report, indent=2)[:12000])
    print(f"\nSaved full report: {args.output}")


if __name__ == "__main__":
    main()

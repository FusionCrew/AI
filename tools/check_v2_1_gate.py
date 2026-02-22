from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_variant(results: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for r in results:
        v = (r.get("variant") or {}).get("name")
        if str(v) == name:
            return r
    raise ValueError(f"variant not found: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate checker for v2.1 ablation report")
    parser.add_argument("--report", required=True, type=str)
    parser.add_argument("--variant", default="parser_fsm_v2_1", type=str)
    parser.add_argument("--min-accuracy", default=0.95, type=float)
    parser.add_argument("--max-route-mismatch-rate", default=0.0, type=float)
    parser.add_argument("--max-ordering-on-info", default=0, type=int)
    args = parser.parse_args()

    report_path = Path(args.report)
    payload = _load(report_path)
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    target = _find_variant(results, args.variant)
    summary = target.get("summary") if isinstance(target.get("summary"), dict) else {}

    accuracy = float(summary.get("accuracy", 0.0))
    route_mismatch_rate = float(summary.get("routeMismatchRate", 1.0))
    ordering_on_info = int(summary.get("orderingOnInfoCount", 999999))

    failures: List[str] = []
    if accuracy < args.min_accuracy:
        failures.append(f"accuracy {accuracy:.4f} < min {args.min_accuracy:.4f}")
    if route_mismatch_rate > args.max_route_mismatch_rate:
        failures.append(
            f"routeMismatchRate {route_mismatch_rate:.4f} > max {args.max_route_mismatch_rate:.4f}"
        )
    if ordering_on_info > args.max_ordering_on_info:
        failures.append(f"orderingOnInfoCount {ordering_on_info} > max {args.max_ordering_on_info}")

    print(
        json.dumps(
            {
                "report": str(report_path),
                "variant": args.variant,
                "accuracy": accuracy,
                "routeMismatchRate": route_mismatch_rate,
                "orderingOnInfoCount": ordering_on_info,
                "thresholds": {
                    "minAccuracy": args.min_accuracy,
                    "maxRouteMismatchRate": args.max_route_mismatch_rate,
                    "maxOrderingOnInfo": args.max_ordering_on_info,
                },
                "pass": len(failures) == 0,
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()

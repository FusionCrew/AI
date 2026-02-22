from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from v2_1.eval_dataset import FIFTY_CASES


def _http_get_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str] | None = None) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=body, headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _counts(metric_data: Dict[str, Any]) -> Dict[str, int]:
    c = ((metric_data or {}).get("data") or {}).get("counts") or {}
    return {
        "total": int(c.get("total", 0)),
        "fallback": int(c.get("fallback", 0)),
        "routeMismatch": int(c.get("routeMismatch", 0)),
        "outOfDomainDrop": int(c.get("outOfDomainDrop", 0)),
    }


def _safe_div(a: int, b: int) -> float:
    return (a / b) if b > 0 else 0.0


def _eval_row(case_text: str, action: str, expected: List[str] | None, forbidden: List[str] | None) -> Dict[str, Any]:
    ok = True
    reason = "ok"
    if expected and action not in expected:
        ok = False
        reason = "unexpected_action"
    if forbidden and action in forbidden:
        ok = False
        reason = "forbidden_action"
    return {"ok": ok, "reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser(description="v2.1 online eval runner via /api/v2/llm/chat")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--target-fallback-max", type=float, default=0.05)
    args = parser.parse_args()

    chat_url = f"{args.api_base.rstrip('/')}/api/v2/llm/chat"
    metric_url = f"{args.api_base.rstrip('/')}/api/v2/metrics/quality"

    before_raw = _http_get_json(metric_url)
    before = _counts(before_raw)
    rows: List[Dict[str, Any]] = []
    passed = 0

    for i, case in enumerate(FIFTY_CASES, start=1):
        expected = sorted(list(case.expected_actions)) if case.expected_actions else None
        forbidden = sorted(list(case.forbidden_actions)) if case.forbidden_actions else None
        payload = {
            "messages": [{"role": "user", "content": case.text}],
            "sessionId": f"online-eval-{int(time.time())}-{i}",
            "context": {
                "sessionId": f"online-eval-{i}",
                "state": case.state,
            },
        }
        action = "ERROR"
        intent = ""
        orchestrator = ""
        err = ""
        try:
            res = _http_post_json(chat_url, payload, headers={"X-AI-Client-Source": "v2_1_online_eval"})
            data = res.get("data", {}) if isinstance(res, dict) else {}
            action = str(data.get("action") or "")
            intent = str(data.get("intent") or "")
            orchestrator = str(data.get("orchestrator") or "")
            row_eval = _eval_row(case.text, action, expected, forbidden)
            ok = bool(row_eval["ok"])
            reason = str(row_eval["reason"])
        except urllib.error.HTTPError as e:
            ok = False
            reason = "http_error"
            err = f"HTTP {e.code}"
        except Exception as e:
            ok = False
            reason = "exception"
            err = str(e)

        if ok:
            passed += 1
        rows.append(
            {
                "caseId": case.case_id,
                "text": case.text,
                "expectedActions": expected,
                "forbiddenActions": forbidden,
                "actualAction": action,
                "actualIntent": intent,
                "orchestrator": orchestrator,
                "ok": ok,
                "reason": reason if not ok else "",
                "error": err,
            }
        )

    after_raw = _http_get_json(metric_url)
    after = _counts(after_raw)
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in before.keys()}
    fallback_rate = _safe_div(delta["fallback"], delta["total"])
    route_mismatch_rate = _safe_div(delta["routeMismatch"], delta["total"])

    report = {
        "generatedAt": datetime.now().isoformat(),
        "apiBase": args.api_base,
        "cases": len(FIFTY_CASES),
        "pass": passed,
        "fail": len(FIFTY_CASES) - passed,
        "accuracy": round((passed / len(FIFTY_CASES)) if FIFTY_CASES else 0.0, 4),
        "metrics": {
            "before": before,
            "after": after,
            "delta": delta,
            "deltaRates": {
                "fallbackRate": round(fallback_rate, 6),
                "routeMismatchRate": round(route_mismatch_rate, 6),
            },
            "target": {
                "fallbackRateMax": args.target_fallback_max,
                "fallbackTargetMet": fallback_rate <= args.target_fallback_max,
            },
        },
        "failedCases": [r for r in rows if not r["ok"]],
        "rows": rows,
    }

    out_path = Path(args.output.strip()) if args.output.strip() else None
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("_runlogs") / f"online_eval_v2_1_{ts}.json"
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out_path),
                "pass": report["pass"],
                "fail": report["fail"],
                "accuracy": report["accuracy"],
                "deltaRates": report["metrics"]["deltaRates"],
                "target": report["metrics"]["target"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

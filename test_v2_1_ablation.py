from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from v2.orchestrator import V2LangChainOrchestrator
from v2_1.eval_dataset import FIFTY_CASES, EvalCase


MENU_ITEMS: List[Dict[str, Any]] = [
    {"menuItemId": "set_1", "name": "징거버거 세트", "categoryId": "cat_set", "price": 8900},
    {"menuItemId": "set_2", "name": "타워버거 세트", "categoryId": "cat_set", "price": 9200},
    {"menuItemId": "burger_1", "name": "징거버거", "categoryId": "cat_burger", "price": 6200},
    {"menuItemId": "burger_2", "name": "불고기버거", "categoryId": "cat_burger", "price": 5800},
    {"menuItemId": "burger_3", "name": "치즈버거", "categoryId": "cat_burger", "price": 5200},
    {"menuItemId": "side_1", "name": "코울슬로", "categoryId": "cat_side", "price": 2200},
    {"menuItemId": "side_2", "name": "감자튀김", "categoryId": "cat_side", "price": 2600},
    {"menuItemId": "drink_1", "name": "콜라", "categoryId": "cat_drink", "price": 1900},
    {"menuItemId": "drink_2", "name": "제로콜라", "categoryId": "cat_drink", "price": 2100},
    {"menuItemId": "drink_3", "name": "사이다", "categoryId": "cat_drink", "price": 1900},
]


async def _menu_list_provider() -> List[Dict[str, Any]]:
    return MENU_ITEMS


async def _menu_detail_provider(menu_item_id: str) -> Dict[str, Any]:
    details = {
        "set_1": {"menuItemId": "set_1", "allergies": ["난류", "우유", "밀"], "calories": 920},
        "set_2": {"menuItemId": "set_2", "allergies": ["난류", "우유", "밀"], "calories": 980},
        "burger_1": {"menuItemId": "burger_1", "allergies": ["난류", "우유", "밀"], "calories": 620},
        "burger_2": {"menuItemId": "burger_2", "allergies": ["밀", "대두"], "calories": 540},
        "burger_3": {"menuItemId": "burger_3", "allergies": ["난류", "우유", "밀"], "calories": 510},
        "side_1": {"menuItemId": "side_1", "allergies": ["난류"], "calories": 180},
        "side_2": {"menuItemId": "side_2", "allergies": ["대두"], "calories": 350},
        "drink_1": {"menuItemId": "drink_1", "allergies": [], "calories": 140},
        "drink_2": {"menuItemId": "drink_2", "allergies": [], "calories": 0},
        "drink_3": {"menuItemId": "drink_3", "allergies": [], "calories": 130},
    }
    return details.get(menu_item_id, {"menuItemId": menu_item_id, "allergies": [], "calories": 0})


@dataclass(frozen=True)
class Variant:
    name: str
    parser_enabled: bool
    fsm_enabled: bool


VARIANTS = [
    Variant(name="baseline_v2", parser_enabled=False, fsm_enabled=False),
    Variant(name="parser_only_v2_1", parser_enabled=True, fsm_enabled=False),
    Variant(name="parser_fsm_v2_1", parser_enabled=True, fsm_enabled=True),
]


def _set_flags(parser_enabled: bool, fsm_enabled: bool) -> None:
    os.environ["AI_V21_PARSER_ENABLED"] = "true" if parser_enabled else "false"
    os.environ["AI_V21_FSM_ENABLED"] = "true" if fsm_enabled else "false"


def _is_ordering_action(action: str) -> bool:
    return action in {"ADD_MENU", "REMOVE_MENU", "CHANGE_QTY", "CHECKOUT", "SELECT_PAYMENT"}


def _looks_like_info_or_recommend(text: str) -> bool:
    compact = "".join((text or "").lower().split())
    keys = [
        "추천",
        "알레르기",
        "알래르기",
        "알러지",
        "알레르겐",
        "재료",
        "원재료",
        "성분",
        "들어가",
        "포함",
        "빼고",
        "제외",
        "칼로리",
        "kcal",
    ]
    return any(k in compact for k in keys)


def _evaluate_case(case: EvalCase, action: str) -> Dict[str, Any]:
    ok = True
    reason = "ok"

    if case.expected_actions is not None and action not in case.expected_actions:
        ok = False
        reason = "unexpected_action"
    if case.forbidden_actions and action in case.forbidden_actions:
        ok = False
        reason = "forbidden_action"
    return {"ok": ok, "reason": reason}


async def _run_one(orch: V2LangChainOrchestrator, case: EvalCase) -> Dict[str, Any]:
    result = await orch._run_stage_policy(
        user_text=case.text,
        state=case.state if isinstance(case.state, dict) else {},
        menu_items=MENU_ITEMS,
    )
    if not isinstance(result, dict):
        result = {"action": "NONE", "intent": "UNKNOWN"}
    action = str(result.get("action") or "NONE")
    intent = str(result.get("intent") or "UNKNOWN")
    eval_row = _evaluate_case(case, action)
    mismatch = _looks_like_info_or_recommend(case.text) and _is_ordering_action(action)
    return {
        "caseId": case.case_id,
        "text": case.text,
        "state": case.state,
        "expectedActions": sorted(list(case.expected_actions)) if case.expected_actions else None,
        "forbiddenActions": sorted(list(case.forbidden_actions)) if case.forbidden_actions else None,
        "actualAction": action,
        "actualIntent": intent,
        "ok": bool(eval_row["ok"]),
        "failReason": eval_row["reason"] if not eval_row["ok"] else "",
        "routeMismatch": mismatch,
    }


async def _run_variant(variant: Variant, selected_cases: List[EvalCase]) -> Dict[str, Any]:
    _set_flags(variant.parser_enabled, variant.fsm_enabled)
    orch = V2LangChainOrchestrator(
        menu_list_provider=_menu_list_provider,
        menu_detail_provider=_menu_detail_provider,
    )
    rows: List[Dict[str, Any]] = []
    for case in selected_cases:
        rows.append(await _run_one(orch, case))

    total = len(rows)
    passed = len([r for r in rows if r["ok"]])
    failed_rows = [r for r in rows if not r["ok"]]
    route_mismatch = len([r for r in rows if r["routeMismatch"]])
    ordering_on_info = len(
        [r for r in rows if _looks_like_info_or_recommend(str(r.get("text") or "")) and _is_ordering_action(str(r.get("actualAction") or ""))]
    )
    return {
        "variant": asdict(variant),
        "summary": {
            "total": total,
            "pass": passed,
            "fail": total - passed,
            "accuracy": round((passed / total) if total else 0.0, 4),
            "routeMismatchCount": route_mismatch,
            "routeMismatchRate": round((route_mismatch / total) if total else 0.0, 4),
            "orderingOnInfoCount": ordering_on_info,
        },
        "failedCases": failed_rows,
        "rows": rows,
    }


def _build_diff_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_variant_fail_set: Dict[str, Set[str]] = {}
    by_variant_row: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in results:
        variant_name = str(((r.get("variant") or {}).get("name")) or "unknown")
        rows = r.get("rows") if isinstance(r.get("rows"), list) else []
        row_map: Dict[str, Dict[str, Any]] = {}
        failed: Set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("caseId") or "")
            if not cid:
                continue
            row_map[cid] = row
            if not bool(row.get("ok")):
                failed.add(cid)
        by_variant_row[variant_name] = row_map
        by_variant_fail_set[variant_name] = failed

    variant_names = list(by_variant_fail_set.keys())
    all_failed = sorted(set().union(*[by_variant_fail_set[v] for v in variant_names])) if variant_names else []
    common_failed = sorted(set.intersection(*[by_variant_fail_set[v] for v in variant_names])) if variant_names else []
    unique_failed: Dict[str, List[str]] = {}
    for v in variant_names:
        other = set().union(*[by_variant_fail_set[x] for x in variant_names if x != v]) if len(variant_names) > 1 else set()
        unique_failed[v] = sorted(by_variant_fail_set[v] - other)

    per_case_compare: List[Dict[str, Any]] = []
    for cid in all_failed:
        actions: Dict[str, str] = {}
        ok_by_variant: Dict[str, bool] = {}
        reasons: Dict[str, str] = {}
        for v in variant_names:
            row = by_variant_row.get(v, {}).get(cid, {})
            actions[v] = str(row.get("actualAction") or "")
            ok_by_variant[v] = bool(row.get("ok"))
            reasons[v] = str(row.get("failReason") or "")
        per_case_compare.append(
            {
                "caseId": cid,
                "actions": actions,
                "okByVariant": ok_by_variant,
                "failReasonByVariant": reasons,
            }
        )

    return {
        "commonFailedCaseIds": common_failed,
        "uniqueFailedCaseIds": unique_failed,
        "allFailedCaseIds": all_failed,
        "perCaseCompare": per_case_compare,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2.1 offline ablation runner")
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated variant names. Use 'all' for baseline_v2,parser_only_v2_1,parser_fsm_v2_1",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output json path. Default: _runlogs/ablation_v2_1_<timestamp>.json",
    )
    parser.add_argument(
        "--no-diff",
        action="store_true",
        help="Disable variant failure diff report generation.",
    )
    return parser.parse_args()


def _select_variants(raw: str) -> List[Variant]:
    if (raw or "").strip().lower() == "all":
        return list(VARIANTS)
    wanted = {x.strip() for x in (raw or "").split(",") if x.strip()}
    picked = [v for v in VARIANTS if v.name in wanted]
    if not picked:
        raise ValueError(f"Unknown variants: {raw}")
    return picked


async def main() -> None:
    args = _parse_args()
    selected_variants = _select_variants(args.variants)
    selected_cases = list(FIFTY_CASES)
    results = []
    for variant in selected_variants:
        results.append(await _run_variant(variant, selected_cases))

    output = {
        "generatedAt": datetime.now().isoformat(),
        "cases": len(selected_cases),
        "variants": [v.name for v in selected_variants],
        "results": results,
    }
    if not args.no_diff:
        output["diffReport"] = _build_diff_report(results)

    out_path = Path(args.output.strip()) if args.output.strip() else None
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("_runlogs") / f"ablation_v2_1_{ts}.json"
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "summary": [r["summary"] for r in results]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

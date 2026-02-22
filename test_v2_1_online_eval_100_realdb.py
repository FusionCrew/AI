from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    text: str
    state: Dict[str, Any]
    expected_actions: Optional[Set[str]]
    forbidden_actions: Optional[Set[str]]
    case_type: str
    expected_stage: Optional[Set[str]] = None
    required_action_data_keys: Optional[Set[str]] = None
    required_action_data_values: Optional[Dict[str, Set[str]]] = None


def _http_get_json(url: str, timeout: int = 15) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 20,
) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _safe_div(a: int, b: int) -> float:
    return (a / b) if b > 0 else 0.0


def _counts(metric_payload: Dict[str, Any]) -> Dict[str, int]:
    c = (((metric_payload or {}).get("data") or {}).get("counts") or {})
    return {
        "total": int(c.get("total", 0)),
        "fallback": int(c.get("fallback", 0)),
        "routeMismatch": int(c.get("routeMismatch", 0)),
        "outOfDomainDrop": int(c.get("outOfDomainDrop", 0)),
    }


def _fetch_menu_list(backend_base: str, size: int = 500) -> List[Dict[str, Any]]:
    qs = urllib.parse.urlencode({"size": size})
    url = f"{backend_base.rstrip('/')}/api/v1/kiosk/menu-items?{qs}"
    payload = _http_get_json(url)
    items = (((payload or {}).get("data") or {}).get("items") or [])
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict) and str(x.get("name") or "").strip()]


def _fetch_menu_detail(backend_base: str, menu_item_id: str) -> Dict[str, Any]:
    url = f"{backend_base.rstrip('/')}/api/v1/kiosk/menu-items/{urllib.parse.quote(menu_item_id)}"
    payload = _http_get_json(url)
    data = (payload or {}).get("data")
    return data if isinstance(data, dict) else {}


def _normalize_allergies(raw: Any) -> List[str]:
    if isinstance(raw, list):
        vals = [str(x).strip() for x in raw if str(x).strip()]
    elif isinstance(raw, str):
        vals = [x.strip() for x in raw.replace("|", ",").replace("/", ",").split(",") if x.strip()]
    else:
        vals = []
    out: List[str] = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def _pick_menu_by_category(items: List[Dict[str, Any]], category_ids: Set[str]) -> List[Dict[str, Any]]:
    return [x for x in items if str(x.get("categoryId") or "") in category_ids]


def _gen_cases(menu_items: List[Dict[str, Any]], detail_map: Dict[str, Dict[str, Any]], n_cases: int, seed: int) -> List[EvalCase]:
    rng = random.Random(seed)
    cases: List[EvalCase] = []
    cid = 1

    def add_case(
        text: str,
        state: Dict[str, Any],
        expected: Optional[Set[str]],
        forbidden: Optional[Set[str]],
        case_type: str,
        expected_stage: Optional[Set[str]] = None,
        required_keys: Optional[Set[str]] = None,
        required_values: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        nonlocal cid
        cases.append(
            EvalCase(
                case_id=f"R{cid:03d}",
                text=text,
                state=state,
                expected_actions=expected,
                forbidden_actions=forbidden,
                case_type=case_type,
                expected_stage=expected_stage,
                required_action_data_keys=required_keys,
                required_action_data_values=required_values,
            )
        )
        cid += 1

    names = [str(x.get("name") or "").strip() for x in menu_items]
    set_items = _pick_menu_by_category(menu_items, {"cat_set"}) or menu_items
    side_items = _pick_menu_by_category(menu_items, {"cat_side"}) or menu_items
    drink_items = _pick_menu_by_category(menu_items, {"cat_drink"}) or menu_items

    allergen_pool: List[str] = []
    for it in menu_items:
        mid = str(it.get("menuItemId") or "")
        det = detail_map.get(mid, {})
        alls = _normalize_allergies(det.get("allergies") or det.get("allergens") or [])
        for a in alls[:2]:
            if a not in allergen_pool:
                allergen_pool.append(a)
    if not allergen_pool:
        allergen_pool = ["우유", "밀", "난류"]

    # 1) 주문 추가 30
    add_templates = [
        "{name} 하나 담아줘",
        "{name} 2개 추가해줘",
        "{name} 주문할게",
        "{name} 하나 넣어줘",
        "{name} 세 개 담아",
    ]
    for _ in range(30):
        it = rng.choice(menu_items)
        name = str(it.get("name") or "").strip()
        t = rng.choice(add_templates).format(name=name)
        add_case(
            text=t,
            state={"stage": "MAIN_MENU", "diningType": "DINE_IN", "cartItems": []},
            expected={"ADD_MENU", "NONE"},
            forbidden=None,
            case_type="add_menu",
            expected_stage={"CART", "MAIN_MENU"},
            required_keys={"menuItemId"},
        )

    # 2) 정보 질의 20
    info_templates = [
        "{name} 알레르기 정보 알려줘",
        "{name} 알래르기 알려줘",
        "{name} 재료 알려줘",
        "{name} 성분 뭐야",
        "{name} 칼로리 알려줘",
        "{name} kcal 얼마야",
    ]
    for _ in range(20):
        name = rng.choice(names)
        t = rng.choice(info_templates).format(name=name)
        add_case(
            text=t,
            state={"stage": "MAIN_MENU", "diningType": "DINE_IN", "cartItems": []},
            expected={"NONE"},
            forbidden={"ADD_MENU", "REMOVE_MENU", "CHECKOUT", "SELECT_PAYMENT"},
            case_type="menu_info",
            expected_stage={"MAIN_MENU", "ASK_DINING_TYPE"},
        )

    # 3) 조건 추천 15
    rec_templates = [
        "{allergen} 안 들어간 메뉴 추천해줘",
        "{allergen} 제외 메뉴 추천",
        "{allergen} 없는 메뉴 뭐 있어",
    ]
    for _ in range(15):
        allergen = rng.choice(allergen_pool)
        t = rng.choice(rec_templates).format(allergen=allergen)
        add_case(
            text=t,
            state={"stage": "MAIN_MENU", "diningType": "DINE_IN", "cartItems": []},
            expected={"NONE"},
            forbidden={"ADD_MENU", "CHECKOUT"},
            case_type="recommend_constraint",
            expected_stage={"MAIN_MENU"},
        )

    # 4) 장바구니 확인 10
    cart_templates = [
        "장바구니 보여줘",
        "장바구니 뭐 담겼어",
        "지금 고른 메뉴 뭐야",
        "주문내역 확인해줘",
    ]
    for _ in range(10):
        it = rng.choice(menu_items)
        add_case(
            text=rng.choice(cart_templates),
            state={"stage": "MAIN_MENU", "diningType": "DINE_IN", "cartItems": [{"menuItemId": str(it.get("menuItemId")), "quantity": 1}]},
            expected={"CHECK_CART", "NONE"},
            forbidden=None,
            case_type="check_cart",
            expected_stage={"CART", "MAIN_MENU"},
        )

    # 5) 결제/결제수단 10
    pay_templates = [
        ("결제할게", {"CHECKOUT", "NONE"}, None),
        ("카드 결제할게", {"SELECT_PAYMENT", "CHECKOUT", "NONE"}, {"method": {"CARD"}}),
        ("포인트로 결제할게", {"SELECT_PAYMENT", "CHECKOUT", "NONE"}, {"method": {"POINT"}}),
        ("주문 완료할게", {"CHECKOUT", "NONE"}, None),
    ]
    for i in range(10):
        text, expected, required_values = rng.choice(pay_templates)
        state: Dict[str, Any] = {
            "stage": "MAIN_MENU",
            "diningType": "DINE_IN",
            "cartItems": [{"menuItemId": str(rng.choice(menu_items).get("menuItemId")), "quantity": 1}],
        }
        if i % 3 == 0:
            state["pageHint"] = {"paymentStep": "select"}
        add_case(
            text=text,
            state=state,
            expected=expected,
            forbidden=None,
            case_type="payment",
            expected_stage={"ORDER_REVIEW", "PAYMENT", "MAIN_MENU"},
            required_keys={"method"} if required_values else None,
            required_values=required_values,
        )

    # 6) 카테고리 이동 10
    nav_inputs = [
        ("세트 메뉴 보여줘", "set"),
        ("단품 메뉴 보여줘", "single"),
        ("사이드 메뉴 보여줘", "side"),
        ("음료 카테고리 보여줘", "drink"),
        ("치킨 카테고리로 가줘", "chicken"),
    ]
    for _ in range(10):
        text, _ = rng.choice(nav_inputs)
        add_case(
            text=text,
            state={"stage": "MAIN_MENU", "diningType": "DINE_IN", "cartItems": []},
            expected={"NAVIGATE_CATEGORY", "CONTINUE_ORDER", "NONE"},
            forbidden=None,
            case_type="navigate_category",
            expected_stage={"MAIN_MENU"},
            required_keys={"categoryKey"},
        )

    # 7) 세트 옵션 단계 15
    opt_templates = [
        "음료는 {drink}로 할게",
        "사이드는 {side}로 해줘",
        "미디엄 사이즈로 해줘",
        "라지 사이즈로 바꿔줘",
    ]
    for _ in range(15):
        t = rng.choice(opt_templates).format(
            drink=str(rng.choice(drink_items).get("name") or ""),
            side=str(rng.choice(side_items).get("name") or ""),
        )
        stage = rng.choice(["SIDE_SELECTION", "DRINK_SELECTION"])
        add_case(
            text=t,
            state={
                "stage": stage,
                "diningType": "DINE_IN",
                "pageHint": {"selectedCategory": "cat_side" if stage == "SIDE_SELECTION" else "cat_drink"},
                "cartItems": [{"menuItemId": str(rng.choice(set_items).get("menuItemId")), "quantity": 1}],
            },
            expected={"ADD_MENU", "NONE", "CHANGE_QTY", "NAVIGATE_CATEGORY"},
            forbidden=None,
            case_type="set_option",
            expected_stage={"SIDE_SELECTION", "DRINK_SELECTION", "CART"},
        )

    rng.shuffle(cases)
    return cases[:n_cases]


def _extract_stage(data: Dict[str, Any]) -> str:
    stage = str(data.get("stage") or "").strip()
    if stage:
        return stage
    action_data = data.get("actionData") if isinstance(data.get("actionData"), dict) else {}
    return str(action_data.get("stage") or "").strip()


def _evaluate_loose(action: str, expected: Optional[Set[str]], forbidden: Optional[Set[str]]) -> Tuple[bool, str]:
    if expected is not None and action not in expected:
        return False, "unexpected_action"
    if forbidden and action in forbidden:
        return False, "forbidden_action"
    return True, "ok"


def _evaluate_strict(case: EvalCase, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    action = str(data.get("action") or "")
    action_data = data.get("actionData") if isinstance(data.get("actionData"), dict) else {}
    stage = _extract_stage(data)

    ok_loose, loose_reason = _evaluate_loose(action, case.expected_actions, case.forbidden_actions)
    if not ok_loose:
        reasons.append(loose_reason)

    if case.expected_stage:
        if stage not in case.expected_stage:
            reasons.append(f"stage_mismatch:{stage}")

    if case.required_action_data_keys and action != "NONE":
        for k in case.required_action_data_keys:
            if k not in action_data or action_data.get(k) in (None, ""):
                reasons.append(f"missing_actionData:{k}")

    if case.required_action_data_values and action != "NONE":
        for k, allowed in case.required_action_data_values.items():
            v = str(action_data.get(k) or "")
            if v and allowed and v not in allowed:
                reasons.append(f"actionData_value_mismatch:{k}={v}")
            if not v and k in (case.required_action_data_keys or set()):
                reasons.append(f"missing_actionData:{k}")

    # Action-specific hard checks
    if action in {"ADD_MENU", "ADD_TO_CART"}:
        if not str(action_data.get("menuItemId") or "").strip():
            reasons.append("add_menu_without_menuItemId")
    if action == "SELECT_PAYMENT":
        method = str(action_data.get("method") or "").upper()
        if method not in {"CARD", "POINT", "SIMPLE"}:
            reasons.append("select_payment_without_valid_method")
    if action == "NAVIGATE_CATEGORY":
        if not str(action_data.get("categoryKey") or "").strip():
            reasons.append("navigate_category_without_categoryKey")

    return (len(reasons) == 0), reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict online eval with 100 generated cases using real menu DB")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000")
    parser.add_argument("--backend-base", type=str, default=os.getenv("BACKEND_BASE_URL", "http://localhost:8080"))
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--target-fallback-max", type=float, default=0.05)
    args = parser.parse_args()

    chat_url = f"{args.api_base.rstrip('/')}/api/v2/llm/chat"
    metric_url = f"{args.api_base.rstrip('/')}/api/v2/metrics/quality"

    menu_items = _fetch_menu_list(args.backend_base, size=500)
    if not menu_items:
        raise RuntimeError(f"Failed to fetch menu items from backend: {args.backend_base}")

    detail_map: Dict[str, Dict[str, Any]] = {}
    for it in menu_items[:200]:
        mid = str(it.get("menuItemId") or "")
        if not mid:
            continue
        try:
            detail_map[mid] = _fetch_menu_detail(args.backend_base, mid)
        except Exception:
            detail_map[mid] = {}

    cases = _gen_cases(
        menu_items=menu_items,
        detail_map=detail_map,
        n_cases=max(1, int(args.size)),
        seed=int(args.seed),
    )

    before_raw = _http_get_json(metric_url)
    before = _counts(before_raw)

    rows: List[Dict[str, Any]] = []
    loose_passed = 0
    strict_passed = 0
    for i, case in enumerate(cases, start=1):
        payload = {
            "messages": [{"role": "user", "content": case.text}],
            "sessionId": f"realdb-strict-{int(time.time())}-{i}",
            "context": {"sessionId": f"realdb-strict-{i}", "state": case.state},
        }
        err = ""
        data: Dict[str, Any] = {}
        try:
            res = _http_post_json(chat_url, payload, headers={"X-AI-Client-Source": "realdb-100-strict-eval"})
            data = (res or {}).get("data") or {}
            if not isinstance(data, dict):
                data = {}
        except Exception as e:
            err = str(e)

        action = str(data.get("action") or "ERROR")
        intent = str(data.get("intent") or "")
        orchestrator = str(data.get("orchestrator") or "")
        stage = _extract_stage(data)
        action_data = data.get("actionData") if isinstance(data.get("actionData"), dict) else {}

        if err:
            loose_ok, loose_reason = False, "request_error"
            strict_ok, strict_reasons = False, ["request_error"]
        else:
            loose_ok, loose_reason = _evaluate_loose(action, case.expected_actions, case.forbidden_actions)
            strict_ok, strict_reasons = _evaluate_strict(case, data)

        if loose_ok:
            loose_passed += 1
        if strict_ok:
            strict_passed += 1

        rows.append(
            {
                "caseId": case.case_id,
                "caseType": case.case_type,
                "text": case.text,
                "state": case.state,
                "expectedActions": sorted(list(case.expected_actions)) if case.expected_actions else None,
                "forbiddenActions": sorted(list(case.forbidden_actions)) if case.forbidden_actions else None,
                "expectedStage": sorted(list(case.expected_stage)) if case.expected_stage else None,
                "actualAction": action,
                "actualIntent": intent,
                "actualStage": stage,
                "actionData": action_data,
                "orchestrator": orchestrator,
                "looseOk": loose_ok,
                "looseReason": "" if loose_ok else loose_reason,
                "strictOk": strict_ok,
                "strictReasons": [] if strict_ok else strict_reasons,
                "error": err,
            }
        )

    after_raw = _http_get_json(metric_url)
    after = _counts(after_raw)
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in before.keys()}
    fallback_rate = _safe_div(delta["fallback"], delta["total"])
    route_mismatch_rate = _safe_div(delta["routeMismatch"], delta["total"])

    case_type_stats: Dict[str, Dict[str, int]] = {}
    for r in rows:
        ctype = str(r.get("caseType") or "unknown")
        slot = case_type_stats.setdefault(ctype, {"count": 0, "loosePass": 0, "strictPass": 0})
        slot["count"] += 1
        slot["loosePass"] += 1 if bool(r.get("looseOk")) else 0
        slot["strictPass"] += 1 if bool(r.get("strictOk")) else 0

    report = {
        "generatedAt": datetime.now().isoformat(),
        "apiBase": args.api_base,
        "backendBase": args.backend_base,
        "menuCount": len(menu_items),
        "cases": len(cases),
        "scores": {
            "loose": {
                "pass": loose_passed,
                "fail": len(cases) - loose_passed,
                "accuracy": round(_safe_div(loose_passed, len(cases)), 4),
            },
            "strict": {
                "pass": strict_passed,
                "fail": len(cases) - strict_passed,
                "accuracy": round(_safe_div(strict_passed, len(cases)), 4),
            },
        },
        "caseTypeStats": case_type_stats,
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
        "failedLooseCases": [x for x in rows if not x.get("looseOk")],
        "failedStrictCases": [x for x in rows if not x.get("strictOk")],
        "rows": rows,
    }

    out_path = Path(args.output.strip()) if args.output.strip() else None
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("_runlogs") / f"online_eval_v2_1_realdb_strict_{len(cases)}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(out_path),
                "menuCount": len(menu_items),
                "cases": len(cases),
                "scores": report["scores"],
                "deltaRates": report["metrics"]["deltaRates"],
                "target": report["metrics"]["target"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

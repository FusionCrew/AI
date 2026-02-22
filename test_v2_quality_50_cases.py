import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


API_BASE = "http://localhost:8000"
CHAT_URL = f"{API_BASE}/api/v2/llm/chat"
METRIC_URL = f"{API_BASE}/api/v2/metrics/quality"


@dataclass
class Case:
    id: str
    text: str
    expected_actions: Optional[Set[str]] = None
    forbidden_actions: Optional[Set[str]] = None


CASES: List[Case] = [
    Case("C01", "징거버거 세트 하나 담아줘", {"ADD_MENU"}),
    Case("C02", "징거버거 단품 2개 추가해줘", {"ADD_MENU"}),
    Case("C03", "불고기버거 하나 주문할게", {"ADD_MENU"}),
    Case("C04", "치즈버거 1개 담아줘", {"ADD_MENU"}),
    Case("C05", "콜라 하나 추가", {"ADD_MENU"}),
    Case("C06", "감자튀김 빼줘", {"REMOVE_MENU", "NONE"}),
    Case("C07", "방금 담은 메뉴 취소", {"REMOVE_MENU", "NONE"}),
    Case("C08", "장바구니 보여줘", {"CHECK_CART"}),
    Case("C09", "결제할게", {"CHECKOUT", "NONE"}),
    Case("C10", "카드로 결제할게", {"CHECKOUT", "SELECT_PAYMENT", "NONE"}),
    Case("C11", "포장으로 해줘", {"SET_DINING"}),
    Case("C12", "매장에서 먹을게", {"SET_DINING"}),
    Case("C13", "추천 메뉴 보여줘", {"NONE"}),
    Case("C14", "알레르기 없는 메뉴 추천해줘", {"NONE"}, {"ADD_MENU", "CONTINUE_ORDER"}),
    Case("C15", "우유 안 들어간 메뉴 추천", {"NONE"}, {"ADD_MENU", "CONTINUE_ORDER"}),
    Case("C16", "밀 제외하고 추천해줘", {"NONE"}, {"ADD_MENU", "CONTINUE_ORDER"}),
    Case("C17", "징거버거 알레르기 정보 알려줘", {"NONE"}),
    Case("C18", "징거버거 알래르기 정보 알려줘", {"NONE"}),
    Case("C19", "징거버거 알러지 알려줘", {"NONE"}),
    Case("C20", "징거버거 재료 뭐 들어가", {"NONE"}),
    Case("C21", "징거버거 원재료 알려줘", {"NONE"}),
    Case("C22", "징거버거 성분 알려줘", {"NONE"}),
    Case("C23", "징거버거 칼로리 알려줘", {"NONE"}),
    Case("C24", "징거버거 kcal 알려줘", {"NONE"}),
    Case("C25", "우유 들어가?", {"NONE"}),
    Case("C26", "재료에 계란 있어?", {"NONE"}),
    Case("C27", "장바구니 뭐 들어있어?", {"CHECK_CART"}),
    Case("C28", "더 주문할게", {"CONTINUE_ORDER", "NONE"}),
    Case("C29", "계속 주문", {"CONTINUE_ORDER", "NONE"}),
    Case("C30", "직원 불러줘", {"CALL_STAFF", "NONE"}),
    Case("C31", "뭐 먹을지 모르겠어", {"NONE"}),
    Case("C32", "고민되네 추천해줘", {"NONE"}),
    Case("C33", "사이드는 켄터키 통다리구이로", {"ADD_MENU", "NONE"}),
    Case("C34", "음료는 제로콜라로 할게", {"ADD_MENU", "NONE"}),
    Case("C35", "미디엄 사이즈로", {"NONE", "CHANGE_QTY"}),
    Case("C36", "라지로 바꿔줘", {"NONE", "CHANGE_QTY"}),
    Case("C37", "장바구니 비었는데 결제할래", {"NONE", "CHECKOUT"}),
    Case("C38", "주문 완료할게", {"CHECKOUT", "NONE"}),
    Case("C39", "포인트로 결제할게", {"SELECT_PAYMENT", "CHECKOUT", "NONE"}),
    Case("C40", "간편결제로 할게", {"SELECT_PAYMENT", "CHECKOUT", "NONE"}),
    Case("C41", "징거버거 세트 사가지고", {"ADD_MENU", "NONE"}),
    Case("C42", "징거버거의 알레르기 정보 알려줘", {"NONE"}, {"ADD_MENU"}),
    Case("C43", "징거버거 우유 들어가?", {"NONE"}, {"ADD_MENU"}),
    Case("C44", "추천해주고 알레르기도 같이 봐줘", {"NONE"}, {"ADD_MENU", "CONTINUE_ORDER"}),
    Case("C45", "새우 없는 메뉴 뭐 있어", {"NONE"}, {"ADD_MENU", "CONTINUE_ORDER"}),
    Case("C46", "닭고기 제외 추천 부탁해", {"NONE"}, {"ADD_MENU", "CONTINUE_ORDER"}),
    Case("C47", "음료 메뉴 보여줘", {"NAVIGATE", "NAVIGATE_CATEGORY", "CONTINUE_ORDER", "NONE"}),
    Case("C48", "사이드 메뉴 보여줘", {"NAVIGATE", "NAVIGATE_CATEGORY", "CONTINUE_ORDER", "NONE"}),
    Case("C49", "버거 메뉴로 가자", {"NAVIGATE", "NAVIGATE_CATEGORY", "CONTINUE_ORDER", "NONE"}),
    Case("C50", "지금 내가 고른 메뉴가 뭐야", {"CHECK_CART", "NONE"}),
]


def _http_get_json(url: str) -> Dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, payload: Dict, headers: Optional[Dict[str, str]] = None) -> Dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _metric_snapshot() -> Dict:
    try:
        return _http_get_json(METRIC_URL).get("data", {})
    except Exception:
        return {}


def _to_counts(metric_data: Dict) -> Dict[str, int]:
    c = metric_data.get("counts", {}) if isinstance(metric_data, dict) else {}
    return {
        "total": int(c.get("total", 0)),
        "fallback": int(c.get("fallback", 0)),
        "routeMismatch": int(c.get("routeMismatch", 0)),
        "outOfDomainDrop": int(c.get("outOfDomainDrop", 0)),
    }


def _safe_div(a: int, b: int) -> float:
    if b <= 0:
        return 0.0
    return a / b


def run() -> None:
    before = _to_counts(_metric_snapshot())
    rows: List[Dict] = []
    passed = 0

    for i, c in enumerate(CASES, start=1):
        payload = {
            "messages": [{"role": "user", "content": c.text}],
            "sessionId": f"quality-50-{int(time.time())}-{i}",
            "context": {"sessionId": f"quality-50-{i}", "state": {}},
        }
        ok = False
        action = "ERROR"
        intent = ""
        orchestrator = ""
        reply = ""
        err = ""
        try:
            res = _http_post_json(
                CHAT_URL,
                payload,
                headers={"X-AI-Client-Source": "quality-50-runner"},
            )
            data = res.get("data", {}) if isinstance(res, dict) else {}
            action = str(data.get("action") or "")
            intent = str(data.get("intent") or "")
            orchestrator = str(data.get("orchestrator") or "")
            reply = str(data.get("reply") or data.get("text") or "")
            ok = True
            if c.expected_actions is not None and action not in c.expected_actions:
                ok = False
            if c.forbidden_actions and action in c.forbidden_actions:
                ok = False
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}"
        except Exception as e:
            err = str(e)

        if ok:
            passed += 1
        rows.append(
            {
                "id": c.id,
                "text": c.text,
                "action": action,
                "intent": intent,
                "orchestrator": orchestrator,
                "ok": ok,
                "error": err,
                "reply": reply[:120],
            }
        )

    after = _to_counts(_metric_snapshot())
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in before.keys()}
    fallback_rate = _safe_div(delta["fallback"], delta["total"])
    mismatch_rate = _safe_div(delta["routeMismatch"], delta["total"])
    ood_rate = _safe_div(delta["outOfDomainDrop"], delta["total"])

    summary = {
        "cases": len(CASES),
        "pass": passed,
        "fail": len(CASES) - passed,
        "metric_before": before,
        "metric_after": after,
        "metric_delta": delta,
        "delta_rates": {
            "fallbackRate": fallback_rate,
            "routeMismatchRate": mismatch_rate,
            "outOfDomainDropRate": ood_rate,
        },
        "target": {"fallbackRateMax": 0.05},
        "target_met": fallback_rate <= 0.05,
        "rows": rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()

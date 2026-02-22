import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from v2.orchestrator import V2LangChainOrchestrator
from v2.studio_graph import graph as studio_graph


MENU_ITEMS: List[Dict[str, Any]] = [
    {"menuItemId": "set_1", "name": "징거세트", "categoryId": "cat_set", "price": 8900},
    {"menuItemId": "burger_1", "name": "징거버거", "categoryId": "cat_burger", "price": 6200},
    {"menuItemId": "side_1", "name": "감자튀김", "categoryId": "cat_side", "price": 2500},
    {"menuItemId": "drink_1", "name": "콜라", "categoryId": "cat_drink", "price": 1900},
]


async def _menu_list_provider() -> List[Dict[str, Any]]:
    return MENU_ITEMS


async def _menu_detail_provider(menu_item_id: str) -> Dict[str, Any]:
    if menu_item_id == "set_1":
        return {"menuItemId": "set_1", "name": "징거세트", "ingredients": ["치킨", "번"], "allergies": ["밀", "대두"]}
    if menu_item_id == "burger_1":
        return {
            "menuItemId": "burger_1",
            "name": "징거버거",
            "ingredientNames": "치킨패티, 양상추, 마요소스",
            "allergens": "wheat,milk,soy",
            "calories": "620 kcal",
        }
    return {"menuItemId": menu_item_id, "ingredients": [], "allergies": []}


def _make_orchestrator() -> V2LangChainOrchestrator:
    return V2LangChainOrchestrator(
        menu_list_provider=_menu_list_provider,
        menu_detail_provider=_menu_detail_provider,
    )


@dataclass
class Scenario:
    name: str
    user_text: str
    state: Dict[str, Any]
    expected_action: str
    expected_intent: str
    forbid_actions: Optional[List[str]] = None


SCENARIOS: List[Scenario] = [
    Scenario(
        name="info_allergy",
        user_text="징거버거의 알레르기 정보 알려줘",
        state={"diningType": "DINE_IN", "stage": "MAIN_MENU"},
        expected_action="NONE",
        expected_intent="MENU_INFO",
        forbid_actions=["ADD_MENU", "REMOVE_MENU"],
    ),
    Scenario(
        name="info_typo",
        user_text="징거버거의 알래르기 정보 알려줘",
        state={"diningType": "DINE_IN", "stage": "MAIN_MENU"},
        expected_action="NONE",
        expected_intent="MENU_INFO",
        forbid_actions=["ADD_MENU", "REMOVE_MENU"],
    ),
    Scenario(
        name="order_add_menu",
        user_text="징거버거 1개 담아줘",
        state={"diningType": "DINE_IN", "stage": "MAIN_MENU"},
        expected_action="ADD_MENU",
        expected_intent="ORDER_FLOW",
    ),
    Scenario(
        name="recommend_allergy_free",
        user_text="알레르기 없는 메뉴 추천해줘",
        state={
            "diningType": "DINE_IN",
            "stage": "MAIN_MENU",
            "vectorCandidates": [
                {"menuItemId": "burger_1", "name": "징거버거", "score": 0.91, "allergies": ["밀", "우유"]},
                {"menuItemId": "set_1", "name": "징거세트", "score": 0.86, "allergies": ["밀", "대두"]},
            ],
        },
        expected_action="NONE",
        expected_intent="MENU_RECOMMEND",
        forbid_actions=["ADD_MENU", "REMOVE_MENU"],
    ),
    Scenario(
        name="checkout",
        user_text="결제할게요",
        state={"diningType": "DINE_IN", "stage": "MAIN_MENU", "cartItems": [{"menuItemId": "set_1", "quantity": 1}]},
        expected_action="CHECKOUT",
        expected_intent="ORDER_FLOW",
    ),
]


def _check(s: Scenario, result: Dict[str, Any]) -> Dict[str, Any]:
    action = str(result.get("action") or "")
    intent = str(result.get("intent") or "")
    ok = (action == s.expected_action) and (intent == s.expected_intent)
    if s.forbid_actions and action in s.forbid_actions:
        ok = False
    return {"ok": ok, "action": action, "intent": intent}


async def _run_orchestrator(orch: V2LangChainOrchestrator, s: Scenario) -> Dict[str, Any]:
    req = SimpleNamespace(
        sessionId=f"trace-{s.name}",
        orderType="DINE_IN",
        context=SimpleNamespace(sessionId=f"trace-{s.name}", state=s.state),
        messages=[SimpleNamespace(role="user", content=s.user_text)],
    )
    pack = await orch.run(request=req, openai_client=None, request_id=f"req_{s.name}")
    result = pack.get("result", {})
    trace = pack.get("trace", [])
    return {"result": result, "trace": trace}


async def _run_studio(s: Scenario) -> Dict[str, Any]:
    out = await studio_graph.ainvoke(
        {
            "user_text": s.user_text,
            "state": dict(s.state),
            "menu_items": MENU_ITEMS,
        }
    )
    return out.get("result", {})


async def main() -> None:
    orch = _make_orchestrator()
    rows: List[Dict[str, Any]] = []

    for s in SCENARIOS:
        o = await _run_orchestrator(orch, s)
        g = await _run_studio(s)
        oc = _check(s, o["result"])
        gc = _check(s, g)
        rows.append(
            {
                "scenario": s.name,
                "expected_action": s.expected_action,
                "expected_intent": s.expected_intent,
                "orchestrator_ok": oc["ok"],
                "orchestrator_action": oc["action"],
                "orchestrator_intent": oc["intent"],
                "orchestrator_steps": [t.get("step") for t in o["trace"] if isinstance(t, dict)],
                "studio_ok": gc["ok"],
                "studio_action": gc["action"],
                "studio_intent": gc["intent"],
            }
        )

    passed = len([r for r in rows if r["orchestrator_ok"] and r["studio_ok"]])
    print(json.dumps({"pass": passed, "total": len(rows), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())


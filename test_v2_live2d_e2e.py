import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List

from v2.live2d_mapping import ALL_MOTIONS
from v2.orchestrator import V2LangChainOrchestrator


MENU_ITEMS: List[Dict[str, Any]] = [
    {"menuItemId": "set_1", "name": "징거세트", "categoryId": "cat_set", "price": 8900},
    {"menuItemId": "burger_1", "name": "징거버거", "categoryId": "cat_burger", "price": 6200},
    {"menuItemId": "side_1", "name": "감자튀김", "categoryId": "cat_side", "price": 2500},
    {"menuItemId": "drink_1", "name": "콜라", "categoryId": "cat_drink", "price": 1900},
]


async def _menu_list_provider() -> List[Dict[str, Any]]:
    return MENU_ITEMS


async def _menu_detail_provider(menu_item_id: str) -> Dict[str, Any]:
    if menu_item_id == "burger_1":
        return {"menuItemId": "burger_1", "name": "징거버거", "allergies": ["밀", "우유"], "ingredients": ["치킨", "번"]}
    return {"menuItemId": menu_item_id, "name": "메뉴", "allergies": [], "ingredients": []}


def _make_orchestrator() -> V2LangChainOrchestrator:
    return V2LangChainOrchestrator(
        menu_list_provider=_menu_list_provider,
        menu_detail_provider=_menu_detail_provider,
    )


async def _run(orch: V2LangChainOrchestrator, user_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
    req = SimpleNamespace(
        sessionId="e2e-live2d",
        orderType="DINE_IN",
        context=SimpleNamespace(sessionId="e2e-live2d", state=state),
        messages=[SimpleNamespace(role="user", content=user_text)],
    )
    pack = await orch.run(request=req, openai_client=None, request_id="req_e2e_live2d")
    return pack.get("result", {})


def _assert_live2d_payload(result: Dict[str, Any]) -> None:
    live2d = result.get("live2d")
    parallel = result.get("parallel")
    assert isinstance(live2d, dict), "live2d missing"
    assert isinstance(parallel, dict), "parallel missing"
    assert parallel.get("runInParallel") is True, "parallel.runInParallel must be true"
    assert isinstance(parallel.get("tts"), dict), "parallel.tts missing"
    assert isinstance(parallel.get("emotion"), dict), "parallel.emotion missing"
    assert str(live2d.get("emotion") or ""), "live2d.emotion missing"
    assert str(live2d.get("motion") or ""), "live2d.motion missing"
    assert str(live2d.get("motion")) in ALL_MOTIONS, "live2d.motion must be known catalog motion"
    assert parallel.get("emotion", {}).get("motion") == live2d.get("motion"), "parallel emotion motion mismatch"


async def main() -> None:
    orch = _make_orchestrator()
    cases = [
        ("징거버거의 알레르기 정보 알려줘", {"diningType": "DINE_IN", "stage": "MAIN_MENU"}, {"MENU_INFO"}, {"NONE"}),
        ("알레르기 없는 메뉴 추천해줘", {"diningType": "DINE_IN", "stage": "MAIN_MENU", "vectorCandidates": []}, {"MENU_RECOMMEND"}, {"NONE"}),
        ("징거버거 1개 담아줘", {"diningType": "DINE_IN", "stage": "MAIN_MENU"}, {"ORDER_FLOW"}, {"ADD_MENU"}),
        ("결제할게요", {"diningType": "DINE_IN", "stage": "MAIN_MENU", "cartItems": [{"menuItemId": "set_1", "quantity": 1}]}, {"ORDER_FLOW"}, {"CHECKOUT"}),
        ("직원 불러줘", {"diningType": "DINE_IN", "stage": "MAIN_MENU"}, {"ORDER_FLOW", "GENERAL"}, {"CALL_STAFF", "NONE"}),
    ]

    for user_text, state, expected_intents, expected_actions in cases:
        result = await _run(orch, user_text, state)
        _assert_live2d_payload(result)
        assert str(result.get("intent") or "") in expected_intents
        assert str(result.get("action") or "") in expected_actions

    print(f"PASS: {len(cases)} live2d e2e cases")


if __name__ == "__main__":
    asyncio.run(main())

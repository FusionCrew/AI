import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List

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
        return {"menuItemId": "set_1", "ingredients": ["치킨", "번"], "allergies": ["밀", "대두"]}
    if menu_item_id == "burger_1":
        # Intentionally messy real-data-like payload (field name drift + string payloads).
        return {
            "menuItemId": "burger_1",
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


async def _run_orchestrator(
    orch: V2LangChainOrchestrator,
    user_text: str,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    request = SimpleNamespace(
        sessionId="test-session",
        orderType="DINE_IN",
        context=SimpleNamespace(sessionId="test-session", state=state),
        messages=[SimpleNamespace(role="user", content=user_text)],
    )
    pack = await orch.run(request=request, openai_client=None, request_id="req_test")
    return pack.get("result", {})


async def test_hesitation_proactive() -> None:
    orch = _make_orchestrator()
    out = await orch._run_stage_policy(
        user_text="잘 모르겠어요",
        state={"diningType": "DINE_IN"},
        menu_items=MENU_ITEMS,
    )
    assert out is not None
    assert out.get("intent") == "PROACTIVE_HELP"
    assert out.get("parallel", {}).get("runInParallel") is True


async def test_set_dining_from_voice() -> None:
    orch = _make_orchestrator()
    out = await orch._run_stage_policy(
        user_text="포장으로 할게요",
        state={},
        menu_items=MENU_ITEMS,
    )
    assert out is not None
    assert out.get("action") == "SET_DINING"
    assert out.get("actionData", {}).get("diningType") == "TAKE_OUT"


async def test_add_menu_and_side_stage_hint() -> None:
    orch = _make_orchestrator()
    out = await orch._run_stage_policy(
        user_text="징거세트 2개 주세요",
        state={"diningType": "DINE_IN"},
        menu_items=MENU_ITEMS,
    )
    assert out is not None
    assert out.get("action") == "ADD_MENU"
    assert out.get("actionData", {}).get("menuItemId") == "set_1"
    assert out.get("actionData", {}).get("quantity") == 2
    assert out.get("actionData", {}).get("nextStage") == "SIDE_SELECTION"


async def test_menu_info_allergy_should_not_add_menu() -> None:
    orch = _make_orchestrator()
    result = await _run_orchestrator(
        orch,
        user_text="징거버거의 알레르기 정보 알려줘",
        state={"diningType": "DINE_IN"},
    )
    assert result.get("action") == "NONE"
    assert result.get("intent") == "MENU_INFO"


async def test_menu_info_allergy_typo_should_not_add_menu() -> None:
    orch = _make_orchestrator()
    result = await _run_orchestrator(
        orch,
        user_text="징거버거의 알래르기 정보 알려줘",
        state={"diningType": "DINE_IN"},
    )
    assert result.get("action") == "NONE"
    assert result.get("intent") == "MENU_INFO"


async def test_menu_info_ingredient_should_not_add_menu() -> None:
    orch = _make_orchestrator()
    result = await _run_orchestrator(
        orch,
        user_text="징거버거 우유 들어가?",
        state={"diningType": "DINE_IN"},
    )
    assert result.get("action") == "NONE"
    assert result.get("intent") == "MENU_INFO"


async def test_menu_info_typo_variant_should_not_add_menu() -> None:
    orch = _make_orchestrator()
    result = await _run_orchestrator(
        orch,
        user_text="징거버거 알러지 잇어?",
        state={"diningType": "DINE_IN"},
    )
    assert result.get("action") == "NONE"
    assert result.get("intent") == "MENU_INFO"


async def test_menu_info_component_variant_should_not_add_menu() -> None:
    orch = _make_orchestrator()
    result = await _run_orchestrator(
        orch,
        user_text="징거버거 성분 뭐 들었어?",
        state={"diningType": "DINE_IN"},
    )
    assert result.get("action") == "NONE"
    assert result.get("intent") == "MENU_INFO"


async def test_menu_info_calorie_from_messy_detail_should_work() -> None:
    orch = _make_orchestrator()
    result = await _run_orchestrator(
        orch,
        user_text="징거버거 칼로리 알려줘",
        state={"diningType": "DINE_IN"},
    )
    assert result.get("action") == "NONE"
    assert result.get("intent") == "MENU_INFO"
    assert "620" in str(result.get("reply") or "")


async def test_ordering_add_menu_should_work() -> None:
    orch = _make_orchestrator()
    result = await _run_orchestrator(
        orch,
        user_text="징거버거 1개 담아줘",
        state={"diningType": "DINE_IN"},
    )
    assert result.get("action") == "ADD_MENU"


async def test_recommendation_from_vector_candidates() -> None:
    orch = _make_orchestrator()
    result = await _run_orchestrator(
        orch,
        user_text="알레르기 없는 메뉴 추천해줘",
        state={
            "diningType": "DINE_IN",
            "vectorCandidates": [
                {"menuItemId": "burger_1", "name": "징거버거", "score": 0.91},
                {"menuItemId": "set_1", "name": "징거세트", "score": 0.86},
            ],
        },
    )
    assert result.get("intent") == "MENU_RECOMMEND"
    assert result.get("action") == "NONE"
    assert isinstance(result.get("actionData", {}).get("recommendationCandidates"), list)
    assert len(result.get("actionData", {}).get("recommendationCandidates")) >= 1


async def test_mixed_recommendation_and_info_should_stay_recommendation() -> None:
    orch = _make_orchestrator()
    result = await _run_orchestrator(
        orch,
        user_text="추천해주고 알레르기도 고려해줘",
        state={
            "diningType": "DINE_IN",
            "vectorCandidates": [
                {"menuItemId": "burger_1", "name": "징거버거", "score": 0.91, "allergies": ["밀", "우유"]},
                {"menuItemId": "set_1", "name": "징거세트", "score": 0.86, "allergies": ["밀", "대두"]},
            ],
        },
    )
    assert result.get("intent") == "MENU_RECOMMEND"
    assert result.get("action") == "NONE"


async def test_studio_graph_matches_info_policy() -> None:
    result = await studio_graph.ainvoke(
        {
            "user_text": "징거버거의 알레르기 정보 알려줘",
            "state": {"diningType": "DINE_IN", "stage": "MAIN_MENU"},
            "menu_items": MENU_ITEMS,
        }
    )
    payload = result.get("result") or {}
    assert payload.get("action") == "NONE"
    assert payload.get("intent") == "MENU_INFO"


async def test_checkout_and_payment() -> None:
    orch = _make_orchestrator()
    checkout = await orch._run_stage_policy(
        user_text="결제할게요",
        state={"diningType": "DINE_IN", "cartItems": [{"menuItemId": "set_1", "quantity": 1}]},
        menu_items=MENU_ITEMS,
    )
    assert checkout is not None
    assert checkout.get("action") == "CHECKOUT"

    pay = await orch._run_stage_policy(
        user_text="카드로 결제",
        state={
            "diningType": "DINE_IN",
            "cartItems": [{"menuItemId": "set_1", "quantity": 1}],
            "pageHint": {"paymentStep": "select"},
        },
        menu_items=MENU_ITEMS,
    )
    assert pay is not None
    assert pay.get("action") == "SELECT_PAYMENT"
    assert pay.get("actionData", {}).get("method") == "CARD"


async def main() -> None:
    tests = [
        test_hesitation_proactive,
        test_set_dining_from_voice,
        test_add_menu_and_side_stage_hint,
        test_menu_info_allergy_should_not_add_menu,
        test_menu_info_allergy_typo_should_not_add_menu,
        test_menu_info_ingredient_should_not_add_menu,
        test_menu_info_typo_variant_should_not_add_menu,
        test_menu_info_component_variant_should_not_add_menu,
        test_menu_info_calorie_from_messy_detail_should_work,
        test_ordering_add_menu_should_work,
        test_recommendation_from_vector_candidates,
        test_mixed_recommendation_and_info_should_stay_recommendation,
        test_studio_graph_matches_info_policy,
        test_checkout_and_payment,
    ]
    for t in tests:
        await t()
    print(f"PASS: {len(tests)} scenario tests")


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
from typing import Any, Dict, List

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
    if menu_item_id == "set_1":
        return {"menuItemId": "set_1", "ingredients": ["치킨", "번"], "allergies": ["밀", "대두"]}
    if menu_item_id == "burger_1":
        return {"menuItemId": "burger_1", "ingredients": ["치킨", "번"], "allergies": ["밀"]}
    return {"menuItemId": menu_item_id, "ingredients": [], "allergies": []}


def _make_orchestrator() -> V2LangChainOrchestrator:
    return V2LangChainOrchestrator(
        menu_list_provider=_menu_list_provider,
        menu_detail_provider=_menu_detail_provider,
    )


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


async def test_recommendation_from_vector_candidates() -> None:
    orch = _make_orchestrator()
    out = await orch._run_stage_policy(
        user_text="알레르기 없는 메뉴 추천해줘",
        state={
            "diningType": "DINE_IN",
            "vectorCandidates": [
                {"menuItemId": "burger_1", "name": "징거버거", "score": 0.91},
                {"menuItemId": "set_1", "name": "징거세트", "score": 0.86},
            ],
        },
        menu_items=MENU_ITEMS,
    )
    assert out is not None
    assert out.get("intent") == "MENU_RECOMMEND"
    assert isinstance(out.get("actionData", {}).get("recommendationCandidates"), list)
    assert len(out.get("actionData", {}).get("recommendationCandidates")) >= 1


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
        test_recommendation_from_vector_candidates,
        test_checkout_and_payment,
    ]
    for t in tests:
        await t()
    print(f"PASS: {len(tests)} scenario tests")


if __name__ == "__main__":
    asyncio.run(main())

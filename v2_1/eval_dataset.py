from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    text: str
    state: Dict[str, object]
    expected_actions: Optional[Set[str]] = None
    forbidden_actions: Optional[Set[str]] = None


DEFAULT_STATE: Dict[str, object] = {
    "stage": "MAIN_MENU",
    "diningType": "DINE_IN",
    "cartItems": [],
}


def _s(**kwargs: object) -> Dict[str, object]:
    out = dict(DEFAULT_STATE)
    out.update(kwargs)
    return out


FIFTY_CASES = [
    EvalCase("C01", "징거버거 세트 하나 담아줘", _s(), {"ADD_MENU"}),
    EvalCase("C02", "징거버거 단품 2개 추가해줘", _s(), {"ADD_MENU"}),
    EvalCase("C03", "불고기버거 하나 주문할게", _s(), {"ADD_MENU"}),
    EvalCase("C04", "치즈버거 1개 담아줘", _s(), {"ADD_MENU"}),
    EvalCase("C05", "콜라 하나 추가", _s(), {"ADD_MENU"}),
    EvalCase("C06", "감자튀김 빼줘", _s(cartItems=[{"menuItemId": "side_1", "quantity": 1}]), {"REMOVE_MENU", "NONE"}),
    EvalCase("C07", "방금 담은 메뉴 취소", _s(cartItems=[{"menuItemId": "set_1", "quantity": 1}]), {"REMOVE_MENU", "NONE"}),
    EvalCase("C08", "장바구니 보여줘", _s(), {"CHECK_CART"}),
    EvalCase("C09", "결제할게", _s(cartItems=[{"menuItemId": "set_1", "quantity": 1}]), {"CHECKOUT", "NONE"}),
    EvalCase("C10", "카드로 결제할게", _s(cartItems=[{"menuItemId": "set_1", "quantity": 1}], pageHint={"paymentStep": "select"}), {"SELECT_PAYMENT", "CHECKOUT", "NONE"}),
    EvalCase("C11", "포장으로 해줘", _s(diningType=""), {"SET_DINING"}),
    EvalCase("C12", "매장에서 먹을게", _s(diningType=""), {"SET_DINING"}),
    EvalCase("C13", "추천 메뉴 보여줘", _s(), {"NONE"}),
    EvalCase("C14", "알레르기 없는 메뉴 추천해줘", _s(), {"NONE"}, {"ADD_MENU", "CONTINUE_ORDER"}),
    EvalCase("C15", "우유 안 들어간 메뉴 추천", _s(), {"NONE"}, {"ADD_MENU", "CONTINUE_ORDER"}),
    EvalCase("C16", "밀 제외하고 추천해줘", _s(), {"NONE"}, {"ADD_MENU", "CONTINUE_ORDER"}),
    EvalCase("C17", "징거버거 알레르기 정보 알려줘", _s(), {"NONE"}, {"ADD_MENU", "REMOVE_MENU"}),
    EvalCase("C18", "징거버거 알래르기 정보 알려줘", _s(), {"NONE"}, {"ADD_MENU", "REMOVE_MENU"}),
    EvalCase("C19", "징거버거 알러지 알려줘", _s(), {"NONE"}, {"ADD_MENU", "REMOVE_MENU"}),
    EvalCase("C20", "징거버거 우유 들어가?", _s(), {"NONE"}, {"ADD_MENU", "REMOVE_MENU"}),
    EvalCase("C21", "징거버거 재료 알려줘", _s(), {"NONE"}, {"ADD_MENU"}),
    EvalCase("C22", "징거버거 성분 알려줘", _s(), {"NONE"}, {"ADD_MENU"}),
    EvalCase("C23", "징거버거 칼로리 알려줘", _s(), {"NONE"}, {"ADD_MENU"}),
    EvalCase("C24", "징거버거 kcal 알려줘", _s(), {"NONE"}, {"ADD_MENU"}),
    EvalCase("C25", "우유 들어가?", _s(), {"NONE"}),
    EvalCase("C26", "음료에 카페인 있어?", _s(), {"NONE"}),
    EvalCase("C27", "장바구니 뭐 들어있어?", _s(cartItems=[{"menuItemId": "set_1", "quantity": 1}]), {"CHECK_CART"}),
    EvalCase("C28", "계속 주문할게", _s(), {"CONTINUE_ORDER", "NONE"}),
    EvalCase("C29", "메뉴 다시 보여줘", _s(), {"CONTINUE_ORDER", "NONE"}),
    EvalCase("C30", "직원 불러줘", _s(), {"CALL_STAFF", "NONE"}),
    EvalCase("C31", "뭐 먹을지 모르겠어", _s(), {"NONE"}),
    EvalCase("C32", "고민되네 추천해줘", _s(), {"NONE"}),
    EvalCase("C33", "사이드는 켄터키로", _s(stage="SIDE_SELECTION"), {"ADD_MENU", "NONE"}),
    EvalCase("C34", "음료는 제로콜라로 할게", _s(stage="DRINK_SELECTION"), {"ADD_MENU", "NONE"}),
    EvalCase("C35", "미디엄 사이즈로", _s(stage="DRINK_SELECTION"), {"NONE", "CHANGE_QTY"}),
    EvalCase("C36", "라지로 바꿔줘", _s(stage="DRINK_SELECTION"), {"NONE", "CHANGE_QTY"}),
    EvalCase("C37", "장바구니 비었는데 결제할래", _s(cartItems=[]), {"NONE", "CHECKOUT"}),
    EvalCase("C38", "주문 완료할게", _s(cartItems=[{"menuItemId": "set_1", "quantity": 1}]), {"CHECKOUT", "NONE"}),
    EvalCase("C39", "포인트로 결제할게", _s(cartItems=[{"menuItemId": "set_1", "quantity": 1}], pageHint={"paymentStep": "select"}), {"SELECT_PAYMENT", "CHECKOUT", "NONE"}),
    EvalCase("C40", "간편결제로 할게", _s(cartItems=[{"menuItemId": "set_1", "quantity": 1}], pageHint={"paymentStep": "select"}), {"SELECT_PAYMENT", "CHECKOUT", "NONE"}),
    EvalCase("C41", "징거버거 세트 사가지고", _s(), {"ADD_MENU", "NONE"}),
    EvalCase("C42", "징거버거의 알레르기 정보 알려줘", _s(), {"NONE"}, {"ADD_MENU"}),
    EvalCase("C43", "징거버거에 우유 들어가?", _s(), {"NONE"}, {"ADD_MENU"}),
    EvalCase("C44", "추천해주고 알레르기도 같이 봐줘", _s(), {"NONE"}, {"ADD_MENU", "CONTINUE_ORDER"}),
    EvalCase("C45", "새우 없는 메뉴 뭐 있어", _s(), {"NONE"}, {"ADD_MENU", "CONTINUE_ORDER"}),
    EvalCase("C46", "돼지고기 제외 추천 부탁해", _s(), {"NONE"}, {"ADD_MENU", "CONTINUE_ORDER"}),
    EvalCase("C47", "음료 메뉴 보여줘", _s(), {"NAVIGATE_CATEGORY", "CONTINUE_ORDER", "NONE"}),
    EvalCase("C48", "사이드 메뉴 보여줘", _s(), {"NAVIGATE_CATEGORY", "CONTINUE_ORDER", "NONE"}),
    EvalCase("C49", "버거 메뉴로 가줘", _s(), {"NAVIGATE_CATEGORY", "CONTINUE_ORDER", "NONE"}),
    EvalCase("C50", "지금 내가 고른 메뉴가 뭐야", _s(cartItems=[{"menuItemId": "set_1", "quantity": 1}]), {"CHECK_CART", "NONE"}),
]

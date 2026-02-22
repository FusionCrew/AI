from v2_1.intent_parser import parse_state_intent


MENU_ITEMS = [
    {"menuItemId": "set_1", "name": "핫크리스피버거세트", "categoryId": "cat_set"},
    {"menuItemId": "side_1", "name": "코울슬로", "categoryId": "cat_side"},
    {"menuItemId": "drink_1", "name": "콜라", "categoryId": "cat_drink"},
]


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run() -> None:
    p1 = parse_state_intent("세트 메뉴 보여줘", {"diningType": "DINE_IN"}, MENU_ITEMS)
    _assert(p1.intent == "NAVIGATE_CATEGORY", f"expected NAVIGATE_CATEGORY, got {p1.intent}")
    _assert(p1.category_key == "set", f"expected set, got {p1.category_key}")

    p2 = parse_state_intent("고울슬로 하나 담아봐", {"diningType": "DINE_IN"}, MENU_ITEMS)
    _assert(p2.intent == "ADD_MENU", f"expected ADD_MENU, got {p2.intent}")
    _assert(p2.menu_item_id == "side_1", f"expected side_1, got {p2.menu_item_id}")

    p3 = parse_state_intent("너 왜 코울슬로 또 시켰니", {"diningType": "DINE_IN"}, MENU_ITEMS)
    _assert(p3.intent == "CHECK_CART", f"expected CHECK_CART, got {p3.intent}")

    p4 = parse_state_intent("메뉴가 뭐 담겨있냐고", {"diningType": "DINE_IN"}, MENU_ITEMS)
    _assert(p4.intent == "CHECK_CART", f"expected CHECK_CART, got {p4.intent}")

    p5 = parse_state_intent("카드 결제", {"diningType": "DINE_IN", "pageHint": {"showOrderView": True}}, MENU_ITEMS)
    _assert(p5.intent == "SELECT_PAYMENT", f"expected SELECT_PAYMENT, got {p5.intent}")
    _assert(p5.payment_method == "CARD", f"expected CARD, got {p5.payment_method}")

    p6 = parse_state_intent("너 진짜 멍청하다", {"diningType": "DINE_IN", "selectedCategory": "side"}, MENU_ITEMS)
    _assert(p6.intent == "NONE", f"expected NONE, got {p6.intent}")

    print("test_v2_1_parser: PASS")


if __name__ == "__main__":
    run()


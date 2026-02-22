import asyncio
import json
from typing import Any, Dict, List, Tuple

from v2.studio_graph import graph


MENU_ITEMS: List[Dict[str, Any]] = [
    {"menuItemId": "set_1", "name": "징거세트", "categoryId": "cat_set", "price": 8900},
    {"menuItemId": "burger_1", "name": "징거버거", "categoryId": "cat_burger", "price": 6200},
    {"menuItemId": "side_1", "name": "감자튀김", "categoryId": "cat_side", "price": 2500},
    {"menuItemId": "drink_1", "name": "콜라", "categoryId": "cat_drink", "price": 1900},
]


CASES: List[Tuple[str, str]] = [
    ("Q01_info_allergy", "징거버거의 알레르기 정보 알려줘"),
    ("Q02_info_allergy_typo", "징거버거의 알래르기 정보 알려줘"),
    ("Q03_info_milk", "징거버거 우유 들어가?"),
    ("Q04_info_ingredient", "징거버거 성분 알려줘"),
    ("Q05_info_calorie", "징거버거 칼로리 알려줘"),
    ("Q06_recommend_allergy_free", "알레르기 없는 메뉴 추천해줘"),
    ("Q07_recommend_mixed", "추천해주고 알레르기도 고려해줘"),
    ("Q08_hesitation", "뭐 먹을지 모르겠어"),
    ("Q09_add_burger", "징거버거 1개 담아줘"),
    ("Q10_add_set", "징거세트 2개 주세요"),
    ("Q11_set_takeout", "포장으로 할게요"),
    ("Q12_set_dinein", "매장에서 먹고 갈게"),
    ("Q13_check_cart", "장바구니 보여줘"),
    ("Q14_checkout", "결제할게요"),
    ("Q15_pay_card", "카드로 결제"),
    ("Q16_pay_point", "포인트로 결제할래"),
    ("Q17_call_staff", "직원 불러줘"),
    ("Q18_continue", "계속 주문할게"),
    ("Q19_add_drink", "콜라 하나 추가"),
    ("Q20_remove_side", "감자튀김 빼고 싶어"),
]


async def main() -> None:
    rows = []
    for case_id, user_text in CASES:
        out = await graph.ainvoke(
            {
                "user_text": user_text,
                "state": {"diningType": "DINE_IN", "stage": "MAIN_MENU"},
                "menu_items": MENU_ITEMS,
            }
        )
        result = out.get("result") or {}
        rows.append(
            {
                "id": case_id,
                "action": result.get("action"),
                "intent": result.get("intent"),
                "stage": result.get("stage"),
                "speech": result.get("speech"),
            }
        )

    print(json.dumps({"total": len(rows), "rows": rows}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    asyncio.run(main())


import asyncio
from typing import Any, Dict, List

from v2.orchestrator import V2LangChainOrchestrator


CATALOG: List[Dict[str, Any]] = [
    {
        "menuItemId": "set_1",
        "name": "징거버거세트",
        "categoryId": "cat_set",
        "ingredients": ["참깨번", "닭가슴살패티", "토마토", "양상추", "마요네즈"],
        "allergies": ["난류", "닭고기", "대두", "밀", "쇠고기", "우유", "토마토"],
    },
    {
        "menuItemId": "set_2",
        "name": "타워버거세트",
        "categoryId": "cat_set",
        "ingredients": ["참깨번", "닭가슴살패티", "해쉬브라운", "토마토", "양상추", "슬라이스치즈"],
        "allergies": ["난류", "닭고기", "대두", "밀", "쇠고기", "우유", "토마토"],
    },
    {
        "menuItemId": "set_3",
        "name": "불고기버거세트",
        "categoryId": "cat_set",
        "ingredients": ["참깨번", "불고기패티", "양상추", "마요네즈"],
        "allergies": ["난류", "대두", "밀", "쇠고기", "우유"],
    },
    {
        "menuItemId": "set_4",
        "name": "핫치즈징거버거세트",
        "categoryId": "cat_set",
        "ingredients": ["참깨번", "닭가슴살패티", "핫치즈소스", "양상추"],
        "allergies": ["난류", "닭고기", "대두", "밀", "쇠고기", "우유"],
    },
    {
        "menuItemId": "set_5",
        "name": "오리지널치킨세트",
        "categoryId": "cat_set",
        "ingredients": ["치킨", "감자튀김", "콜라"],
        "allergies": ["닭고기", "밀"],
    },
    {
        "menuItemId": "set_6",
        "name": "치킨안심버거세트",
        "categoryId": "cat_set",
        "ingredients": ["브리오슈번", "치킨안심패티", "양상추", "토마토"],
        "allergies": ["닭고기", "밀", "대두", "토마토"],
    },
    {
        "menuItemId": "burger_1",
        "name": "징거버거",
        "categoryId": "cat_burger",
        "ingredients": ["참깨번", "닭가슴살패티", "토마토", "양상추", "마요네즈"],
        "allergies": ["난류", "닭고기", "대두", "밀", "쇠고기", "우유", "토마토"],
    },
    {
        "menuItemId": "burger_2",
        "name": "커스텀 버거",
        "categoryId": "cat_burger",
        "ingredients": ["참깨번", "식물성패티", "양상추"],
        "allergies": ["대두", "밀"],
    },
    {
        "menuItemId": "side_1",
        "name": "코울슬로",
        "categoryId": "cat_side",
        "ingredients": ["양배추", "마요네즈", "식초"],
        "allergies": ["난류", "대두"],
    },
    {
        "menuItemId": "drink_1",
        "name": "제로콜라",
        "categoryId": "cat_drink",
        "ingredients": ["탄산수", "카라멜색소"],
        "allergies": [],
    },
]


async def _menu_list_provider() -> List[Dict[str, Any]]:
    return CATALOG


async def _menu_detail_provider(menu_item_id: str) -> Dict[str, Any]:
    for item in CATALOG:
        if str(item.get("menuItemId")) == menu_item_id:
            return item
    return {"menuItemId": menu_item_id, "ingredients": [], "allergies": []}


def _make_orch() -> V2LangChainOrchestrator:
    return V2LangChainOrchestrator(
        menu_list_provider=_menu_list_provider,
        menu_detail_provider=_menu_detail_provider,
    )


def _candidate_names(result: Dict[str, Any]) -> List[str]:
    action_data = result.get("actionData") if isinstance(result, dict) else {}
    if not isinstance(action_data, dict):
        return []
    candidates = action_data.get("recommendationCandidates")
    if not isinstance(candidates, list):
        return []
    out: List[str] = []
    for c in candidates:
        if isinstance(c, dict):
            n = str(c.get("name") or "").strip()
            if n:
                out.append(n)
    return out


def _run_case(orch: V2LangChainOrchestrator, query: str, expect_empty: bool = False) -> None:
    result = orch._build_vector_recommendation_response(query, {"menuCatalog": CATALOG}) or {}
    assert result.get("intent") == "MENU_RECOMMEND"
    assert result.get("action") == "NONE"
    names = _candidate_names(result)
    speech = str(result.get("speech") or "")
    if expect_empty:
        assert "찾지 못했어요" in speech
    else:
        assert len(names) >= 1


async def main() -> None:
    orch = _make_orch()
    test_queries = [
        "닭가슴살 패티가 들어간 메뉴 추천해줘",
        "닭가슴살패티 포함 메뉴 추천해줘",
        "토마토가 들어간 메뉴 추천해줘",
        "해쉬브라운이 들어간 메뉴 추천해줘",
        "양상추 들어간 메뉴 추천",
        "마요네즈 들어간 세트 추천해줘",
        "치킨안심패티 포함 메뉴 추천",
        "불고기패티가 들어간 메뉴 추천",
        "식물성패티 들어간 메뉴 추천",
        "탄산수가 들어간 메뉴 추천",
        "닭가슴살 패티가 안 들어간 메뉴 추천해줘",
        "토마토 안들어간 메뉴 추천해줘",
        "해쉬브라운 제외 메뉴 추천",
        "양상추 빼고 추천해줘",
        "마요네즈 제외 메뉴 추천",
        "불고기패티 없는 메뉴 추천",
        "치킨안심패티 안들어간 메뉴 추천",
        "식물성패티 제외 메뉴 추천",
        "탄산수 제외 메뉴 추천",
        "토마토 없는 세트 메뉴 추천",
        "우유 안들어가는 메뉴 추천해줘",
        "우유 없는 세트 메뉴 추천",
        "돼지고기 안들어간 메뉴 추천",
        "대두 제외 메뉴 추천",
        "밀 제외 메뉴 추천",
        "쇠고기 포함 메뉴 추천",
        "닭고기 포함 메뉴 추천",
        "토마토 포함 메뉴 추천",
        "난류 제외 메뉴 추천",
        "굴 포함 메뉴 추천해줘",
    ]

    for i, q in enumerate(test_queries, start=1):
        _run_case(orch, q, expect_empty=("굴 포함" in q))
    print(f"PASS: {len(test_queries)} recommendation regression tests")


if __name__ == "__main__":
    asyncio.run(main())

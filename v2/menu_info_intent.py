from __future__ import annotations

import re
from typing import Dict, List


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    return re.sub(r"[\s\?\!\.\,\~\-\_\/'\"]+", "", t)


_ALLERGY_QUERY_KEYWORDS = [
    "알레르기",
    "알레르겐",
    "알러지",
    "민감",
    "못먹",
]

_INGREDIENT_QUERY_KEYWORDS = [
    "재료",
    "원재료",
    "성분",
    "들어가",
    "포함",
    "비고",
    "제외",
    "빼고",
]

_CALORIE_QUERY_KEYWORDS = [
    "칼로리",
    "kcal",
]


def is_allergy_query(text: str) -> bool:
    t = _normalize(text)
    return any(tok in t for tok in _ALLERGY_QUERY_KEYWORDS)


def is_ingredient_query(text: str) -> bool:
    t = _normalize(text)
    return any(tok in t for tok in _INGREDIENT_QUERY_KEYWORDS)


def is_calorie_query(text: str) -> bool:
    t = _normalize(text)
    return any(tok in t for tok in _CALORIE_QUERY_KEYWORDS)


def is_menu_info_query(text: str) -> bool:
    return is_allergy_query(text) or is_ingredient_query(text) or is_calorie_query(text)


def detect_allergen_terms(text: str) -> List[str]:
    norm = _normalize(text)
    canonical = ["난류", "우유", "대두", "밀", "토마토", "닭고기", "쇠고기", "돼지고기", "새우", "굴"]
    synonyms: Dict[str, str] = {
        "계란": "난류",
        "달걀": "난류",
        "유제품": "우유",
        "치즈": "우유",
        "콩": "대두",
        "간장": "대두",
        "글루텐": "밀",
        "소맥": "밀",
        "소고기": "쇠고기",
        "한우": "쇠고기",
        "비프": "쇠고기",
        "포크": "돼지고기",
        "돈육": "돼지고기",
        "치킨": "닭고기",
        "shrimp": "새우",
        "oyster": "굴",
        "milk": "우유",
        "soy": "대두",
        "wheat": "밀",
        "tomato": "토마토",
        "chicken": "닭고기",
        "beef": "쇠고기",
        "pork": "돼지고기",
        "egg": "난류",
        "eggs": "난류",
    }
    found = [c for c in canonical if _normalize(c) in norm]
    for k, v in synonyms.items():
        if _normalize(k) in norm and v not in found:
            found.append(v)
    return found


def wants_exclusion(text: str) -> bool:
    t = _normalize(text)
    return any(
        tok in t
        for tok in [
            "없는",
            "빼고",
            "제외",
            "못먹",
            "안먹",
            "안들어가",
            "안들어간",
            "안들어가는",
            "들어가지않은",
            "들어가지않는",
            "미포함",
            "불포함",
            "제외된",
            "without",
            "notinclude",
        ]
    )

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from v2.menu_info_intent import is_menu_info_query


_ALIASES = {
    "고울슬로": "코울슬로",
    "홀슬로": "코울슬로",
    "콜슬로": "코울슬로",
    "콜슬로우": "코울슬로",
    "상바구니": "장바구니",
    "청바구니": "장바구니",
}

_CATEGORY_HINTS: Dict[str, List[str]] = {
    "set": ["세트"],
    "single": ["단품", "버거"],
    "chicken": ["치킨"],
    "side": ["사이드", "감자", "코울슬로"],
    "drink": ["음료", "콜라", "사이다", "커피"],
}

_ALLOWED_INTENTS_BY_STAGE: Dict[str, set[str]] = {
    "ASK_DINING_TYPE": {"SET_DINING", "CHECK_CART", "CONTINUE_ORDER", "NONE", "MENU_INFO", "RECOMMEND"},
    "MAIN_MENU": {
        "ADD_MENU",
        "CHECK_CART",
        "CHECKOUT",
        "SELECT_PAYMENT",
        "NAVIGATE_CATEGORY",
        "SET_DINING",
        "CONTINUE_ORDER",
        "MENU_INFO",
        "RECOMMEND",
        "NONE",
    },
    "ORDER_REVIEW": {"CHECK_CART", "CHECKOUT", "SELECT_PAYMENT", "CONTINUE_ORDER", "NONE"},
    "SIDE_SELECTION": {"ADD_MENU", "CHECK_CART", "NAVIGATE_CATEGORY", "NONE"},
    "DRINK_SELECTION": {"ADD_MENU", "CHECK_CART", "NAVIGATE_CATEGORY", "NONE"},
    "PAYMENT": {"SELECT_PAYMENT", "CHECK_CART", "CHECKOUT", "NONE"},
    "PROACTIVE_HELP": {"CONTINUE_ORDER", "CHECK_CART", "NONE"},
}


@dataclass
class ParsedIntent:
    intent: str
    confidence: float
    menu_item_id: Optional[str] = None
    quantity: int = 1
    category_key: Optional[str] = None
    payment_method: Optional[str] = None
    dining_type: Optional[str] = None
    reason: str = ""
    normalized_text: str = ""


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    for src, dst in _ALIASES.items():
        t = t.replace(src, dst)
    t = re.sub(r"[.,!?~，。、！？]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _infer_stage(state: Dict[str, Any]) -> str:
    explicit_stage = str(state.get("stage") or "").strip().upper()
    if explicit_stage in _ALLOWED_INTENTS_BY_STAGE:
        return explicit_stage
    dining_type = str(state.get("diningType") or "").upper()
    page_hint = state.get("pageHint") if isinstance(state.get("pageHint"), dict) else {}
    payment_step = str(page_hint.get("paymentStep") or "").lower()
    show_order_view = bool(page_hint.get("showOrderView"))
    category = str(state.get("selectedCategory") or page_hint.get("selectedCategory") or "").lower()

    if not dining_type:
        return "ASK_DINING_TYPE"
    if payment_step in {"select", "method", "processing", "confirm"}:
        return "PAYMENT"
    if show_order_view:
        return "ORDER_REVIEW"
    if category in {"side", "cat_side"}:
        return "SIDE_SELECTION"
    if category in {"drink", "cat_drink"}:
        return "DRINK_SELECTION"
    return "MAIN_MENU"


def _extract_qty(text: str) -> int:
    m = re.search(r"(\d+)\s*개?", text)
    if not m:
        return 1
    try:
        return max(1, int(m.group(1)))
    except Exception:
        return 1


def _resolve_menu_mention(normalized_text: str, menu_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not normalized_text:
        return None
    q = _compact(normalized_text)
    if len(q) < 2:
        return None
    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for it in menu_items:
        if not isinstance(it, dict):
            continue
        name = _compact(_normalize(str(it.get("name") or "")))
        if not name:
            continue
        if q in name or name in q:
            return it
        score = difflib.SequenceMatcher(a=q[:40], b=name[:40]).ratio()
        if score > best_score:
            best = it
            best_score = score
    if best is not None and best_score >= 0.62:
        return best
    return None


def _has_add_intent(compact: str) -> bool:
    return any(tok in compact for tok in ["담아", "추가", "주문", "넣어", "줘"])


def _looks_like_complaint(compact: str) -> bool:
    return any(tok in compact for tok in ["왜", "또", "시켰", "담았", "알아", "냐", "니", "뭐야"])


def parse_state_intent(
    user_text: str,
    state: Dict[str, Any],
    menu_items: List[Dict[str, Any]],
) -> ParsedIntent:
    text = _normalize(user_text)
    compact = _compact(text)
    stage = _infer_stage(state if isinstance(state, dict) else {})
    allowed = _ALLOWED_INTENTS_BY_STAGE.get(stage, _ALLOWED_INTENTS_BY_STAGE["MAIN_MENU"])
    qty = _extract_qty(text)

    if not text:
        return ParsedIntent(intent="NONE", confidence=1.0, reason="empty", normalized_text=text)

    if "장바구니" in compact or ("담겨" in compact and ("뭐" in compact or "무엇" in compact)):
        intent = "CHECK_CART" if "CHECK_CART" in allowed else "NONE"
        return ParsedIntent(intent=intent, confidence=0.97, reason="cart_query", normalized_text=text)

    if any(tok in compact for tok in ["포장", "테이크아웃"]):
        intent = "SET_DINING" if "SET_DINING" in allowed else "NONE"
        return ParsedIntent(intent=intent, confidence=0.95, dining_type="TAKE_OUT", reason="dining_takeout", normalized_text=text)
    if any(tok in compact for tok in ["매장", "먹고", "여기서"]):
        intent = "SET_DINING" if "SET_DINING" in allowed else "NONE"
        return ParsedIntent(intent=intent, confidence=0.95, dining_type="DINE_IN", reason="dining_dinein", normalized_text=text)

    for key, hints in _CATEGORY_HINTS.items():
        if any(h in compact for h in hints) and any(h2 in compact for h2 in ["보여", "메뉴", "카테고리"]):
            intent = "NAVIGATE_CATEGORY" if "NAVIGATE_CATEGORY" in allowed else "NONE"
            return ParsedIntent(intent=intent, confidence=0.96, category_key=key, reason="category_nav", normalized_text=text)

    if any(tok in compact for tok in ["카드결제", "카드로결제", "카드결제할게", "카드결제할래"]):
        intent = "SELECT_PAYMENT" if "SELECT_PAYMENT" in allowed else "NONE"
        return ParsedIntent(intent=intent, confidence=0.95, payment_method="CARD", reason="pay_card", normalized_text=text)
    if any(tok in compact for tok in ["포인트결제", "포인트로결제"]):
        intent = "SELECT_PAYMENT" if "SELECT_PAYMENT" in allowed else "NONE"
        return ParsedIntent(intent=intent, confidence=0.95, payment_method="POINT", reason="pay_point", normalized_text=text)
    if "결제" in compact:
        intent = "CHECKOUT" if "CHECKOUT" in allowed else "NONE"
        return ParsedIntent(intent=intent, confidence=0.90, reason="checkout", normalized_text=text)

    if is_menu_info_query(text):
        intent = "MENU_INFO" if "MENU_INFO" in allowed else "NONE"
        return ParsedIntent(intent=intent, confidence=0.92, reason="menu_info", normalized_text=text)

    if "추천" in compact:
        intent = "RECOMMEND" if "RECOMMEND" in allowed else "NONE"
        return ParsedIntent(intent=intent, confidence=0.9, reason="recommend", normalized_text=text)

    menu_mention = _resolve_menu_mention(text, menu_items)
    if menu_mention is not None:
        if _looks_like_complaint(compact):
            intent = "CHECK_CART" if "CHECK_CART" in allowed else "NONE"
            return ParsedIntent(intent=intent, confidence=0.85, reason="complaint_with_menu", normalized_text=text)
        if _has_add_intent(compact) or stage in {"SIDE_SELECTION", "DRINK_SELECTION"}:
            intent = "ADD_MENU" if "ADD_MENU" in allowed else "NONE"
            return ParsedIntent(
                intent=intent,
                confidence=0.9,
                menu_item_id=str(menu_mention.get("menuItemId") or ""),
                quantity=qty,
                reason="menu_add",
                normalized_text=text,
            )
        # Mention-only defaults to browse/help, not immediate add.
        intent = "CONTINUE_ORDER" if "CONTINUE_ORDER" in allowed else "NONE"
        return ParsedIntent(intent=intent, confidence=0.7, reason="menu_mention_no_add_verb", normalized_text=text)

    if any(tok in compact for tok in ["메뉴", "보여", "다시"]):
        intent = "CONTINUE_ORDER" if "CONTINUE_ORDER" in allowed else "NONE"
        return ParsedIntent(intent=intent, confidence=0.7, reason="browse", normalized_text=text)

    return ParsedIntent(intent="NONE", confidence=0.5, reason="unknown", normalized_text=text)

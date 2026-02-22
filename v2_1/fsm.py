from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from v2_1.intent_parser import ParsedIntent


@dataclass
class FsmDecision:
    apply: bool
    blocked: bool = False
    reason: str = ""
    block_speech: str = ""


_ALLOWED_BY_STAGE: Dict[str, set[str]] = {
    "ASK_DINING_TYPE": {"SET_DINING", "CHECK_CART", "CONTINUE_ORDER", "MENU_INFO", "RECOMMEND", "ADD_MENU", "NONE"},
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
    "SIDE_SELECTION": {"ADD_MENU", "NAVIGATE_CATEGORY", "CHECK_CART", "NONE"},
    "DRINK_SELECTION": {"ADD_MENU", "NAVIGATE_CATEGORY", "CHECK_CART", "NONE"},
    "PAYMENT": {"SELECT_PAYMENT", "CHECK_CART", "CHECKOUT", "NONE"},
    "PROACTIVE_HELP": {"CONTINUE_ORDER", "CHECK_CART", "NONE"},
}

_CONF_THRESHOLDS: Dict[str, float] = {
    "ADD_MENU": 0.86,
    "CHECKOUT": 0.90,
    "SELECT_PAYMENT": 0.90,
    "SET_DINING": 0.85,
    "CHECK_CART": 0.80,
    "NAVIGATE_CATEGORY": 0.80,
    "CONTINUE_ORDER": 0.92,
}


def evaluate_fsm_gate(
    parsed: ParsedIntent,
    stage: str,
    cart_count: int,
) -> FsmDecision:
    intent = str(parsed.intent or "NONE").upper()
    if intent == "NONE":
        return FsmDecision(apply=False, reason="none")

    allowed = _ALLOWED_BY_STAGE.get(stage, _ALLOWED_BY_STAGE["MAIN_MENU"])
    if intent not in allowed:
        return FsmDecision(
            apply=False,
            blocked=True,
            reason="intent_not_allowed_in_stage",
            block_speech="지금 단계에서는 해당 요청을 처리할 수 없어요. 현재 화면 기준으로 다시 말씀해 주세요.",
        )

    if intent in {"CHECKOUT", "SELECT_PAYMENT"} and cart_count <= 0:
        return FsmDecision(
            apply=False,
            blocked=True,
            reason="empty_cart_payment_block",
            block_speech="장바구니가 비어 있어요. 먼저 메뉴를 담아주세요.",
        )

    threshold = _CONF_THRESHOLDS.get(intent, 0.80)
    if float(parsed.confidence or 0.0) < threshold:
        return FsmDecision(
            apply=False,
            reason=f"low_confidence:{parsed.confidence:.2f}<{threshold:.2f}",
        )

    return FsmDecision(apply=True, reason="ok")


from v2_1.fsm import evaluate_fsm_gate
from v2_1.intent_parser import ParsedIntent


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run() -> None:
    d1 = evaluate_fsm_gate(
        ParsedIntent(intent="ADD_MENU", confidence=0.95, menu_item_id="x"),
        stage="MAIN_MENU",
        cart_count=0,
    )
    _assert(d1.apply, "ADD_MENU on MAIN_MENU should pass")

    d2 = evaluate_fsm_gate(
        ParsedIntent(intent="SELECT_PAYMENT", confidence=0.97, payment_method="CARD"),
        stage="MAIN_MENU",
        cart_count=0,
    )
    _assert(d2.blocked, "SELECT_PAYMENT with empty cart should be blocked")

    d3 = evaluate_fsm_gate(
        ParsedIntent(intent="SELECT_PAYMENT", confidence=0.97, payment_method="CARD"),
        stage="PAYMENT",
        cart_count=1,
    )
    _assert(d3.apply, "SELECT_PAYMENT on PAYMENT with cart should pass")

    d4 = evaluate_fsm_gate(
        ParsedIntent(intent="ADD_MENU", confidence=0.96, menu_item_id="x"),
        stage="PAYMENT",
        cart_count=1,
    )
    _assert(d4.blocked, "ADD_MENU on PAYMENT stage should be blocked")

    d5 = evaluate_fsm_gate(
        ParsedIntent(intent="CHECK_CART", confidence=0.61),
        stage="MAIN_MENU",
        cart_count=1,
    )
    _assert(not d5.apply and not d5.blocked, "low confidence should defer")

    print("test_v2_1_fsm: PASS")


if __name__ == "__main__":
    run()


from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from v2.menu_info_intent import is_menu_info_query
from v2.orchestrator import V2LangChainOrchestrator


class StudioState(TypedDict, total=False):
    user_text: str
    state: Dict[str, Any]
    menu_items: List[Dict[str, Any]]
    stage: str
    info_done: bool
    route: str
    done: bool
    result: Dict[str, Any]


_DEFAULT_MENU_ITEMS: List[Dict[str, Any]] = [
    {"menuItemId": "set_1", "name": "징거세트", "categoryId": "cat_set", "price": 8900},
    {"menuItemId": "burger_1", "name": "징거버거", "categoryId": "cat_burger", "price": 6200},
    {"menuItemId": "side_1", "name": "감자튀김", "categoryId": "cat_side", "price": 2500},
    {"menuItemId": "drink_1", "name": "콜라", "categoryId": "cat_drink", "price": 1900},
]


async def _menu_list_provider() -> List[Dict[str, Any]]:
    return _DEFAULT_MENU_ITEMS


async def _menu_detail_provider(menu_item_id: str) -> Optional[Dict[str, Any]]:
    _ = menu_item_id
    return None


orch = V2LangChainOrchestrator(
    menu_list_provider=_menu_list_provider,
    menu_detail_provider=_menu_detail_provider,
)


def _safe_state(graph_state: StudioState) -> Dict[str, Any]:
    state = graph_state.get("state")
    if isinstance(state, dict):
        return state
    return {}


def _safe_text(graph_state: StudioState) -> str:
    return str(graph_state.get("user_text") or "").strip()


def _safe_menu_items(graph_state: StudioState) -> List[Dict[str, Any]]:
    items = graph_state.get("menu_items")
    if isinstance(items, list) and items:
        return [x for x in items if isinstance(x, dict)]
    return _DEFAULT_MENU_ITEMS


def session_init_node(graph_state: StudioState) -> Dict[str, Any]:
    stage = orch._infer_stage(_safe_state(graph_state))
    return {"stage": stage}


async def info_node(graph_state: StudioState) -> Dict[str, Any]:
    text = _safe_text(graph_state)
    state = _safe_state(graph_state)
    menu_items = _safe_menu_items(graph_state)
    menu_mention = orch._resolve_menu_mention(text, menu_items)

    # Keep recommendation-intent on recommendation route.
    if menu_mention is None and is_menu_info_query(text) and orch._is_recommendation_query(text):
        return {"info_done": False}

    info = await orch._run_info_tools(text, menu_items, session_id="studio")
    if not info:
        return {"info_done": False}

    stage = orch._infer_stage(state)
    payload = dict(info)
    payload.setdefault("stage", stage)
    return {
        "info_done": True,
        "done": True,
        "result": orch._decorate_parallel_channels(
            payload,
            emotion="neutral",
            expression="attentive",
            motion="listen",
        ),
    }


def route_node(graph_state: StudioState) -> Dict[str, Any]:
    text = _safe_text(graph_state)
    state = _safe_state(graph_state)
    if orch._is_recommendation_query(text):
        return {"route": "recommend"}
    if orch._is_hesitation_signal(text, state):
        return {"route": "policy"}
    return {"route": "policy"}


def recommend_node(graph_state: StudioState) -> Dict[str, Any]:
    text = _safe_text(graph_state)
    state = _safe_state(graph_state)
    rec = orch._build_vector_recommendation_response(text, state)
    if rec:
        return {"done": True, "result": rec}
    return {"done": False}


def policy_node(graph_state: StudioState) -> Dict[str, Any]:
    text = _safe_text(graph_state)
    state = _safe_state(graph_state)
    menu_items = _safe_menu_items(graph_state)
    dining_type = str(state.get("diningType") or "").upper()
    menu_mention = orch._resolve_menu_mention(text, menu_items)
    quantity = orch._extract_quantity(text)

    if orch._is_hesitation_signal(text, state):
        return {
            "done": True,
            "result": orch._decorate_parallel_channels(
                {
                    "speech": "도움이 필요하시면 말씀해 주세요. 주문을 단계별로 도와드릴게요.",
                    "action": "NONE",
                    "actionData": {"stage": "PROACTIVE_HELP", "proactiveHelp": True},
                    "intent": "PROACTIVE_HELP",
                    "stage": "PROACTIVE_HELP",
                },
                emotion="supportive",
                expression="soft_smile",
                motion="offer_help",
            ),
        }

    if not dining_type:
        if any(tok in text for tok in ["매장", "먹고", "여기서"]):
            return {
                "done": True,
                "result": orch._decorate_parallel_channels(
                    {
                        "speech": "매장 식사로 진행할게요. 원하시는 메뉴를 말씀해 주세요.",
                        "action": "SET_DINING",
                        "actionData": {"diningType": "DINE_IN", "stage": "MAIN_MENU"},
                        "intent": "ORDER_FLOW",
                        "stage": "MAIN_MENU",
                    },
                    emotion="confident",
                    expression="smile",
                    motion="confirm",
                ),
            }
        if any(tok in text for tok in ["포장", "가져", "테이크아웃"]):
            return {
                "done": True,
                "result": orch._decorate_parallel_channels(
                    {
                        "speech": "포장으로 진행할게요. 원하시는 메뉴를 말씀해 주세요.",
                        "action": "SET_DINING",
                        "actionData": {"diningType": "TAKE_OUT", "stage": "MAIN_MENU"},
                        "intent": "ORDER_FLOW",
                        "stage": "MAIN_MENU",
                    },
                    emotion="confident",
                    expression="smile",
                    motion="confirm",
                ),
            }

    method = orch._extract_payment_method(text)
    if method:
        return {
            "done": True,
            "result": orch._decorate_parallel_channels(
                {
                    "speech": "선택하신 결제 수단으로 진행할게요.",
                    "action": "SELECT_PAYMENT",
                    "actionData": {"method": method, "stage": "PAYMENT"},
                    "intent": "ORDER_FLOW",
                    "stage": "PAYMENT",
                },
                emotion="confident",
                expression="smile",
                motion="confirm",
            ),
        }

    if orch._is_checkout_intent(text):
        return {
            "done": True,
            "result": orch._decorate_parallel_channels(
                {
                    "speech": "주문 내역 화면으로 이동할게요. 확인 후 결제를 진행해 주세요.",
                    "action": "CHECKOUT",
                    "actionData": {"stage": "ORDER_REVIEW"},
                    "intent": "ORDER_FLOW",
                    "stage": "ORDER_REVIEW",
                },
                emotion="confident",
                expression="smile",
                motion="confirm",
            ),
        }

    if menu_mention is not None:
        menu_name = str(menu_mention.get("name") or "선택한 메뉴")
        menu_item_id = str(menu_mention.get("menuItemId") or "")
        if menu_item_id:
            action_data: Dict[str, Any] = {"menuItemId": menu_item_id, "quantity": quantity, "stage": "CART"}
            if orch._is_set_like(menu_mention):
                action_data["nextStage"] = "SIDE_SELECTION"
            return {
                "done": True,
                "result": orch._decorate_parallel_channels(
                    {
                        "speech": f"{menu_name} {quantity}개를 선택했어요.",
                        "action": "ADD_MENU",
                        "actionData": action_data,
                        "intent": "ORDER_FLOW",
                        "stage": "CART",
                    },
                    emotion="happy",
                    expression="smile",
                    motion="confirm",
                ),
            }

    return {"done": False}


def plan_node(graph_state: StudioState) -> Dict[str, Any]:
    _ = graph_state
    return {
        "result": {
            "speech": "원하시는 메뉴를 다시 말씀해 주세요.",
            "action": "NONE",
            "actionData": {},
            "intent": "GENERAL",
            "stage": "MAIN_MENU",
        }
    }


def build_graph():
    graph = StateGraph(StudioState)
    graph.add_node("session_init", session_init_node)
    graph.add_node("info", info_node)
    graph.add_node("route", route_node)
    graph.add_node("recommend", recommend_node)
    graph.add_node("policy", policy_node)
    graph.add_node("plan", plan_node)

    graph.set_entry_point("session_init")
    graph.add_edge("session_init", "info")
    graph.add_conditional_edges(
        "info",
        lambda s: "done" if bool(s.get("info_done")) else "route",
        {
            "done": END,
            "route": "route",
        },
    )
    graph.add_conditional_edges(
        "route",
        lambda s: s.get("route", "policy"),
        {
            "recommend": "recommend",
            "policy": "policy",
        },
    )
    graph.add_conditional_edges(
        "recommend",
        lambda s: "done" if bool(s.get("done")) else "policy",
        {
            "done": END,
            "policy": "policy",
        },
    )
    graph.add_conditional_edges(
        "policy",
        lambda s: "done" if bool(s.get("done")) else "plan",
        {
            "done": END,
            "plan": "plan",
        },
    )
    graph.add_edge("plan", END)
    return graph.compile()


graph = build_graph()

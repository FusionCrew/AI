from __future__ import annotations

import json
import logging
import os
import re
import time
import asyncio
import random
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from v2.menu_info_intent import (
    detect_allergen_terms,
    is_allergy_query,
    is_calorie_query,
    is_ingredient_query,
    is_menu_info_query,
    wants_exclusion,
)
from v2.live2d_mapping import ALL_MOTIONS, pick_live2d_profile
from v2.tracing_utils import traceable_safe
from v2_1.intent_parser import ParsedIntent, parse_state_intent
from v2_1.fsm import evaluate_fsm_gate

logger = logging.getLogger(__name__)

AllowedAction = {
    "NONE",
    "NAVIGATE",
    "NAVIGATE_CATEGORY",
    "ADD_MENU",
    "ADD_TO_CART",
    "REMOVE_MENU",
    "REMOVE_FROM_CART",
    "CHANGE_QTY",
    "CHECK_CART",
    "CHECKOUT",
    "SELECT_PAYMENT",
    "CONTINUE_ORDER",
    "SET_DINING",
    "CALL_STAFF",
}


class StructuredAction(BaseModel):
    speech: str = Field(default="원하시는 메뉴를 다시 말씀해 주세요.")
    action: str = Field(default="NONE")
    actionData: Dict[str, Any] = Field(default_factory=dict)


class V2LangChainOrchestrator:
    """
    v2 chat orchestrator:
    - tool fast-path (menu/allergy/facts)
    - optional vector candidate retrieval on ambiguous queries
    - LangGraph planner
    - OpenAI fallback
    """

    def __init__(
        self,
        menu_list_provider: Callable[[], Awaitable[List[Dict[str, Any]]]],
        menu_detail_provider: Optional[Callable[[str], Awaitable[Optional[Dict[str, Any]]]]] = None,
        vector_search_provider: Optional[Callable[[str, int], Awaitable[Dict[str, Any]]]] = None,
        model_env_var: str = "OPENAI_CHAT_MODEL_V2",
        fallback_model_env_var: str = "OPENAI_CHAT_MODEL",
    ) -> None:
        self.menu_list_provider = menu_list_provider
        self.menu_detail_provider = menu_detail_provider
        self.vector_search_provider = vector_search_provider
        self.model_env_var = model_env_var
        self.fallback_model_env_var = fallback_model_env_var
        self._langgraph_disabled_until_ts: float = 0.0
        self._langgraph_warned_in_window: bool = False
        self._last_openai_api_key: str = ""
        self.enable_v21_parser = str(os.getenv("AI_V21_PARSER_ENABLED", "true")).strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        self.enable_v21_fsm = str(os.getenv("AI_V21_FSM_ENABLED", "true")).strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }

    def set_vector_search_provider(
        self,
        provider: Optional[Callable[[str, int], Awaitable[Dict[str, Any]]]],
    ) -> None:
        self.vector_search_provider = provider

    @traceable_safe(
        name="aikiosk.v2.orchestrator.run",
        run_type="chain",
        tags=["aikiosk", "v2", "orchestrator", "voice-order"],
    )
    async def run(
        self,
        request: Any,
        openai_client: Any,
        request_id: str,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        trace: List[Dict[str, Any]] = []
        client_api_key = str(getattr(openai_client, "api_key", "") or "")
        if client_api_key:
            self._last_openai_api_key = client_api_key
        self._ensure_langgraph_openai_env()

        def mark(step: str, extra: Optional[Dict[str, Any]] = None) -> None:
            item = {"step": step, "elapsedMs": int((time.perf_counter() - started) * 1000)}
            if extra:
                item.update(extra)
            trace.append(item)

        session_id = (
            getattr(request, "sessionId", None)
            or (getattr(getattr(request, "context", None), "sessionId", None))
            or "default"
        )
        order_type = getattr(request, "orderType", None) or "UNKNOWN"
        messages = getattr(request, "messages", []) or []
        latest_user = self._latest_user_text(messages)
        mark("prepare", {"sessionId": session_id, "hasUserText": bool(latest_user)})
        if not latest_user:
            return {
                "result": self._normalize_output(
                    {"speech": "?ъ슜??諛쒗솕媛 鍮꾩뼱 ?덉뼱??", "action": "NONE", "actionData": {}},
                    "tool-fastpath",
                ),
                "trace": trace,
            }

        state = getattr(getattr(request, "context", None), "state", None) or {}
        menu_items = self._extract_menu_catalog_from_state(state)
        mark("state_catalog", {"count": len(menu_items)})
        if not menu_items:
            try:
                menu_items = await self.menu_list_provider()
                mark("menu_list_fetch", {"count": len(menu_items), "source": "provider"})
            except Exception:
                menu_items = []
                mark("menu_list_fetch", {"count": 0, "source": "provider", "error": True})

        fastpath = await self._run_info_tools(latest_user, menu_items, session_id)
        if fastpath:
            mark("tool_fastpath_hit")
            return {
                "result": self._normalize_output(fastpath, "tool-fastpath"),
                "trace": trace,
            }
        mark("tool_fastpath_miss")

        # Auto vector retrieval for ambiguous queries.
        working_state = dict(state) if isinstance(state, dict) else {}
        if (self._is_ambiguous_query(latest_user) or self._is_recommendation_query(latest_user)) and self.vector_search_provider is not None:
            try:
                vs = await self.vector_search_provider(latest_user, 6)
                candidates = vs.get("results") if isinstance(vs, dict) else None
                if isinstance(candidates, list) and candidates:
                    working_state["vectorCandidates"] = candidates[:6]
                    mark("vector_search_hit", {"count": len(candidates)})
                else:
                    mark("vector_search_miss")
            except Exception:
                logger.exception("v2 vector search failed")
                mark("vector_search_error")
        else:
            mark("vector_search_skip")

        # For allergen-constrained recommendation queries, enrich catalog with
        # menu-detail allergens so filtering is DB-grounded and not stale.
        if self._is_recommendation_query(latest_user):
            rec_state = dict(working_state) if isinstance(working_state, dict) else {}
            if isinstance(menu_items, list):
                rec_state["menuCatalog"] = [dict(it) for it in menu_items if isinstance(it, dict)]
            rec_allergen_terms = self._detect_query_allergen_terms(latest_user, rec_state)
            rec_ingredient_vocab = self._build_ingredient_vocab_from_state(rec_state)
            rec_ingredient_term = self._extract_recommend_ingredient_term(
                latest_user,
                ingredient_vocab=rec_ingredient_vocab,
                exclude_terms=rec_allergen_terms,
            )
            rec_want_exclude = self._wants_exclusion(latest_user)
            mark(
                "recommendation_parse",
                {
                    "exclude": rec_want_exclude,
                    "allergens": rec_allergen_terms,
                    "ingredientTerm": rec_ingredient_term,
                },
            )
            if rec_allergen_terms or rec_ingredient_term:
                enrich_timeout_sec = float(os.getenv("AI_RECOMMEND_ENRICH_TIMEOUT_SEC", "8.0"))
                try:
                    menu_items = await asyncio.wait_for(
                        self._enrich_menu_items_with_allergens(menu_items),
                        timeout=max(0.2, enrich_timeout_sec),
                    )
                    mark(
                        "recommendation_allergen_enriched",
                        {"count": len(menu_items), "timeoutSec": enrich_timeout_sec},
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "v2 recommendation allergen enrichment timed out (timeoutSec=%s)",
                        enrich_timeout_sec,
                    )
                    mark(
                        "recommendation_allergen_enrich_timeout",
                        {"count": len(menu_items), "timeoutSec": enrich_timeout_sec},
                    )
                except Exception:
                    logger.exception("v2 recommendation allergen enrichment failed")
                    mark("recommendation_allergen_enrich_error", {"count": len(menu_items)})
                if isinstance(working_state, dict):
                    working_state["menuCatalog"] = [dict(it) for it in menu_items if isinstance(it, dict)]

        if self._can_use_langgraph():
            try:
                out = await self._run_langgraph(messages, latest_user, working_state, menu_items, order_type)
                mark("langgraph_success")
                self._langgraph_warned_in_window = False
                self._langgraph_disabled_until_ts = 0.0
                return {
                    "result": self._normalize_output(out, "langgraph"),
                    "trace": trace,
                }
            except Exception:
                if not self._langgraph_warned_in_window:
                    logger.exception("v2 langgraph path failed; falling back to direct OpenAI")
                    self._langgraph_warned_in_window = True
                mark("langgraph_error")
                self._langgraph_disabled_until_ts = time.time() + 60.0

        policy = await self._run_stage_policy(latest_user, working_state, menu_items)
        if policy:
            mark("stage_policy_hit", {"stage": str(policy.get("stage") or "UNKNOWN")})
            return {
                "result": self._normalize_output(policy, "stage-policy"),
                "trace": trace,
            }

        out = await self._run_openai_fallback(openai_client, messages, latest_user, working_state, menu_items, order_type)
        mark("openai_fallback_success", {"requestId": request_id})
        return {
            "result": self._normalize_output(out, "openai-fallback"),
            "trace": trace,
        }

    def _can_use_langgraph(self) -> bool:
        if time.time() < self._langgraph_disabled_until_ts:
            return False
        if not os.getenv("OPENAI_API_KEY"):
            return False
        try:
            import langgraph  # noqa: F401
            import langchain_openai  # noqa: F401
            import langchain_core  # noqa: F401
            return True
        except Exception:
            return False

    def _ensure_langgraph_openai_env(self) -> None:
        """
        ChatOpenAI reads OPENAI_API_KEY from process env by default.
        Keep env in sync with already-created OpenAI client to avoid LangGraph-only key misses.
        """
        if self._last_openai_api_key:
            current = os.getenv("OPENAI_API_KEY")
            if current != self._last_openai_api_key:
                os.environ["OPENAI_API_KEY"] = self._last_openai_api_key

    @traceable_safe(name="v2.orchestrator.langgraph", run_type="chain")
    async def _run_langgraph(
        self,
        messages: List[Any],
        latest_user: str,
        state: Dict[str, Any],
        menu_items: List[Dict[str, Any]],
        order_type: str,
    ) -> Dict[str, Any]:
        from langchain_openai import ChatOpenAI
        from langgraph.graph import END, StateGraph

        model_name = os.getenv(self.model_env_var, os.getenv(self.fallback_model_env_var, "gpt-4o-mini"))
        llm = ChatOpenAI(model=model_name, temperature=0.1)
        system_prompt = self._build_system_prompt(order_type, state, menu_items)
        history_text = self._history_for_model(messages, latest_user)

        async def session_init_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
            stage = self._infer_stage(graph_state.get("state") or {})
            return {"stage": stage}

        async def route_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
            _ = graph_state
            text = latest_user
            state_obj = state if isinstance(state, dict) else {}
            if self._is_hesitation_signal(text, state_obj):
                return {"route": "policy"}
            if self._is_recommendation_query(text):
                return {"route": "recommend"}
            return {"route": "policy"}

        async def recommend_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
            _ = graph_state
            text = latest_user
            state_obj = state if isinstance(state, dict) else {}
            rec_allergens = self._detect_query_allergen_terms(text, state_obj)
            rec_ing = self._extract_recommend_ingredient_term(
                text,
                ingredient_vocab=self._build_ingredient_vocab_from_state(state_obj),
                exclude_terms=rec_allergens,
            )
            if (rec_allergens or rec_ing) and not self._has_catalog_ingredient_data(state_obj):
                logger.warning(
                    "v2.recommend.langgraph.enrich_on_demand trigger=true allergens=%s ingredient=%s",
                    rec_allergens,
                    rec_ing,
                )
                try:
                    enriched = await self._enrich_menu_items_with_allergens(menu_items if isinstance(menu_items, list) else [])
                    state_obj = dict(state_obj)
                    state_obj["menuCatalog"] = [dict(it) for it in enriched if isinstance(it, dict)]
                except Exception:
                    logger.exception("v2.recommend.langgraph.enrich_on_demand failed")
            result = self._build_vector_recommendation_response(text, state_obj)
            if result:
                return {"done": True, "result": result}
            return {"done": False}

        async def policy_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
            _ = graph_state
            result = await self._run_stage_policy(
                user_text=latest_user,
                state=state if isinstance(state, dict) else {},
                menu_items=menu_items if isinstance(menu_items, list) else [],
            )
            if result:
                return {"done": True, "result": result}
            return {"done": False}

        async def fsm_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
            _ = graph_state
            if not self.enable_v21_parser or not self.enable_v21_fsm:
                return {"done": False}
            state_obj = state if isinstance(state, dict) else {}
            stage_now = self._infer_stage(state_obj)
            cart_items = state_obj.get("cartItems") if isinstance(state_obj.get("cartItems"), list) else []
            parsed = parse_state_intent(
                user_text=latest_user,
                state=state_obj,
                menu_items=menu_items if isinstance(menu_items, list) else [],
            )
            out = self._run_v21_fsm_policy(
                parsed=parsed,
                stage=stage_now,
                menu_items=menu_items if isinstance(menu_items, list) else [],
                cart_items=[x for x in cart_items if isinstance(x, dict)],
            )
            if out:
                return {"done": True, "result": out}
            return {"done": False}

        async def plan_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
            _ = graph_state
            raw_msg = await llm.ainvoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{history_text}\n\n諛섎뱶??JSON留?異쒕젰?섏꽭??"},
                ]
            )
            raw = str(getattr(raw_msg, "content", "") or "").strip()
            if not raw:
                return {"result": {"speech": "죄송해요. 다시 말씀해 주세요.", "action": "NONE", "actionData": {}}}
            try:
                return {"result": json.loads(raw)}
            except Exception:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        return {"result": json.loads(raw[start : end + 1])}
                    except Exception:
                        pass
                return {"result": {"speech": raw, "action": "NONE", "actionData": {}}}

        graph = StateGraph(dict)
        graph.add_node("session_init", session_init_node)
        graph.add_node("route", route_node)
        graph.add_node("recommend", recommend_node)
        graph.add_node("fsm", fsm_node)
        graph.add_node("policy", policy_node)
        graph.add_node("plan", plan_node)
        graph.set_entry_point("session_init")
        graph.add_edge("session_init", "route")
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
            lambda s: "done" if bool(s.get("done")) else "fsm",
            {
                "done": END,
                "fsm": "fsm",
            },
        )
        graph.add_conditional_edges(
            "fsm",
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
        app = graph.compile()

        result = await app.ainvoke(
            {
                "history_text": history_text,
                "latest_user": latest_user,
                "state": state,
                "menu_items": menu_items,
            }
        )
        return result.get("result") or {}

    @traceable_safe(name="v2.orchestrator.openai_fallback", run_type="llm")
    async def _run_openai_fallback(
        self,
        openai_client: Any,
        messages: List[Any],
        latest_user: str,
        state: Dict[str, Any],
        menu_items: List[Dict[str, Any]],
        order_type: str,
    ) -> Dict[str, Any]:
        model_name = os.getenv(self.model_env_var, os.getenv(self.fallback_model_env_var, "gpt-4o-mini"))
        system_prompt = self._build_system_prompt(order_type, state, menu_items)
        history_text = self._history_for_model(messages, latest_user)
        content = f"{history_text}\n\n諛섎뱶??JSON留?異쒕젰?섏꽭??"

        resp = openai_client.chat.completions.create(
            model=model_name,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            return {"speech": "???ㅼ뿀?댁슂. ??踰???留먯???二쇱꽭??", "action": "NONE", "actionData": {}}

        try:
            return json.loads(raw)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except Exception:
                    pass
            return {"speech": raw, "action": "NONE", "actionData": {}}

    def _build_system_prompt(
        self,
        order_type: str,
        state: Dict[str, Any],
        menu_items: List[Dict[str, Any]],
    ) -> str:
        catalog = []
        for it in menu_items[:200]:
            mid = it.get("menuItemId")
            name = it.get("name")
            if mid and name:
                catalog.append(
                    {
                        "menuItemId": mid,
                        "name": name,
                        "categoryId": it.get("categoryId") or it.get("category"),
                        "price": it.get("price"),
                    }
                )
        vector_candidates = []
        if isinstance(state, dict):
            vc = state.get("vectorCandidates")
            if isinstance(vc, list):
                vector_candidates = vc[:6]
        stage = self._infer_stage(state if isinstance(state, dict) else {})

        return (
            "?덈뒗 ?ㅼ삤?ㅽ겕 ?뚯꽦 二쇰Ц ?ㅼ??ㅽ듃?덉씠?곕떎.\n"
            "JSON ?ㅽ궎留덈쭔 異쒕젰?대씪: {\"speech\": string, \"action\": string, \"actionData\": object}\n"
            f"?덉슜 action: {sorted(list(AllowedAction))}\n"
            "洹쒖튃: menuItemId??移댄깉濡쒓렇???덈뒗 媛믩쭔 ?ъ슜.\n"
            "vectorCandidates??紐⑦샇 吏덉쓽 ?꾨낫?대ŉ 理쒖쥌 ?뺤젙? 移댄깉濡쒓렇? 臾몃㎘?쇰줈 ?섎씪.\n"
            "二쇰Ц ?④퀎(stage)瑜?怨좊젮???듯빐?? 寃곗젣 ?꾩뿉 寃곗젣?섎떒 ?좏깮???좊룄?섏? 留덈씪.\n"
            "?묐떟?먮뒗 live2d 媛먯젙 ?좊땲硫붿씠??蹂묐젹 ?ㅽ뻾??怨좊젮?섎릺, action 寃곗젙???곗꽑?섎씪.\n"
            f"?꾩옱 二쇰Ц ??? {order_type}\n"
            f"異붾줎???④퀎(stage): {stage}\n"
            f"?꾩옱 ?곹깭: {json.dumps(state, ensure_ascii=False)}\n"
            f"메뉴 카탈로그: {json.dumps(catalog, ensure_ascii=False)}\n"
            f"Vector ?꾨낫: {json.dumps(vector_candidates, ensure_ascii=False)}\n"
        )

    def _history_for_model(self, messages: List[Any], latest_user: str) -> str:
        lines: List[str] = []
        for m in messages[-10:]:
            role = str(getattr(m, "role", "") or "").strip()
            content = str(getattr(m, "content", "") or "").strip()
            if role in ("user", "assistant") and content:
                lines.append(f"{role}: {content}")
        if not lines or not lines[-1].startswith("user:"):
            lines.append(f"user: {latest_user}")
        return "\n".join(lines[-12:])

    def _latest_user_text(self, messages: List[Any]) -> str:
        for m in reversed(messages):
            role = str(getattr(m, "role", "") or "").strip()
            if role == "user":
                return str(getattr(m, "content", "") or "").strip()
        return ""

    def _extract_menu_catalog_from_state(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(state, dict):
            return []
        catalog = state.get("menuCatalog")
        if isinstance(catalog, list):
            return [it for it in catalog if isinstance(it, dict)]
        return []

    async def _enrich_menu_items_with_allergens(self, menu_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(menu_items, list):
            return []
        if not self.menu_detail_provider:
            return [dict(it) for it in menu_items if isinstance(it, dict)]

        out = [dict(it) for it in menu_items if isinstance(it, dict)]
        alias_map: Dict[str, List[str]] = {}
        for it in out:
            mid = str(it.get("menuItemId") or "").strip()
            name = str(it.get("name") or "").strip()
            if not mid or not name:
                continue
            key = self._canonical_name(name)
            if not key:
                continue
            alias_map.setdefault(key, [])
            if mid not in alias_map[key]:
                alias_map[key].append(mid)
        # Include provider-wide aliases too (helps when incoming catalog contains only menu_ ids).
        try:
            provider_items = await self.menu_list_provider()
            if isinstance(provider_items, list):
                for it in provider_items:
                    if not isinstance(it, dict):
                        continue
                    mid = str(it.get("menuItemId") or "").strip()
                    name = str(it.get("name") or "").strip()
                    if not mid or not name:
                        continue
                    key = self._canonical_name(name)
                    if not key:
                        continue
                    alias_map.setdefault(key, [])
                    if mid not in alias_map[key]:
                        alias_map[key].append(mid)
        except Exception:
            # alias extension is best-effort.
            pass
        enrich_concurrency = int(os.getenv("AI_RECOMMEND_ENRICH_CONCURRENCY", "16"))
        sem = asyncio.Semaphore(max(1, enrich_concurrency))
        detail_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        stats: Dict[str, int] = {
            "total": len(out),
            "missingMenuItemId": 0,
            "detailNotDict": 0,
            "detailError": 0,
            "allergyFilled": 0,
            "ingredientFilled": 0,
            "aliasHit": 0,
        }

        async def _fetch_detail(mid: str) -> Optional[Dict[str, Any]]:
            if mid in detail_cache:
                return detail_cache[mid]
            async with sem:
                try:
                    detail = await self.menu_detail_provider(mid)
                except Exception:
                    stats["detailError"] += 1
                    detail = None
            detail_cache[mid] = detail if isinstance(detail, dict) else None
            return detail_cache[mid]

        async def enrich_one(it: Dict[str, Any]) -> None:
            mid = str(it.get("menuItemId") or "").strip()
            if not mid:
                stats["missingMenuItemId"] += 1
                return
            detail = await _fetch_detail(mid)
            if not isinstance(detail, dict):
                stats["detailNotDict"] += 1
                return
            normalized = self._normalize_menu_detail(detail, str(it.get("name") or ""))
            allergies = normalized.get("allergies") if isinstance(normalized, dict) else []
            ingredients = normalized.get("ingredients") if isinstance(normalized, dict) else []

            # Fallback: if this id has poor detail, try sibling ids with same canonical name.
            has_allergy_data = isinstance(allergies, list) and len(allergies) > 0
            has_ingredient_data = isinstance(ingredients, list) and len(ingredients) > 0
            if not (has_allergy_data and has_ingredient_data):
                key = self._canonical_name(str(it.get("name") or ""))
                sibling_ids = alias_map.get(key, []) if key else []
                for sid in sibling_ids:
                    if sid == mid:
                        continue
                    sdetail = await _fetch_detail(sid)
                    if not isinstance(sdetail, dict):
                        continue
                    snorm = self._normalize_menu_detail(sdetail, str(it.get("name") or ""))
                    sall = snorm.get("allergies") if isinstance(snorm, dict) else []
                    sing = snorm.get("ingredients") if isinstance(snorm, dict) else []
                    if isinstance(sall, list) and len(sall) > 0:
                        allergies = sall
                    if isinstance(sing, list) and len(sing) > 0:
                        ingredients = sing
                    if (isinstance(allergies, list) and allergies) and (isinstance(ingredients, list) and ingredients):
                        stats["aliasHit"] += 1
                        break

            if isinstance(allergies, list) and allergies:
                it["allergies"] = [str(x).strip() for x in allergies if str(x).strip()]
                if it.get("allergies"):
                    stats["allergyFilled"] += 1
            if isinstance(ingredients, list) and ingredients:
                it["ingredients"] = [str(x).strip() for x in ingredients if str(x).strip()]
                if it.get("ingredients"):
                    stats["ingredientFilled"] += 1

        tasks = [asyncio.create_task(enrich_one(it)) for it in out]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(
            "v2.recommend.enrich.summary total=%s missingMenuItemId=%s detailNotDict=%s detailError=%s allergyFilled=%s ingredientFilled=%s aliasHit=%s",
            stats["total"],
            stats["missingMenuItemId"],
            stats["detailNotDict"],
            stats["detailError"],
            stats["allergyFilled"],
            stats["ingredientFilled"],
            stats["aliasHit"],
        )
        return out

    def _build_cart_summary_speech(self, cart_items: List[Dict[str, Any]]) -> str:
        if not cart_items:
            return "현재 장바구니가 비어 있어요."
        chunks: List[str] = []
        for x in cart_items[:4]:
            if not isinstance(x, dict):
                continue
            name = str(x.get("name") or "메뉴").strip() or "메뉴"
            qty = max(1, int(x.get("quantity") or 1))
            chunks.append(f"{name} {qty}개")
        if not chunks:
            return "현재 장바구니가 비어 있어요."
        suffix = " 등이" if len(cart_items) > 4 else "가"
        return f"현재 장바구니에는 {', '.join(chunks)}{suffix} 담겨 있어요."

    def _run_v21_parser_policy(
        self,
        parsed: ParsedIntent,
        stage: str,
        menu_items: List[Dict[str, Any]],
        cart_items: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        intent = str(parsed.intent or "NONE").upper()
        if intent == "NONE":
            return None

        if intent == "CHECK_CART":
            return self._decorate_parallel_channels(
                {
                    "speech": self._build_cart_summary_speech(cart_items),
                    "action": "CHECK_CART",
                    "actionData": {"stage": "CART", "parser": "v2.1", "reason": parsed.reason},
                    "intent": "ORDER_FLOW",
                    "stage": "CART",
                },
                emotion="neutral",
                expression="attentive",
                motion="listen",
            )

        if intent == "NAVIGATE_CATEGORY" and parsed.category_key:
            return self._decorate_parallel_channels(
                {
                    "speech": "?붿껌?섏떊 移댄뀒怨좊━濡??대룞?좉쾶??",
                    "action": "NAVIGATE_CATEGORY",
                    "actionData": {
                        "categoryKey": parsed.category_key,
                        "stage": stage,
                        "parser": "v2.1",
                        "reason": parsed.reason,
                    },
                    "intent": "ORDER_FLOW",
                    "stage": stage,
                },
                emotion="neutral",
                expression="attentive",
                motion="listen",
            )

        if intent == "SET_DINING" and parsed.dining_type in {"DINE_IN", "TAKE_OUT"}:
            return self._decorate_parallel_channels(
                {
                    "speech": "?앹궗 諛⑹떇???ㅼ젙?좉쾶??",
                    "action": "SET_DINING",
                    "actionData": {
                        "diningType": parsed.dining_type,
                        "stage": "MAIN_MENU",
                        "parser": "v2.1",
                        "reason": parsed.reason,
                    },
                    "intent": "ORDER_FLOW",
                    "stage": "MAIN_MENU",
                },
                emotion="confident",
                expression="smile",
                motion="confirm",
            )

        if intent == "SELECT_PAYMENT" and parsed.payment_method in {"CARD", "POINT", "SIMPLE"}:
            return self._decorate_parallel_channels(
                {
                    "speech": "?좏깮??寃곗젣 ?섎떒?쇰줈 吏꾪뻾?좉쾶??",
                    "action": "SELECT_PAYMENT",
                    "actionData": {
                        "method": parsed.payment_method,
                        "stage": "PAYMENT",
                        "parser": "v2.1",
                        "reason": parsed.reason,
                    },
                    "intent": "ORDER_FLOW",
                    "stage": "PAYMENT",
                },
                emotion="confident",
                expression="smile",
                motion="confirm",
            )

        if intent == "CHECKOUT":
            return self._decorate_parallel_channels(
                {
                    "speech": "二쇰Ц ?댁뿭???뺤씤????寃곗젣瑜?吏꾪뻾?좉쾶??",
                    "action": "CHECKOUT",
                    "actionData": {"stage": "ORDER_REVIEW", "parser": "v2.1", "reason": parsed.reason},
                    "intent": "ORDER_FLOW",
                    "stage": "ORDER_REVIEW",
                },
                emotion="confident",
                expression="smile",
                motion="confirm",
            )

        if intent == "ADD_MENU" and parsed.menu_item_id:
            mentioned = None
            for x in menu_items:
                if str(x.get("menuItemId") or "") == parsed.menu_item_id:
                    mentioned = x
                    break
            menu_name = str((mentioned or {}).get("name") or "선택 메뉴")
            action_data: Dict[str, Any] = {
                "menuItemId": parsed.menu_item_id,
                "quantity": max(1, int(parsed.quantity or 1)),
                "stage": "CART",
                "parser": "v2.1",
                "reason": parsed.reason,
            }
            if mentioned is not None and self._is_set_like(mentioned):
                action_data["nextStage"] = "SIDE_SELECTION"
                speech = f"{menu_name} 선택했어요. 사이드와 음료를 고를게요."
            else:
                speech = f"{menu_name} {max(1, int(parsed.quantity or 1))}개를 장바구니에 담을게요."
            return self._decorate_parallel_channels(
                {
                    "speech": speech,
                    "action": "ADD_MENU",
                    "actionData": action_data,
                    "intent": "ORDER_FLOW",
                    "stage": "CART",
                },
                emotion="happy",
                expression="smile",
                motion="confirm",
            )

        if intent == "CONTINUE_ORDER":
            return self._decorate_parallel_channels(
                {
                    "speech": "메뉴를 계속 고를 수 있어요.",
                    "action": "CONTINUE_ORDER",
                    "actionData": {"stage": "MAIN_MENU", "parser": "v2.1", "reason": parsed.reason},
                    "intent": "ORDER_FLOW",
                    "stage": "MAIN_MENU",
                },
                emotion="happy",
                expression="smile",
                motion="nod",
            )

        return None

    def _run_v21_fsm_policy(
        self,
        parsed: ParsedIntent,
        stage: str,
        menu_items: List[Dict[str, Any]],
        cart_items: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        decision = evaluate_fsm_gate(parsed=parsed, stage=stage, cart_count=len(cart_items))
        if decision.blocked:
            return self._decorate_parallel_channels(
                {
                    "speech": decision.block_speech or "지금 단계에서는 해당 요청을 처리할 수 없어요.",
                    "action": "NONE",
                    "actionData": {
                        "stage": stage,
                        "parser": "v2.1",
                        "fsm": "v2.1",
                        "fsmBlocked": True,
                        "reason": decision.reason,
                    },
                    "intent": "ORDER_FLOW",
                    "stage": stage,
                },
                emotion="neutral",
                expression="attentive",
                motion="listen",
            )
        if not decision.apply:
            return None
        out = self._run_v21_parser_policy(
            parsed=parsed,
            stage=stage,
            menu_items=menu_items,
            cart_items=cart_items,
        )
        if isinstance(out, dict):
            action_data = out.get("actionData")
            if isinstance(action_data, dict):
                action_data.setdefault("fsm", "v2.1")
                action_data.setdefault("fsmReason", decision.reason)
        return out

    @traceable_safe(name="v2.orchestrator.stage_policy", run_type="chain")
    async def _run_stage_policy(
        self,
        user_text: str,
        state: Dict[str, Any],
        menu_items: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        text = (user_text or "").strip()
        if not text:
            return None

        stage = self._infer_stage(state)
        dining_type = str(state.get("diningType") or "").upper()
        page_hint = state.get("pageHint") if isinstance(state.get("pageHint"), dict) else {}
        cart_items = state.get("cartItems") if isinstance(state.get("cartItems"), list) else []
        cart_count = len([x for x in cart_items if isinstance(x, dict)])
        menu_mention = self._resolve_menu_mention(text, menu_items)
        quantity = self._extract_quantity(text)

        if self._is_hesitation_signal(text, state):
            return self._decorate_parallel_channels(
                {
                    "speech": "?꾩????꾩슂?섏떆硫?留먯???二쇱꽭?? 二쇰Ц???④퀎蹂꾨줈 ?꾩??쒕┫寃뚯슂.",
                    "action": "NONE",
                    "actionData": {"stage": "PROACTIVE_HELP", "proactiveHelp": True},
                    "intent": "PROACTIVE_HELP",
                    "stage": "PROACTIVE_HELP",
                },
                emotion="supportive",
                expression="soft_smile",
                motion="offer_help",
            )

        compact = re.sub(r"\s+", "", text)
        in_option_stage = stage in {"SIDE_SELECTION", "DRINK_SELECTION"}
        has_size = any(tok in compact for tok in ["라지", "미디엄", "스몰", "large", "medium", "small"])
        wants_drink = any(tok in compact for tok in ["음료", "콜라", "사이다", "제로", "커피"])
        wants_side = any(tok in compact for tok in ["사이드", "감자", "코울슬로", "치즈스틱"])
        if in_option_stage:
            if has_size:
                return self._decorate_parallel_channels(
                    {
                        "speech": "사이즈를 반영할게요.",
                        "action": "CHANGE_QTY",
                        "actionData": {"stage": stage, "optionType": "SIZE"},
                        "intent": "ORDER_FLOW",
                        "stage": stage,
                    },
                    emotion="confident",
                    expression="smile",
                    motion="confirm",
                )
            if wants_drink:
                return self._decorate_parallel_channels(
                    {
                        "speech": "음료 옵션으로 반영할게요.",
                        "action": "ADD_MENU",
                        "actionData": {"stage": "DRINK_SELECTION"},
                        "intent": "ORDER_FLOW",
                        "stage": "DRINK_SELECTION",
                    },
                    emotion="confident",
                    expression="smile",
                    motion="confirm",
                )
            if wants_side:
                return self._decorate_parallel_channels(
                    {
                        "speech": "사이드 옵션으로 반영할게요.",
                        "action": "ADD_MENU",
                        "actionData": {"stage": "SIDE_SELECTION"},
                        "intent": "ORDER_FLOW",
                        "stage": "SIDE_SELECTION",
                    },
                    emotion="confident",
                    expression="smile",
                    motion="confirm",
                )

        if self.enable_v21_parser:
            parsed = parse_state_intent(
                user_text=text,
                state=state if isinstance(state, dict) else {},
                menu_items=menu_items if isinstance(menu_items, list) else [],
            )
            if self.enable_v21_fsm:
                parsed_out = self._run_v21_fsm_policy(
                    parsed=parsed,
                    stage=stage,
                    menu_items=menu_items if isinstance(menu_items, list) else [],
                    cart_items=[x for x in cart_items if isinstance(x, dict)],
                )
                if parsed_out is not None:
                    return parsed_out
            else:
                # compatibility path: parser-only application
                parsed_out = self._run_v21_parser_policy(
                    parsed=parsed,
                    stage=stage,
                    menu_items=menu_items if isinstance(menu_items, list) else [],
                    cart_items=[x for x in cart_items if isinstance(x, dict)],
                )
                if parsed_out is not None:
                    return parsed_out

        if self._is_recommendation_query(text):
            state_obj = state if isinstance(state, dict) else {}
            rec_allergens = self._detect_query_allergen_terms(text, state_obj)
            rec_ing = self._extract_recommend_ingredient_term(
                text,
                ingredient_vocab=self._build_ingredient_vocab_from_state(state_obj),
                exclude_terms=rec_allergens,
            )
            if (rec_allergens or rec_ing) and not self._has_catalog_ingredient_data(state_obj):
                logger.warning(
                    "v2.recommend.stage_policy.enrich_on_demand trigger=true allergens=%s ingredient=%s",
                    rec_allergens,
                    rec_ing,
                )
                try:
                    enriched = await self._enrich_menu_items_with_allergens(menu_items if isinstance(menu_items, list) else [])
                    state_obj = dict(state_obj)
                    state_obj["menuCatalog"] = [dict(it) for it in enriched if isinstance(it, dict)]
                except Exception:
                    logger.exception("v2.recommend.stage_policy.enrich_on_demand failed")
            from_vector = self._build_vector_recommendation_response(text, state_obj)
            if from_vector:
                return from_vector

            allergen_terms = self._detect_allergen_terms(text)
            want_exclude = wants_exclusion(text)
            candidates = [it for it in menu_items if isinstance(it, dict)]
            if allergen_terms:
                filtered: List[Dict[str, Any]] = []
                for it in candidates:
                    known_allergens = set(self._candidate_known_allergens(it))
                    if not known_allergens:
                        # Safety-first: unknown allergen metadata is not treated as safe.
                        continue
                    if want_exclude and all(term not in known_allergens for term in allergen_terms):
                        filtered.append(it)
                    if (not want_exclude) and any(term in known_allergens for term in allergen_terms):
                        filtered.append(it)
                candidates = filtered

            names = [str(it.get("name") or "").strip() for it in candidates[:3] if str(it.get("name") or "").strip()]
            if names:
                filter_prefix = ""
                if allergen_terms:
                    mode = "제외" if want_exclude else "포함"
                    filter_prefix = f"{', '.join(allergen_terms)} {mode} 조건을 반영한 "
                return self._decorate_parallel_channels(
                    {
                        "speech": f"{filter_prefix}메뉴는 {', '.join(names)}가 있어요. 원하시는 메뉴 이름을 말씀해 주세요.",
                        "action": "NONE",
                        "actionData": {"stage": stage},
                        "intent": "RECOMMENDATION",
                        "stage": stage,
                    },
                    emotion="happy",
                    expression="smile",
                    motion="recommend",
                )

        # Guardrail: menu-info queries must never become ordering actions.
        if menu_mention is not None and self._is_menu_info_query(text):
            return self._decorate_parallel_channels(
                {
                    "speech": "메뉴 정보를 안내해드릴게요. 알레르기, 재료, 칼로리 중 어떤 정보를 원하시나요?",
                    "action": "NONE",
                    "actionData": {
                        "menuItemId": str(menu_mention.get("menuItemId") or ""),
                        "stage": stage,
                    },
                    "intent": "MENU_INFO",
                    "stage": stage,
                },
                emotion="neutral",
                expression="attentive",
                motion="listen",
            )

        if menu_mention is None and self._is_menu_info_query(text) and not self._is_recommendation_query(text):
            return self._decorate_parallel_channels(
                {
                    "speech": "어떤 메뉴의 정보를 확인할까요? 메뉴 이름과 함께 다시 말씀해 주세요.",
                    "action": "NONE",
                    "actionData": {"stage": stage},
                    "intent": "MENU_INFO",
                    "stage": stage,
                },
                emotion="neutral",
                expression="attentive",
                motion="listen",
            )

        if not dining_type:
            if any(tok in text for tok in ["매장", "먹고", "여기"]):
                return self._decorate_parallel_channels(
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
                )
            if any(tok in text for tok in ["포장", "가져", "테이크아웃"]):
                return self._decorate_parallel_channels(
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
                )
            if menu_mention is not None:
                return self._decorate_parallel_channels(
                    {
                        "speech": "좋아요. 먼저 매장 식사인지 포장인지 말씀해 주세요.",
                        "action": "NONE",
                        "actionData": {
                            "pendingMenuItemId": str(menu_mention.get("menuItemId") or ""),
                            "quantity": quantity,
                            "stage": "ASK_DINING_TYPE",
                        },
                        "intent": "ORDER_FLOW",
                        "stage": "ASK_DINING_TYPE",
                    },
                    emotion="neutral",
                    expression="attentive",
                    motion="listen",
                )

        method = self._extract_payment_method(text)
        if method:
            current_payment_step = str(page_hint.get("paymentStep") or "").lower()
            if cart_count <= 0 and current_payment_step not in {"select", "method", "confirm"}:
                return self._decorate_parallel_channels(
                    {
                        "speech": "결제할 주문이 없어요. 먼저 메뉴를 담아 주세요.",
                        "action": "NONE",
                        "actionData": {"stage": stage},
                        "intent": "ORDER_FLOW",
                        "stage": stage,
                    },
                    emotion="neutral",
                    expression="attentive",
                    motion="listen",
                )
            return self._decorate_parallel_channels(
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
            )

        if self._is_checkout_intent(text):
            if cart_count <= 0:
                return self._decorate_parallel_channels(
                    {
                        "speech": "아직 장바구니가 비어 있어요. 메뉴를 먼저 담아볼까요?",
                        "action": "NONE",
                        "actionData": {"stage": stage},
                        "intent": "ORDER_FLOW",
                        "stage": stage,
                    },
                    emotion="neutral",
                    expression="attentive",
                    motion="listen",
                )
            return self._decorate_parallel_channels(
                {
                    "speech": "二쇰Ц ?댁뿭 ?붾㈃?쇰줈 ?대룞?좉쾶?? ?뺤씤 ??寃곗젣瑜?吏꾪뻾??二쇱꽭??",
                    "action": "CHECKOUT",
                    "actionData": {"stage": "ORDER_REVIEW"},
                    "intent": "ORDER_FLOW",
                    "stage": "ORDER_REVIEW",
                },
                emotion="confident",
                expression="smile",
                motion="confirm",
            )

        if self._is_check_cart_intent(text):
            return self._decorate_parallel_channels(
                {
                    "speech": "?λ컮援щ땲瑜??뺤씤???쒕┫寃뚯슂.",
                    "action": "CHECK_CART",
                    "actionData": {"stage": "CART"},
                    "intent": "ORDER_FLOW",
                    "stage": "CART",
                },
                emotion="neutral",
                expression="attentive",
                motion="listen",
            )

        if self._is_continue_order_intent(text):
            return self._decorate_parallel_channels(
                {
                    "speech": "醫뗭븘?? 怨꾩냽 二쇰Ц???꾩??쒕┫寃뚯슂.",
                    "action": "CONTINUE_ORDER",
                    "actionData": {"stage": "MAIN_MENU"},
                    "intent": "ORDER_FLOW",
                    "stage": "MAIN_MENU",
                },
                emotion="happy",
                expression="smile",
                motion="nod",
            )

        if self._is_staff_call_intent(text):
            return self._decorate_parallel_channels(
                {
                    "speech": "吏곸썝???몄텧?좉쾶?? ?좎떆留?湲곕떎??二쇱꽭??",
                    "action": "CALL_STAFF",
                    "actionData": {"stage": stage},
                    "intent": "ORDER_FLOW",
                    "stage": stage,
                },
                emotion="supportive",
                expression="serious",
                motion="notify",
            )

        rec = self._build_vector_recommendation_response(text, state)
        if rec:
            return rec

        if menu_mention is not None:
            menu_name = str(menu_mention.get("name") or "선택한 메뉴")
            menu_item_id = str(menu_mention.get("menuItemId") or "")
            if menu_item_id:
                speech = f"{menu_name} {quantity}개를 선택했어요."
                action_data: Dict[str, Any] = {
                    "menuItemId": menu_item_id,
                    "quantity": quantity,
                    "stage": "CART",
                }
                if self._is_set_like(menu_mention):
                    speech = f"{menu_name} 선택했어요. 사이드 메뉴를 골라 주세요."
                    action_data["nextStage"] = "SIDE_SELECTION"
                return self._decorate_parallel_channels(
                    {
                        "speech": speech,
                        "action": "ADD_MENU",
                        "actionData": action_data,
                        "intent": "ORDER_FLOW",
                        "stage": "CART",
                    },
                    emotion="happy",
                    expression="smile",
                    motion="confirm",
                )

        return None

    @traceable_safe(name="v2.orchestrator.vector_recommend", run_type="tool")
    def _build_vector_recommendation_response(self, text: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._is_recommendation_query(text):
            return None
        allergen_terms = self._detect_query_allergen_terms(text, state)
        ingredient_vocab = self._build_ingredient_vocab_from_state(state)
        ingredient_term = self._extract_recommend_ingredient_term(
            text,
            ingredient_vocab=ingredient_vocab,
            exclude_terms=allergen_terms,
        )
        ingredient_term_norm = self._norm_text(ingredient_term)
        want_exclude = self._wants_exclusion(text)
        logger.info(
            "v2.recommend.parse text=%r exclude=%s allergens=%s ingredient=%s",
            text,
            want_exclude,
            allergen_terms,
            ingredient_term_norm,
        )
        has_constraints = bool(allergen_terms or ingredient_term_norm)

        def _filter_with_constraints(items: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
            if not has_constraints:
                return items
            before_count = len(items)
            filtered: List[Dict[str, Any]] = []
            debug_rows: List[str] = []
            for c in items:
                ok = True
                reasons: List[str] = []
                menu_name = str(c.get("name") or c.get("menuItemId") or "unknown")

                if allergen_terms:
                    known_allergens = set(self._candidate_known_allergens(c))
                    if not known_allergens:
                        ok = False
                        reasons.append("no_allergen_data")
                    elif want_exclude:
                        pass_allergen = all(term not in known_allergens for term in allergen_terms)
                        ok = ok and pass_allergen
                        if not pass_allergen:
                            reasons.append(f"allergen_exclude_hit:{sorted(list(known_allergens))}")
                    else:
                        pass_allergen = any(term in known_allergens for term in allergen_terms)
                        ok = ok and pass_allergen
                        if not pass_allergen:
                            reasons.append(f"allergen_include_miss:{sorted(list(known_allergens))}")

                if ok and ingredient_term_norm:
                    known_ingredients = self._candidate_known_ingredients(c)
                    if not known_ingredients:
                        ok = False
                        reasons.append("no_ingredient_data")
                    else:
                        has_ingredient = any(
                            ingredient_term_norm in ing or ing in ingredient_term_norm
                            for ing in known_ingredients
                        )
                        pass_ingredient = (not has_ingredient) if want_exclude else has_ingredient
                        ok = ok and pass_ingredient
                        if not pass_ingredient:
                            sample = ",".join(known_ingredients[:6])
                            reasons.append(f"ingredient_miss:{sample}")

                if ok:
                    filtered.append(c)
                elif len(debug_rows) < 20:
                    debug_rows.append(f"{menu_name}|{'/'.join(reasons) if reasons else 'unknown'}")

            logger.info(
                "v2.recommend.filter source=%s before=%s after=%s exclude=%s allergens=%s ingredient=%s",
                source,
                before_count,
                len(filtered),
                want_exclude,
                allergen_terms,
                ingredient_term_norm,
            )
            if debug_rows:
                logger.info("v2.recommend.filter.debug source=%s drops=%s", source, debug_rows)
            return filtered

        # Safety policy:
        # For allergen-constrained recommendations, do not trust vector-only metadata.
        # Use state catalog only (or return no recommendation) to avoid stale vector suggestions.
        candidates: List[Dict[str, Any]] = []
        # For exclusion queries, avoid vector-only metadata (stale risk).
        # For inclusion/general recommendation, vector candidates are still useful.
        should_use_vector = not (want_exclude and has_constraints)
        if should_use_vector:
            vc = state.get("vectorCandidates") if isinstance(state.get("vectorCandidates"), list) else []
            candidates = [c for c in vc if isinstance(c, dict)]
        logger.info(
            "v2.recommend.candidates source=%s count=%s",
            "vector" if candidates else "catalog-fallback",
            len(candidates),
        )

        candidates = _filter_with_constraints(candidates, "vector")

        # Fallback: when vector candidates are missing, use menu catalog candidates
        # so recommendation can still answer in one turn.
        if not candidates:
            catalog = self._extract_menu_catalog_from_state(state)
            candidates = [c for c in catalog if isinstance(c, dict)]
            logger.info("v2.recommend.catalog.load count=%s", len(candidates))
            candidates = _filter_with_constraints(candidates, "catalog")

        if not candidates:
            logger.info(
                "v2.recommend.result empty exclude=%s allergens=%s ingredient=%s",
                want_exclude,
                allergen_terms,
                ingredient_term_norm,
            )
            if has_constraints:
                mode = "제외" if want_exclude else "포함"
                cond_parts: List[str] = []
                if allergen_terms:
                    cond_parts.append(f"{', '.join(allergen_terms)} {mode}")
                if ingredient_term_norm:
                    cond_parts.append(f"{self._format_term_for_tts(ingredient_term)} {mode}")
                cond_text = " 및 ".join(cond_parts) if cond_parts else "요청한"
                return self._decorate_parallel_channels(
                    {
                        "speech": (
                            f"{cond_text} 조건에 맞는 검증된 메뉴를 찾지 못했어요. 다른 조건으로 말씀해 주세요."
                        ),
                        "action": "NONE",
                        "actionData": {
                            "stage": "RECOMMENDATION",
                            "condition": {
                                "allergenExclude": allergen_terms if want_exclude else [],
                                "allergenInclude": allergen_terms if not want_exclude else [],
                                "ingredientExclude": [ingredient_term] if (want_exclude and ingredient_term_norm) else [],
                                "ingredientInclude": [ingredient_term] if ((not want_exclude) and ingredient_term_norm) else [],
                            },
                            "recommendationCandidates": [],
                        },
                        "intent": "MENU_RECOMMEND",
                        "stage": "RECOMMENDATION",
                    },
                    emotion="neutral",
                    expression="attentive",
                    motion="listen",
                )
            return None
        # Deduplicate by display name first to avoid repeated same-name recommendations.
        unique: List[Dict[str, Any]] = []
        seen_name: set[str] = set()
        seen_id: set[str] = set()
        for c in candidates:
            menu_id = str(c.get("menuItemId") or "").strip().lower()
            menu_name = self._norm_text(str(c.get("name") or ""))
            if menu_name and menu_name in seen_name:
                continue
            if menu_id and menu_id in seen_id:
                continue
            if menu_name:
                seen_name.add(menu_name)
            if menu_id:
                seen_id.add(menu_id)
            unique.append(c)
        set_candidates = [c for c in unique if self._is_set_like(c)]
        non_set_candidates = [c for c in unique if not self._is_set_like(c)]
        random.shuffle(set_candidates)
        random.shuffle(non_set_candidates)
        prioritized = set_candidates + non_set_candidates
        top = prioritized[:3]
        names = [str(c.get("name") or "").strip() for c in top]
        names = [n for n in names if n]
        if not names:
            return None
        logger.info(
            "v2.recommend.result names=%s exclude=%s allergens=%s ingredient=%s setFirst=true setCount=%s nonSetCount=%s",
            names,
            want_exclude,
            allergen_terms,
            ingredient_term_norm,
            len(set_candidates),
            len(non_set_candidates),
        )
        mode = "제외" if want_exclude else "포함"
        cond_parts: List[str] = []
        if allergen_terms:
            cond_parts.append(f"{', '.join(allergen_terms)} {mode}")
        if ingredient_term_norm:
            cond_parts.append(f"{self._format_term_for_tts(ingredient_term)} {mode}")
        filter_prefix = f"{' 및 '.join(cond_parts)} 조건을 반영한 " if cond_parts else ""
        focus_note = "세트 메뉴 위주로, " if set_candidates else ""
        return self._decorate_parallel_channels(
            {
                "speech": f"{focus_note}{filter_prefix}메뉴는 {', '.join(names)}가 있어요. 원하시는 메뉴 이름을 말씀해 주세요.",
                "action": "NONE",
                "actionData": {
                    "stage": "RECOMMENDATION",
                    "condition": {
                        "allergenExclude": allergen_terms if want_exclude else [],
                        "allergenInclude": allergen_terms if not want_exclude else [],
                        "ingredientExclude": [ingredient_term] if (want_exclude and ingredient_term_norm) else [],
                        "ingredientInclude": [ingredient_term] if ((not want_exclude) and ingredient_term_norm) else [],
                    },
                    "priority": {"setFirst": True},
                    "recommendationCandidates": [
                        {
                            "menuItemId": str(c.get("menuItemId") or ""),
                            "name": str(c.get("name") or ""),
                            "score": float(c.get("score") or 0.0),
                        }
                        for c in top
                    ],
                },
                "intent": "MENU_RECOMMEND",
                "stage": "RECOMMENDATION",
            },
            emotion="supportive",
            expression="smile",
            motion="offer_help",
        )

    def _infer_stage(self, state: Dict[str, Any]) -> str:
        if not isinstance(state, dict):
            return "SESSION_INIT"
        explicit_stage = str(state.get("stage") or "").strip().upper()
        if explicit_stage in {
            "SESSION_INIT",
            "ASK_DINING_TYPE",
            "MAIN_MENU",
            "SIDE_SELECTION",
            "DRINK_SELECTION",
            "CART",
            "ORDER_REVIEW",
            "PAYMENT",
            "PROACTIVE_HELP",
            "RECOMMENDATION",
        }:
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

    def _is_recommendation_query(self, text: str) -> bool:
        t = re.sub(r"\s+", "", (text or "").lower())
        tokens = [
            "추천",
            "추천해줘",
            "추천해줘요",
            "추천좀",
            "추천해",
            "비슷",
            "비슷한",
            "유사",
            "어울리는",
            "뭐먹",
            "뭐먹지",
            "뭐먹을까",
            "뭐가좋",
            "골라줘",
            "골라주",
            "어떤메뉴",
            "모르겠",
            "고민",
            "알레르기",
            "알레르겐",
            "알러지",
            "안들어가",
            "안들어간",
            "안들어가는",
            "없는메뉴",
            "제외",
            "빼고",
        ]
        return any(tok in t for tok in tokens)

    def _is_hesitation_signal(self, text: str, state: Dict[str, Any]) -> bool:
        if bool(state.get("isHesitating")):
            return True
        score = state.get("hesitationScore")
        try:
            if score is not None and float(score) >= 0.6:
                return True
        except Exception:
            pass
        t = (text or "").replace(" ", "")
        return any(tok in t for tok in ["모르겠", "어려", "헷갈", "고민"])

    def _is_checkout_intent(self, text: str) -> bool:
        t = (text or "").replace(" ", "")
        return any(tok in t for tok in ["결제", "주문끝", "주문완료", "그만", "진행"])

    def _is_check_cart_intent(self, text: str) -> bool:
        t = (text or "").replace(" ", "")
        return any(tok in t for tok in ["장바구니", "주문내역", "뭐담", "현재메뉴"])

    def _is_continue_order_intent(self, text: str) -> bool:
        t = (text or "").replace(" ", "")
        return any(tok in t for tok in ["계속", "추가주문", "추가주문", "다른메뉴"])

    def _is_staff_call_intent(self, text: str) -> bool:
        t = (text or "").replace(" ", "")
        return any(tok in t for tok in ["직원", "도와", "불러", "문의"])

    def _extract_payment_method(self, text: str) -> Optional[str]:
        t = (text or "").replace(" ", "").lower()
        if any(tok in t for tok in ["카드", "card"]):
            return "CARD"
        if any(tok in t for tok in ["포인트", "point"]):
            return "POINT"
        if any(tok in t for tok in ["간편", "simple", "삼성페이", "애플페이", "네이버페이", "카카오페이"]):
            return "SIMPLE"
        return None

    def _extract_quantity(self, text: str) -> int:
        m = re.search(r"(\d+)\s*(개|개요|개만|개씩|명)?", text or "")
        if not m:
            return 1
        try:
            return max(1, int(m.group(1)))
        except Exception:
            return 1

    def _is_set_like(self, item: Dict[str, Any]) -> bool:
        name = str(item.get("name") or "")
        category = str(item.get("categoryId") or item.get("category") or "").lower()
        return ("세트" in name) or ("set" in category)

    def _decorate_parallel_channels(
        self,
        payload: Dict[str, Any],
        emotion: str,
        expression: str,
        motion: str,
    ) -> Dict[str, Any]:
        out = dict(payload)
        speech = str(out.get("speech") or out.get("reply") or "").strip()
        out["live2d"] = {
            "emotion": emotion,
            "expression": expression,
            "motion": motion,
            "ts": datetime.utcnow().isoformat() + "Z",
        }
        out["parallel"] = {
            "runInParallel": True,
            "tts": {
                "mode": "stream",
                "text": speech,
            },
            "emotion": {
                "mode": "event",
                "emotion": emotion,
                "expression": expression,
                "motion": motion,
            },
        }
        return out

    @traceable_safe(name="v2.orchestrator.info_tools", run_type="tool")
    async def _run_info_tools(
        self,
        user_text: str,
        menu_items: List[Dict[str, Any]],
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        text = (user_text or "").strip()
        if not text:
            return None

        if self._is_menu_list_question(text):
            counts: Dict[str, int] = {}
            for it in menu_items:
                label = self._category_label(it.get("categoryId") or it.get("category"))
                counts[label] = counts.get(label, 0) + 1
            parts = [f"{k} {v}개" for k, v in counts.items() if v > 0]
            parts_str = ", ".join(parts) if parts else "메뉴 데이터가 아직 준비되지 않았어요."
            return {
                "speech": f"{parts_str}가 있어요. 어떤 카테고리를 보여드릴까요?",
                "action": "NONE",
                "actionData": {},
                "intent": "MENU_INFO",
                "sessionId": session_id,
            }

        allergen_terms = self._detect_allergen_terms(text)
        mentioned = self._resolve_menu_mention(text, menu_items)
        info_query = self._is_menu_info_query(text)

        # "알레르기 없는 메뉴 추천" 유형은 정보질의가 아니라 추천 분기로 보낸다.
        if mentioned is None and info_query and self._is_recommendation_query(text):
            return None

        if mentioned and info_query:
            detail = None
            if self.menu_detail_provider:
                try:
                    detail = await self.menu_detail_provider(str(mentioned.get("menuItemId") or ""))
                except Exception:
                    detail = None

            if not detail:
                return {
                    "speech": "해당 메뉴 정보를 DB에서 찾지 못했어요. 다른 메뉴로 다시 말씀해 주세요.",
                    "action": "NONE",
                    "actionData": {},
                    "intent": "MENU_INFO",
                }

            detail = self._normalize_menu_detail(detail, fallback_name=str(mentioned.get("name") or "해당 메뉴"))

            if is_calorie_query(text):
                kcal = ((detail.get("nutrition") or {}).get("kcal"))
                if kcal is None:
                    speech = f"{detail.get('name', '해당 메뉴')} 칼로리 정보가 DB에 없어요."
                else:
                    speech = f"{detail.get('name', '해당 메뉴')} 칼로리는 {kcal}kcal입니다."
            elif is_allergy_query(text) or allergen_terms:
                alls = detail.get("allergies") or []
                speech = (
                    f"{detail.get('name', '해당 메뉴')} 알레르기 정보는 {', '.join(alls)}입니다."
                    if alls
                    else f"{detail.get('name', '해당 메뉴')} 알레르기 정보가 DB에 없어요."
                )
            else:
                ings = detail.get("ingredients") or []
                speech = (
                    f"{detail.get('name', '해당 메뉴')} 재료는 {', '.join(ings)}입니다."
                    if ings
                    else f"{detail.get('name', '해당 메뉴')} 재료 정보가 아직 준비 중이에요."
                )

            return {
                "speech": speech,
                "action": "NONE",
                "actionData": {},
                "intent": "MENU_INFO",
            }

        if mentioned is None and info_query:
            kinds: List[str] = []
            if is_allergy_query(text):
                kinds.append("알레르기")
            if is_ingredient_query(text):
                kinds.append("재료")
            if is_calorie_query(text):
                kinds.append("칼로리")
            target = ", ".join(kinds) if kinds else "메뉴 정보"
            speech = f"어떤 메뉴의 {target}를 확인할까요? 메뉴 이름과 함께 말씀해 주세요."
            return {
                "speech": speech,
                "action": "NONE",
                "actionData": {},
                "intent": "MENU_INFO",
            }

        return None

    def _is_ambiguous_query(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        t_no_space = re.sub(r"\s+", "", t)
        if len(t_no_space) <= 3:
            return True
        ambiguity_tokens = [
            "뭐",
            "추천",
            "비슷",
            "어울리는",
            "그거",
            "뭘까",
            "아무거나",
        ]
        return any(tok in t_no_space for tok in ambiguity_tokens)

    def _is_menu_list_question(self, text: str) -> bool:
        t = text.replace(" ", "")
        return any(k in t for k in ["뭐뭐있", "메뉴뭐", "메뉴있", "메뉴목록", "전체메뉴", "뭐가있", "뭐있어"])

    def _is_ingredient_question(self, text: str) -> bool:
        return is_ingredient_query(text) or is_allergy_query(text) or is_calorie_query(text)

    def _is_menu_info_query(self, text: str) -> bool:
        return is_menu_info_query(text)

    def _detect_allergen_terms(self, text: str) -> List[str]:
        return detect_allergen_terms(text)

    def _wants_exclusion(self, text: str) -> bool:
        if wants_exclusion(text):
            return True
        t = self._norm_text(text)
        return any(
            tok in t
            for tok in [
                "없는",
                "안들어가",
                "안들어간",
                "안들어가는",
                "들어가지않은",
                "들어가지않는",
                "빼고",
                "제외",
                "미포함",
                "불포함",
                "notinclude",
                "without",
            ]
        )

    def _allergen_vocab_from_state(self, state: Dict[str, Any]) -> List[str]:
        items: List[Dict[str, Any]] = []
        if isinstance(state, dict):
            catalog = state.get("menuCatalog")
            vectors = state.get("vectorCandidates")
            if isinstance(catalog, list):
                items.extend([it for it in catalog if isinstance(it, dict)])
            if isinstance(vectors, list):
                items.extend([it for it in vectors if isinstance(it, dict)])
        out: List[str] = []
        seen: set[str] = set()
        for it in items:
            for a in self._candidate_known_allergens(it):
                key = self._norm_text(a)
                if key and key not in seen:
                    seen.add(key)
                    out.append(a)
        return out

    def _format_term_for_tts(self, raw: str) -> str:
        t = str(raw or "").strip()
        if not t:
            return t
        # Improve Korean TTS readability for compact ingredient tokens.
        replacements = {
            "닭가슴살패티": "닭가슴살 패티",
            "해쉬브라운": "해쉬 브라운",
        }
        for src, dst in replacements.items():
            t = t.replace(src, dst)
        return re.sub(r"\s+", " ", t).strip()

    def _detect_query_allergen_terms(self, text: str, state: Dict[str, Any]) -> List[str]:
        norm_text = self._norm_text(text)
        if not norm_text:
            return []
        legacy = self._detect_allergen_terms(text)
        vocab = self._allergen_vocab_from_state(state)
        if vocab:
            found: List[str] = []
            for a in vocab:
                key = self._norm_text(a)
                if key and key in norm_text and a not in found:
                    found.append(a)
            for a in legacy:
                if a not in found:
                    found.append(a)
            return found
        return legacy

    def _build_ingredient_vocab_from_state(self, state: Dict[str, Any]) -> List[str]:
        items: List[Dict[str, Any]] = []
        if isinstance(state, dict):
            catalog = state.get("menuCatalog")
            vectors = state.get("vectorCandidates")
            if isinstance(catalog, list):
                items.extend([it for it in catalog if isinstance(it, dict)])
            if isinstance(vectors, list):
                items.extend([it for it in vectors if isinstance(it, dict)])
        out: List[str] = []
        seen: set[str] = set()
        for it in items:
            for ing in self._candidate_known_ingredients(it):
                if ing and ing not in seen:
                    seen.add(ing)
                    out.append(ing)
        return out

    def _has_catalog_ingredient_data(self, state: Dict[str, Any]) -> bool:
        if not isinstance(state, dict):
            return False
        catalog = state.get("menuCatalog")
        if not isinstance(catalog, list):
            return False
        checked = 0
        for it in catalog:
            if not isinstance(it, dict):
                continue
            checked += 1
            if self._candidate_known_ingredients(it):
                return True
            if checked >= 40:
                break
        return False

    def _extract_recommend_ingredient_term(
        self,
        text: str,
        ingredient_vocab: Optional[List[str]] = None,
        exclude_terms: Optional[List[str]] = None,
    ) -> str:
        raw = (text or "").strip().lower()
        if not raw:
            return ""
        # Common STT/spacing variants.
        raw = raw.replace("해시브라운", "해쉬브라운").replace("세시브라운", "해쉬브라운")
        norm_text = self._norm_text(raw)
        exclude_norm = {self._norm_text(x) for x in (exclude_terms or []) if self._norm_text(x)}

        if ingredient_vocab:
            vocab_sorted = sorted(
                [v for v in ingredient_vocab if v],
                key=lambda x: len(self._norm_text(x)),
                reverse=True,
            )
            for ing in vocab_sorted:
                key = self._norm_text(ing)
                if not key or key in exclude_norm:
                    continue
                if key in norm_text:
                    return ing

        tokens = re.findall(r"[0-9a-zA-Z가-힣]+", raw)
        def _strip_particle(token: str) -> str:
            t = token
            for suf in ["으로", "로", "은", "는", "이", "가", "을", "를", "와", "과", "랑", "도"]:
                if len(t) > len(suf) + 1 and t.endswith(suf):
                    return t[: -len(suf)]
            return t
        stop = {
            "추천",
            "메뉴",
            "음식",
            "재료",
            "원재료",
            "성분",
            "알레르기",
            "알레르겐",
            "포함",
            "제외",
            "없는",
            "안들어가",
            "안들어간",
            "안들어가는",
            "들어간",
            "들어가는",
            "들어가",
            "들어가지않은",
            "들어가지않는",
            "않은",
            "않는",
            "않아",
            "해줘",
            "좀",
            "으로",
        }
        stop_norm = {self._norm_text(x) for x in stop}
        noisy_substrings = [
            "추천",
            "메뉴",
            "음식",
            "재료",
            "원재료",
            "성분",
            "알레르기",
            "알레르겐",
            "포함",
            "제외",
            "안들어가",
            "들어가",
            "들어가지않",
            "않는",
            "않은",
            "없는",
            "말해",
            "알려",
            "해주세요",
            "해줘",
        ]
        candidates: List[str] = []
        for token in tokens:
            t = _strip_particle(str(token or "").strip())
            if not t:
                continue
            tn = self._norm_text(t)
            if not tn or tn in stop_norm:
                continue
            if tn in exclude_norm:
                continue
            if len(tn) <= 1 or len(tn) > 12:
                continue
            if any(noise in tn for noise in noisy_substrings):
                continue
            candidates.append(t)
        if not candidates:
            return ""
        # Prefer longest content token.
        return sorted(candidates, key=len, reverse=True)[0]

    def _resolve_menu_mention(self, text: str, menu_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not text or not menu_items:
            return None
        t_norm = self._norm_text(self._canonical_name(text))
        best = None
        best_score = 0
        for it in menu_items:
            name = it.get("name") or ""
            if not name:
                continue
            n_norm = self._norm_text(self._canonical_name(name))
            if not n_norm:
                continue
            if n_norm in t_norm:
                score = len(n_norm)
            elif t_norm in n_norm:
                score = max(1, len(t_norm) // 2)
            else:
                continue
            if score > best_score:
                best = it
                best_score = score
        return best

    def _canonical_name(self, s: str) -> str:
        s = (s or "").strip()
        for tok in ["세트", "단품", "세트메뉴", "세트 메뉴", "(m)", "(r)", "(l)"]:
            s = s.replace(tok, "")
        return re.sub(r"\s+", "", s)

    def _norm_text(self, s: str) -> str:
        s = (s or "").strip().lower()
        return re.sub(r"\s+", "", s)

    def _category_label(self, category_id: Optional[str]) -> str:
        mapping = {
            "cat_set": "세트 메뉴",
            "cat_burger": "단품",
            "cat_side": "사이드",
            "cat_drink": "음료",
            "cat_chicken": "치킨",
            "cat_best": "베스트메뉴",
        }
        return mapping.get(category_id or "", "기타")

    def _normalize_menu_detail(self, detail: Dict[str, Any], fallback_name: str) -> Dict[str, Any]:
        out = detail if isinstance(detail, dict) else {}
        name = str(out.get("name") or fallback_name or "해당 메뉴").strip()

        raw_allergies = (
            out.get("allergies")
            or out.get("allergens")
            or out.get("allergyTags")
            or out.get("allergy")
            or []
        )
        if isinstance(raw_allergies, str):
            raw_allergies = [x.strip() for x in re.split(r"[,/|]", raw_allergies) if x.strip()]
        if not isinstance(raw_allergies, list):
            raw_allergies = []
        allergy_tokens: List[str] = []
        for x in raw_allergies:
            if isinstance(x, dict):
                v = (
                    x.get("name")
                    or x.get("allergen")
                    or x.get("allergy")
                    or x.get("allergyTag")
                    or x.get("label")
                    or ""
                )
            else:
                v = x
            s = str(v or "").strip()
            if s:
                allergy_tokens.append(s)
        allergies = self._normalize_allergy_tokens(allergy_tokens)

        raw_ingredients = (
            out.get("ingredients")
            or out.get("ingredientNames")
            or out.get("ingredient")
            or out.get("materials")
            or out.get("rawMaterials")
            or []
        )
        if isinstance(raw_ingredients, str):
            raw_ingredients = [x.strip() for x in re.split(r"[,/|]", raw_ingredients) if x.strip()]
        if not isinstance(raw_ingredients, list):
            raw_ingredients = []
        ingredients: List[str] = []
        for x in raw_ingredients:
            if isinstance(x, dict):
                v = (
                    x.get("name")
                    or x.get("ingredientName")
                    or x.get("ingredient")
                    or x.get("label")
                    or ""
                )
            else:
                v = x
            s = str(v or "").strip()
            if s:
                ingredients.append(s)

        nutrition = out.get("nutrition") if isinstance(out.get("nutrition"), dict) else {}
        kcal = nutrition.get("kcal")
        if kcal is None:
            kcal = out.get("kcal")
        if kcal is None:
            kcal = out.get("calories")
        if kcal is None:
            kcal = out.get("energyKcal")
        if isinstance(kcal, str):
            m = re.search(r"(\d+(?:\.\d+)?)", kcal)
            kcal = float(m.group(1)) if m else None

        normalized = {
            "name": name,
            "allergies": allergies,
            "ingredients": ingredients,
            "nutrition": {"kcal": kcal} if kcal is not None else {},
        }
        return normalized

    def _normalize_allergy_tokens(self, values: List[Any]) -> List[str]:
        mapping = {
            "egg": "난류",
            "eggs": "난류",
            "milk": "우유",
            "soy": "대두",
            "soybean": "대두",
            "wheat": "밀",
            "tomato": "토마토",
            "chicken": "닭고기",
            "beef": "쇠고기",
            "pork": "돼지고기",
            "shrimp": "새우",
            "oyster": "굴",
            "알러지": "알레르기",
        }
        out: List[str] = []
        for raw in values:
            token = str(raw or "").strip()
            if not token:
                continue
            lower = token.lower()
            mapped = mapping.get(lower, token)
            canon = detect_allergen_terms(mapped)
            if canon:
                for c in canon:
                    if c not in out:
                        out.append(c)
                continue
            if mapped not in out:
                out.append(mapped)
        return out

    def _candidate_known_allergens(self, candidate: Dict[str, Any]) -> List[str]:
        raw_values = (
            candidate.get("allergies")
            or candidate.get("allergens")
            or candidate.get("allergyTags")
            or candidate.get("allergy")
            or []
        )
        if isinstance(raw_values, str):
            raw_values = [x.strip() for x in re.split(r"[,/|]", raw_values) if x.strip()]
        if not isinstance(raw_values, list):
            raw_values = []

        normalized = self._normalize_allergy_tokens(raw_values)
        unknown_markers = {
            "정보준비중",
            "unknown",
            "none",
            "null",
            "na",
            "n/a",
            "미정",
            "확인중",
            "준비중",
            "-",
            "?",
        }
        out: List[str] = []
        for tok in normalized:
            s = str(tok or "").strip()
            if not s:
                continue
            s_norm = self._norm_text(s)
            if not s_norm or s_norm in unknown_markers:
                continue
            canon = self._detect_allergen_terms(s)
            if not canon:
                # Only trust known allergen vocabulary for exclusion filtering.
                continue
            for c in canon:
                if c not in out:
                    out.append(c)
        return out

    def _candidate_known_ingredients(self, candidate: Dict[str, Any]) -> List[str]:
        raw_values = candidate.get("ingredients") or candidate.get("ingredient") or []
        if isinstance(raw_values, str):
            raw_values = [x.strip() for x in re.split(r"[,/|]", raw_values) if x.strip()]
        if not isinstance(raw_values, list):
            raw_values = []
        unknown_markers = {
            "정보준비중",
            "unknown",
            "none",
            "null",
            "na",
            "n/a",
            "미정",
            "확인중",
            "준비중",
            "-",
            "?",
        }
        out: List[str] = []
        for tok in raw_values:
            if isinstance(tok, dict):
                v = (
                    tok.get("name")
                    or tok.get("ingredientName")
                    or tok.get("ingredient")
                    or tok.get("label")
                    or ""
                )
            else:
                v = tok
            s = str(v or "").strip()
            if not s:
                continue
            n = self._norm_text(s)
            if not n:
                continue
            if n in unknown_markers:
                continue
            if n not in out:
                out.append(n)
        return out

    def _normalize_output(self, raw: Dict[str, Any], orchestrator_name: str) -> Dict[str, Any]:
        payload = dict(raw) if isinstance(raw, dict) else {}
        speech = str(payload.get("speech") or payload.get("reply") or "").strip() or "원하시는 메뉴를 다시 말씀해 주세요."
        action = str(payload.get("action") or "NONE").upper()
        if action not in AllowedAction:
            action = "NONE"
        if action == "ADD_TO_CART":
            action = "ADD_MENU"
        elif action == "REMOVE_FROM_CART":
            action = "REMOVE_MENU"
        intent = str(payload.get("intent") or "GENERAL")
        action_data = payload.get("actionData") if isinstance(payload.get("actionData"), dict) else {}

        if action == "NAVIGATE":
            if str(action_data.get("categoryKey") or "").strip() or str(action_data.get("selectedCategory") or "").strip():
                action = "NAVIGATE_CATEGORY"
            else:
                action = "CONTINUE_ORDER"

        if action == "NAVIGATE_CATEGORY" and not str(action_data.get("categoryKey") or "").strip():
            selected = str(action_data.get("selectedCategory") or "").lower().strip()
            if selected in {"cat_set", "set"}:
                action_data["categoryKey"] = "set"
            elif selected in {"cat_burger", "single", "burger"}:
                action_data["categoryKey"] = "single"
            elif selected in {"cat_side", "side"}:
                action_data["categoryKey"] = "side"
            elif selected in {"cat_drink", "drink"}:
                action_data["categoryKey"] = "drink"
            elif selected in {"cat_chicken", "chicken"}:
                action_data["categoryKey"] = "chicken"

        if action == "SELECT_PAYMENT" and not str(action_data.get("method") or "").strip():
            t = re.sub(r"\s+", "", speech.lower())
            if "카드" in speech or "card" in t:
                action_data["method"] = "CARD"
            elif "포인트" in speech or "point" in t:
                action_data["method"] = "POINT"
            elif any(x in speech for x in ["간편", "삼성페이", "애플페이", "카카오페이"]):
                action_data["method"] = "SIMPLE"
        if action == "SELECT_PAYMENT" and not str(action_data.get("method") or "").strip():
            action = "NONE"
        if action == "ADD_MENU" and not str(action_data.get("menuItemId") or "").strip():
            action = "NONE"
        if action == "NAVIGATE_CATEGORY" and not str(action_data.get("categoryKey") or "").strip():
            action = "CONTINUE_ORDER"

        stage = str(payload.get("stage") or action_data.get("stage") or "").strip().upper()
        if not stage:
            if action == "CHECK_CART":
                stage = "CART"
            elif action == "CHECKOUT":
                stage = "ORDER_REVIEW"
            elif action == "SELECT_PAYMENT":
                stage = "PAYMENT"
            elif action in {"ADD_MENU", "REMOVE_MENU", "CHANGE_QTY"}:
                stage = "CART"
            elif action == "NAVIGATE_CATEGORY":
                ck = str(action_data.get("categoryKey") or "").strip().lower()
                if ck == "side":
                    stage = "SIDE_SELECTION"
                elif ck == "drink":
                    stage = "DRINK_SELECTION"
                else:
                    stage = "MAIN_MENU"
            else:
                stage = "MAIN_MENU"
        action_data.setdefault("stage", stage)
        payload["actionData"] = action_data
        payload["stage"] = stage

        live2d_obj = payload.get("live2d") if isinstance(payload.get("live2d"), dict) else {}
        parallel_obj = payload.get("parallel") if isinstance(payload.get("parallel"), dict) else {}
        current_motion = str((live2d_obj or {}).get("motion") or "")
        has_valid_motion = bool(current_motion) and current_motion in ALL_MOTIONS
        has_parallel = isinstance(parallel_obj, dict) and isinstance(parallel_obj.get("emotion"), dict) and isinstance(parallel_obj.get("tts"), dict)

        if (not has_valid_motion) or (not has_parallel):
            emotion, expression, motion = pick_live2d_profile(intent=intent, action=action, stage=stage, speech=speech)
            payload = self._decorate_parallel_channels(payload, emotion=emotion, expression=expression, motion=motion)

        action_data = payload.get("actionData") if isinstance(payload.get("actionData"), dict) else {}
        out = {
            "reply": speech,
            "text": speech,
            "intent": intent,
            "action": action,
            "actionData": action_data,
            "orchestrator": orchestrator_name,
            "generatedAt": datetime.utcnow().isoformat() + "Z",
        }
        live2d = payload.get("live2d")
        if isinstance(live2d, dict):
            out["live2d"] = live2d
        parallel = payload.get("parallel")
        if isinstance(parallel, dict):
            out["parallel"] = parallel
        if stage:
            out["stage"] = stage
        return out


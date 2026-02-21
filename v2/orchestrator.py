from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

AllowedAction = {
    "NONE",
    "NAVIGATE",
    "ADD_MENU",
    "REMOVE_MENU",
    "CHANGE_QTY",
    "CHECK_CART",
    "CHECKOUT",
    "SELECT_PAYMENT",
    "CONTINUE_ORDER",
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

    def set_vector_search_provider(
        self,
        provider: Optional[Callable[[str, int], Awaitable[Dict[str, Any]]]],
    ) -> None:
        self.vector_search_provider = provider

    async def run(
        self,
        request: Any,
        openai_client: Any,
        request_id: str,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        trace: List[Dict[str, Any]] = []

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
                    {"speech": "사용자 발화가 비어 있어요.", "action": "NONE", "actionData": {}},
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
        if self._is_ambiguous_query(latest_user) and self.vector_search_provider is not None:
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

        if self._can_use_langgraph():
            try:
                out = await self._run_langgraph(messages, latest_user, working_state, menu_items, order_type)
                mark("langgraph_success")
                return {
                    "result": self._normalize_output(out, "langgraph"),
                    "trace": trace,
                }
            except Exception:
                logger.exception("v2 langgraph path failed; falling back to direct OpenAI")
                mark("langgraph_error")

        out = await self._run_openai_fallback(openai_client, messages, latest_user, working_state, menu_items, order_type)
        mark("openai_fallback_success", {"requestId": request_id})
        return {
            "result": self._normalize_output(out, "openai-fallback"),
            "trace": trace,
        }

    def _can_use_langgraph(self) -> bool:
        try:
            import langgraph  # noqa: F401
            import langchain_openai  # noqa: F401
            import langchain_core  # noqa: F401
            return True
        except Exception:
            return False

    async def _run_langgraph(
        self,
        messages: List[Any],
        latest_user: str,
        state: Dict[str, Any],
        menu_items: List[Dict[str, Any]],
        order_type: str,
    ) -> Dict[str, Any]:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        from langgraph.graph import END, StateGraph

        model_name = os.getenv(self.model_env_var, os.getenv(self.fallback_model_env_var, "gpt-4o-mini"))
        llm = ChatOpenAI(model=model_name, temperature=0.1)
        structured_llm = llm.with_structured_output(StructuredAction)
        system_prompt = self._build_system_prompt(order_type, state, menu_items)
        history_text = self._history_for_model(messages, latest_user)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{history_text}"),
            ]
        )
        chain = prompt | structured_llm

        async def route_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
            # placeholder for future branch expansion
            _ = graph_state.get("latest_user", "")
            return {"route": "plan"}

        async def plan_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
            res: StructuredAction = await chain.ainvoke({"history_text": graph_state["history_text"]})
            return {"result": res.model_dump()}

        graph = StateGraph(dict)
        graph.add_node("route", route_node)
        graph.add_node("plan", plan_node)
        graph.set_entry_point("route")
        graph.add_conditional_edges("route", lambda s: s.get("route", "plan"), {"plan": "plan"})
        graph.add_edge("plan", END)
        app = graph.compile()

        result = await app.ainvoke({"history_text": history_text, "latest_user": latest_user})
        return result.get("result") or {}

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
        content = f"{history_text}\n\n반드시 JSON만 출력하세요."

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
            return {"speech": "잘 들었어요. 한 번 더 말씀해 주세요.", "action": "NONE", "actionData": {}}

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

        return (
            "너는 키오스크 음성 주문 오케스트레이터다.\n"
            "JSON 스키마만 출력해라: {\"speech\": string, \"action\": string, \"actionData\": object}\n"
            f"허용 action: {sorted(list(AllowedAction))}\n"
            "규칙: menuItemId는 카탈로그에 있는 값만 사용.\n"
            "vectorCandidates는 모호 질의 후보이며 최종 확정은 카탈로그와 문맥으로 하라.\n"
            f"현재 주문 타입: {order_type}\n"
            f"현재 상태: {json.dumps(state, ensure_ascii=False)}\n"
            f"메뉴 카탈로그: {json.dumps(catalog, ensure_ascii=False)}\n"
            f"Vector 후보: {json.dumps(vector_candidates, ensure_ascii=False)}\n"
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
            parts_str = ", ".join(parts) if parts else "메뉴 데이터가 아직 준비되지 않았어요"
            return {
                "speech": f"{parts_str}가 있어요. 어떤 카테고리를 보여드릴까요?",
                "action": "NONE",
                "actionData": {},
                "intent": "MENU_INFO",
                "sessionId": session_id,
            }

        allergen_terms = self._detect_allergen_terms(text)
        mentioned = self._resolve_menu_mention(text, menu_items)

        if mentioned and (
            self._is_ingredient_question(text)
            or allergen_terms
            or "칼로리" in text
            or "kcal" in text.lower()
        ):
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

            if "칼로리" in text or "kcal" in text.lower():
                kcal = ((detail.get("nutrition") or {}).get("kcal"))
                if kcal is None:
                    speech = f"{detail.get('name', '해당 메뉴')} 칼로리 정보가 DB에 없어요."
                else:
                    speech = f"{detail.get('name', '해당 메뉴')} 칼로리는 {kcal}kcal입니다."
            elif "알레르기" in text or allergen_terms:
                alls = detail.get("allergies") or []
                speech = (
                    f"{detail.get('name', '해당 메뉴')} 알레르기는 {', '.join(alls)}입니다."
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

        if allergen_terms and not mentioned:
            term = allergen_terms[0]
            want_exclude = any(k in text for k in ["없는", "빼고", "제외"])
            matched: List[str] = []

            for it in menu_items[:120]:
                name = str(it.get("name") or "").strip()
                if not name:
                    continue
                alls = it.get("allergies")
                if not isinstance(alls, list) and self.menu_detail_provider:
                    mid = str(it.get("menuItemId") or "")
                    if mid:
                        try:
                            d = await self.menu_detail_provider(mid)
                            alls = (d or {}).get("allergies") or []
                        except Exception:
                            alls = []
                if not isinstance(alls, list):
                    alls = []
                has = term in alls
                if (not want_exclude and has) or (want_exclude and not has):
                    matched.append(name)

            if not matched:
                speech = f"DB 기준으로 '{term}' 조건에 맞는 메뉴를 찾지 못했어요."
            else:
                show = matched[:8]
                suffix = f" 등 {len(matched)}개가 있어요." if len(matched) > len(show) else "가 있어요."
                speech = f"'{term}' {'없는' if want_exclude else '포함된'} 메뉴는 {', '.join(show)}{suffix}"
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
            "비슷한",
            "어울리는",
            "그거",
            "그걸로",
            "아까",
            "비슷한거",
            "아무거나",
        ]
        return any(tok in t_no_space for tok in ambiguity_tokens)

    def _is_menu_list_question(self, text: str) -> bool:
        t = text.replace(" ", "")
        return any(k in t for k in ["뭐뭐있", "메뉴뭐", "메뉴있", "메뉴목록", "전체메뉴", "뭐가있", "뭐있어"])

    def _is_ingredient_question(self, text: str) -> bool:
        t = text.replace(" ", "")
        return any(k in t for k in ["재료", "들어가", "들어간", "빼고", "없는", "제외", "포함"])

    def _detect_allergen_terms(self, text: str) -> List[str]:
        candidates = ["난류", "우유", "대두", "밀", "토마토", "닭고기", "쇠고기", "돼지고기", "새우", "굴"]
        synonyms = {
            "달걀": "난류",
            "계란": "난류",
            "유제품": "우유",
            "치즈": "우유",
            "콩": "대두",
            "간장": "대두",
            "글루텐": "밀",
            "빵": "밀",
            "소고기": "쇠고기",
            "돼지": "돼지고기",
            "치킨": "닭고기",
            "쉬림프": "새우",
            "조개": "굴",
        }
        found = [c for c in candidates if c in text]
        for k, v in synonyms.items():
            if k in text and v not in found:
                found.append(v)
        return found

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

    def _normalize_output(self, raw: Dict[str, Any], orchestrator_name: str) -> Dict[str, Any]:
        speech = str(raw.get("speech") or raw.get("reply") or "").strip() or "원하시는 메뉴를 다시 말씀해 주세요."
        action = str(raw.get("action") or "NONE").upper()
        if action not in AllowedAction:
            action = "NONE"
        action_data = raw.get("actionData")
        if not isinstance(action_data, dict):
            action_data = {}
        return {
            "reply": speech,
            "text": speech,
            "intent": str(raw.get("intent") or "GENERAL"),
            "action": action,
            "actionData": action_data,
            "orchestrator": orchestrator_name,
            "generatedAt": datetime.utcnow().isoformat() + "Z",
        }


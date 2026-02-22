from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import pytz
import os
import re
import time
import asyncio
import urllib.request
import urllib.parse
import base64
import tempfile
import json
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict
import logging
from v2.orchestrator import V2LangChainOrchestrator
from v2.vector_index import QdrantMenuVectorStore

# Load env from Backend folder as user specified
env_path = Path(__file__).parent.parent / "Backend" / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API") or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

app = FastAPI(
    title="AI Kiosk - AI Server",
    description="MediaPipe Pose? FaceMesh???λ윭??紐⑤뜽 援щ룞 ?쒕쾭 + OpenAI Integration",
    version="0.2.0"
)
logger = logging.getLogger("ai.v2")

# CORS ?ㅼ젙
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_timestamp():
    korea_timezone = pytz.timezone("Asia/Seoul")
    return datetime.now(korea_timezone).isoformat()

# --- Common Models ---

class CommonResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    timestamp: str = ""
    requestId: str = "req_default"

    def __init__(self, **data):
        super().__init__(**data)
        if not self.timestamp:
            self.timestamp = get_timestamp()

# --- DTO Models ---

class SttRequest(BaseModel):
    audioBase64: str
    language: str = "ko"
    mimeType: Optional[str] = "audio/wav"
    model: Optional[str] = "whisper-1"

class TtsRequest(BaseModel):
    text: str
    voice: str = "nova"
    speed: float = 1.0

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatContext(BaseModel):
    stage: Optional[str] = None
    userType: Optional[str] = None
    sessionId: Optional[str] = None
    kioskState: Optional[str] = None
    state: Optional[Dict[str, Any]] = None

class LlmChatRequest(BaseModel):
    messages: List[ChatMessage]
    context: Optional[ChatContext] = None
    sessionId: Optional[str] = None
    orderType: Optional[str] = None

# In-memory chat memory by kiosk session.
CHAT_MEMORY: Dict[str, List[Dict[str, str]]] = defaultdict(list)

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8080").rstrip("/")

# Simple in-process caches (avoid repeated HTTP calls from the AI server).
_MENU_LIST_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}
_MENU_DETAIL_CACHE: Dict[str, Dict[str, Any]] = {}

def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "", s)
    return s

def _canonical_name(s: str) -> str:
    # Basic normalization for matching. Keep conservative (Korean-only product names).
    s = (s or "").strip()
    for tok in ["세트", "단품", "세트메뉴", "세트 메뉴", "(m)", "(r)", "(l)"]:
        s = s.replace(tok, "")
    s = re.sub(r"\s+", "", s)
    return s

def _http_get_json(url: str, timeout_sec: float = 3.0) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ai-kiosk-ai-server/0.2.0",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)

async def _http_get_json_async(url: str, timeout_sec: float = 3.0) -> Dict[str, Any]:
    return await asyncio.to_thread(_http_get_json, url, timeout_sec)

async def _get_kiosk_menu_items_cached(force: bool = False) -> List[Dict[str, Any]]:
    now = time.time()
    if not force and _MENU_LIST_CACHE["data"] is not None and (now - _MENU_LIST_CACHE["ts"]) < 30:
        return _MENU_LIST_CACHE["data"]

    qs = urllib.parse.urlencode({"size": 200})
    url = f"{BACKEND_BASE_URL}/api/v1/kiosk/menu-items?{qs}"
    payload = await _http_get_json_async(url)
    items = (((payload or {}).get("data") or {}).get("items") or [])
    if not isinstance(items, list):
        items = []

    _MENU_LIST_CACHE["ts"] = now
    _MENU_LIST_CACHE["data"] = items
    return items

async def _get_kiosk_menu_detail_cached(menu_item_id: str, force: bool = False) -> Optional[Dict[str, Any]]:
    if not menu_item_id:
        return None
    now = time.time()
    cached = _MENU_DETAIL_CACHE.get(menu_item_id)
    if not force and cached and (now - cached.get("ts", 0.0)) < 300:
        return cached.get("data")

    url = f"{BACKEND_BASE_URL}/api/v1/kiosk/menu-items/{urllib.parse.quote(menu_item_id)}"
    payload = await _http_get_json_async(url)
    data = (payload or {}).get("data")
    if not isinstance(data, dict):
        return None
    _MENU_DETAIL_CACHE[menu_item_id] = {"ts": now, "data": data}
    return data

def _category_label(category_id: Optional[str]) -> str:
    m = {
        "cat_set": "세트 메뉴",
        "cat_burger": "단품",
        "cat_side": "사이드",
        "cat_drink": "음료",
        "cat_chicken": "치킨",
        "cat_best": "베스트메뉴",
    }
    return m.get(category_id or "", "기타")

def _detect_allergen_terms(text: str) -> List[str]:
    # Keep aligned with DB values (Korean strings).
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
    found = []
    for c in candidates:
        if c in text:
            found.append(c)
    for k, v in synonyms.items():
        if k in text and v not in found:
            found.append(v)
    return found

def _is_menu_list_question(text: str) -> bool:
    t = text.replace(" ", "")
    return any(k in t for k in ["뭐뭐있", "메뉴뭐", "메뉴있", "메뉴목록", "전체메뉴", "뭐가있", "뭐있어"])

def _is_similarity_question(text: str) -> bool:
    t = text.replace(" ", "")
    return any(k in t for k in ["비슷", "비슷한", "유사", "비슷한거", "추천"])

def _is_ingredient_question(text: str) -> bool:
    t = text.replace(" ", "")
    return any(k in t for k in ["재료", "들어가", "들어간", "빼고", "없는", "제외", "포함", "알레르기"])

def _resolve_menu_mention(text: str, menu_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Resolve the most likely menu item referenced in user text.
    Uses conservative longest-substring matching over catalog names.
    """
    if not text or not menu_items:
        return None
    t_norm = _norm_text(_canonical_name(text))

    best = None
    best_score = 0
    for it in menu_items:
        name = it.get("name") or ""
        if not name:
            continue
        n_norm = _norm_text(_canonical_name(name))
        if not n_norm:
            continue
        # Exact/substring matching.
        if n_norm and n_norm in t_norm:
            score = len(n_norm)
        elif t_norm and t_norm in n_norm:
            score = max(1, len(t_norm) // 2)
        else:
            continue
        if score > best_score:
            best = it
            best_score = score
    return best

async def _ensure_all_details_cached(menu_items: List[Dict[str, Any]], concurrency: int = 8) -> None:
    # Fetch missing details in parallel with a small concurrency cap.
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(mid: str):
        async with sem:
            try:
                await _get_kiosk_menu_detail_cached(mid)
            except Exception:
                return

    tasks = []
    for it in menu_items:
        mid = it.get("menuItemId")
        if not mid:
            continue
        cached = _MENU_DETAIL_CACHE.get(mid)
        if cached and (time.time() - cached.get("ts", 0.0)) < 300:
            continue
        tasks.append(asyncio.create_task(_one(mid)))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


v2_orchestrator = V2LangChainOrchestrator(
    menu_list_provider=_get_kiosk_menu_items_cached,
    menu_detail_provider=_get_kiosk_menu_detail_cached,
)
qdrant_menu_store: Optional[QdrantMenuVectorStore] = None
if client and os.getenv("QDRANT_URL"):
    qdrant_menu_store = QdrantMenuVectorStore(
        openai_client=client,
        collection_name=os.getenv("QDRANT_MENU_COLLECTION", "kiosk_menu_v2"),
    )


async def _v2_vector_search_candidates(query: str, top_k: int = 6) -> Dict[str, Any]:
    if not qdrant_menu_store:
        return {"results": []}
    return await qdrant_menu_store.search_menus(
        query=query,
        top_k=top_k,
    )


v2_orchestrator.set_vector_search_provider(_v2_vector_search_candidates)

class NluRequest(BaseModel):
    utterance: str

class LlmSuggestRequest(BaseModel):
    hesitationLevel: float
    state: str

class LlmSummarizeRequest(BaseModel):
    history: List[ChatMessage]
    sessionId: str


class V2VectorSyncRequest(BaseModel):
    forceRefresh: bool = False
    size: int = 200


class V2VectorSearchRequest(BaseModel):
    query: str
    topK: int = 5
    categoryId: Optional[str] = None
    includeAllergens: List[str] = []
    excludeAllergens: List[str] = []
    minPrice: Optional[float] = None
    maxPrice: Optional[float] = None

class VisionFrameRequest(BaseModel):
    frameBase64: str
    sessionId: Optional[str] = None
    maxHands: Optional[int] = 2

class SignLanguageRequest(BaseModel):
    handLandmarks: List[Any]

class Landmark(BaseModel):
    x: float
    y: float
    z: float = 0.0
    visibility: Optional[float] = None

class LandmarkGroup(BaseModel):
    landmarks: List[Landmark]

class HesitationRequest(BaseModel):
    face: Optional[LandmarkGroup] = None
    pose: Optional[LandmarkGroup] = None

# --- YHG-pose Models ---

class HesitationResponse(BaseModel):
    """Hesitation response model"""
    hesitation_level: int
    confidence: float
    label: str
    body_score: Optional[float] = None
    face_score: Optional[float] = None
    pose_score: Optional[float] = None
    final_raw: Optional[float] = None
    final_ema: Optional[float] = None
    status: Optional[str] = None
    is_hesitating: Optional[bool] = None
    pose_features: Optional[Dict[str, Any]] = None
    pose_points: Optional[List[Dict[str, float]]] = None
    pose_connections: Optional[List[List[int]]] = None
    probabilities: Optional[List[float]] = None
    error: Optional[str] = None


class SignLanguageResponse(BaseModel):
    """?섑솕 ?몄떇 ?묐떟 紐⑤뜽"""
    text: str
    error: Optional[str] = None


class Base64ImageRequest(BaseModel):
    """Base64 ?대?吏 ?붿껌 紐⑤뜽"""
    image: str  # Base64 encoded image
    binary: bool = False  # ?댁쭊 遺꾨쪟 紐⑤뱶

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <body>
        <div>
            <h1>?쨼 AI Kiosk - AI Server</h1>
            <p>FastAPI Server is running with OpenAI Integration.</p>
            <p class="tech">Python + FastAPI + MediaPipe + OpenAI</p>
            <p style="margin-top: 2rem; font-size: 0.9rem; opacity: 0.7">FusionCrew 짤 2025~2026</p>
            <p><a href="/hesitation" style="color: #00bcd4; text-decoration: none;">?몛 View Hesitation Dashboard</a></p>
        </div>
    </body>
    </html>
    """

@app.get("/hesitation", response_class=HTMLResponse)
async def hesitation_page():
    template_path = Path(__file__).parent / "templates" / "hesitation.html"
    if not template_path.exists():
        return HTMLResponse(content="<h1>Error: Template not found</h1>", status_code=404)
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/api/ping")
async def ping():
    return {"message": "pong", "server": "ai-server", "openai_connected": bool(client)}

@app.get("/api/v1/meta/health")
@app.get("/api/v2/meta/health")
async def meta_health():
    return {"status": "UP"}

@app.get("/api/v1/meta/models")
@app.get("/api/v2/meta/models")
async def meta_models():
    return {
        "models": [
            {"name": "whisper-1", "loaded": True, "provider": "OpenAI"},
            {"name": "tts-1", "loaded": True, "provider": "OpenAI"},
            {"name": "gpt-3.5-turbo", "loaded": True, "provider": "OpenAI"},
            {"name": "mediapipe", "loaded": True, "provider": "Local"},
            {"name": "hesitation-model", "loaded": True, "provider": "Local"}, # Added
        ]
    }

# 1. STT (OpenAI Whisper)
@app.post("/api/v1/stt")
@app.post("/api/v2/stt")
async def stt(request: SttRequest):
    if not client:
        return CommonResponse(success=False, error={"code": "NO_API_KEY", "message": "OpenAI API Key not found"})

    temp_audio_path = None
    try:
        # Decode Base64 payload (supports raw base64 and data URL).
        b64 = request.audioBase64.split(",", 1)[-1]
        audio_data = base64.b64decode(b64)

        suffix = ".wav"
        mt = (request.mimeType or "").lower()
        if "webm" in mt:
            suffix = ".webm"
        elif "mp3" in mt or "mpeg" in mt:
            suffix = ".mp3"
        elif "mp4" in mt or "m4a" in mt:
            suffix = ".m4a"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name

        # Call OpenAI Whisper
        with open(temp_audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=request.model or "whisper-1",
                file=audio_file,
                language=request.language,
                prompt="키오스크 주문 대화. 주요 단어: 주문하기, 이전, 뒤로, 버거, 사이드, 음료, 세트, 단품, 결제, 매장, 포장."
            )
        
        return CommonResponse(
            success=True,
            data={
                "text": transcript.text,
                "confidence": 0.99 
            },
            meta={"model": request.model or "whisper-1", "provider": "openai"},
            requestId=f"req_stt_{int(datetime.now().timestamp())}"
        )
    except Exception as e:
        err_msg = str(e)
        if "invalid_api_key" in err_msg or "Incorrect API key provided" in err_msg:
            err_msg = "Invalid OpenAI API key"
        return CommonResponse(success=False, error={"code": "STT_FAILED", "message": err_msg})
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass

# 2. TTS (OpenAI TTS)
@app.post("/api/v1/tts")
@app.post("/api/v2/tts")
async def tts(request: TtsRequest):
    if not client:
        return CommonResponse(success=False, error={"code": "NO_API_KEY", "message": "OpenAI API Key not found"})

    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=request.voice if request.voice in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"] else "nova",
            input=request.text,
            speed=request.speed
        )
        
        # Binary to Base64
        audio_content = response.content
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')

        return CommonResponse(
            success=True,
            data={"audioBase64": audio_base64},
            meta={"model": "tts-1", "provider": "openai"},
            requestId=f"req_tts_{int(datetime.now().timestamp())}"
        )
    except Exception as e:
        return CommonResponse(success=False, error={"code": "TTS_FAILED", "message": str(e)})

@app.post("/api/v1/llm/chat")
async def llm_chat(request: LlmChatRequest):
    if not client:
        return CommonResponse(success=False, error={"code": "NO_API_KEY", "message": "OpenAI API Key not found"})

    try:
        # Prefer explicit top-level sessionId used by frontend.
        session_id = request.sessionId or (request.context.sessionId if request.context else None) or "default"
        order_type = request.orderType or "UNKNOWN"

        latest_user = ""
        for m in reversed(request.messages):
            if m.role == "user":
                latest_user = m.content.strip()
                break
        if not latest_user:
            return CommonResponse(success=False, error={"code": "INVALID_REQUEST", "message": "No user message"})

        state = (request.context.state if request.context else None) or {}
        # Prefer catalog from frontend state (already in sync with UI). Fallback to backend list.
        menu_items: List[Dict[str, Any]] = []
        if isinstance(state, dict):
            mc = state.get("menuCatalog")
            if isinstance(mc, list) and mc:
                menu_items = mc
        if not menu_items:
            try:
                menu_items = await _get_kiosk_menu_items_cached()
            except Exception:
                menu_items = []

        # -------------------------
        # DB-grounded deterministic Q&A (avoid hallucinations)
        # -------------------------
        user_text = latest_user.strip()

        # 1) Menu list / catalog question
        if _is_menu_list_question(user_text):
            # Keep it short for voice UX.
            counts: Dict[str, int] = {}
            for it in menu_items:
                cid = it.get("categoryId")
                label = _category_label(cid)
                counts[label] = counts.get(label, 0) + 1
            parts = [f"{k} {v}개" for k, v in counts.items() if v > 0]
            parts_str = ", ".join(parts) if parts else "메뉴 데이터가 아직 준비되지 않았어요"
            reply = f"{parts_str}가 있어요. 어떤 카테고리를 보여드릴까요?"
            CHAT_MEMORY[session_id].extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply},
            ])
            CHAT_MEMORY[session_id] = CHAT_MEMORY[session_id][-20:]
            return CommonResponse(
                success=True,
                data={"reply": reply, "text": reply, "intent": "MENU_INFO", "action": "NONE", "actionData": {}},
                meta={"sessionId": session_id},
                requestId=f"req_chat_{int(datetime.now().timestamp())}",
            )

        # 2) Ingredient / allergen question
        allergen_terms = _detect_allergen_terms(user_text)
        mentioned = _resolve_menu_mention(user_text, menu_items)

        # If user asks about a specific menu's ingredients/allergens, answer from detail.
        if mentioned and (_is_ingredient_question(user_text) or allergen_terms):
            mid = mentioned.get("menuItemId")
            detail = None
            try:
                detail = await _get_kiosk_menu_detail_cached(mid)
            except Exception:
                detail = None

            if not detail:
                reply = "해당 메뉴 정보를 DB에서 찾지 못했어요. 다른 메뉴로 다시 말씀해 주세요."
            else:
                if ("칼로리" in user_text) or ("kcal" in user_text.lower()):
                    n = detail.get("nutrition") or {}
                    kcal = n.get("kcal")
                    if kcal is None:
                        reply = f"{detail.get('name','해당 메뉴')} 칼로리 정보가 DB에 없어요."
                    else:
                        reply = f"{detail.get('name','해당 메뉴')} 칼로리는 {kcal}kcal입니다."
                elif ("알레르기" in user_text) or allergen_terms:
                    alls = detail.get("allergies") or []
                    if not alls:
                        reply = f"{detail.get('name','해당 메뉴')} 알레르기 정보가 DB에 없어요."
                    else:
                        reply = f"{detail.get('name','해당 메뉴')} 알레르기는 {', '.join(alls)}입니다."
                else:
                    ings = detail.get("ingredients") or []
                    if not ings:
                        reply = f"{detail.get('name','해당 메뉴')} 재료 정보가 아직 준비 중이에요."
                    else:
                        reply = f"{detail.get('name','해당 메뉴')} 재료는 {', '.join(ings)}입니다."

            CHAT_MEMORY[session_id].extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply},
            ])
            CHAT_MEMORY[session_id] = CHAT_MEMORY[session_id][-20:]
            return CommonResponse(
                success=True,
                data={"reply": reply, "text": reply, "intent": "MENU_INFO", "action": "NONE", "actionData": {}},
                meta={"sessionId": session_id},
                requestId=f"req_chat_{int(datetime.now().timestamp())}",
            )

        # Ingredient include/exclude across menus (uses list endpoint ingredients if available).
        if _is_ingredient_question(user_text) and not mentioned and not allergen_terms:
            # Extract a conservative ingredient keyword from text.
            tokens = re.findall(r"[0-9A-Za-z가-힣]+", user_text)
            stop = {
                "메뉴", "재료", "알레르기", "있어", "뭐", "뭐뭐", "들어가", "들어간", "없는", "빼고", "제외", "포함", "추천", "비슷한",
            }
            cand = ""
            for tok in tokens:
                if tok in stop:
                    continue
                if len(tok) > len(cand):
                    cand = tok

            if cand:
                # If frontend catalog doesn't include ingredients, use backend list (it includes ingredients).
                if not any(isinstance((it.get("ingredients") if isinstance(it, dict) else None), list) for it in menu_items):
                    try:
                        menu_items = await _get_kiosk_menu_items_cached()
                    except Exception:
                        pass
                want_exclude = any(k in user_text for k in ["없는", "빼고", "제외"])
                matched = []
                for it in menu_items:
                    ings = it.get("ingredients") or []
                    if not isinstance(ings, list):
                        continue
                    has = any(cand in (x or "") for x in ings)
                    if (not want_exclude and has) or (want_exclude and not has):
                        matched.append(it.get("name"))
                matched = [m for m in matched if m]
                if not matched:
                    reply = f"DB 기준으로 '{cand}' 조건에 맞는 메뉴를 찾지 못했어요."
                else:
                    show = matched[:8]
                    suffix = f" 등 {len(matched)}개가 있어요." if len(matched) > len(show) else "가 있어요."
                    reply = f"'{cand}' {'없는' if want_exclude else '들어간'} 메뉴는 {', '.join(show)}{suffix}"
                CHAT_MEMORY[session_id].extend([
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": reply},
                ])
                CHAT_MEMORY[session_id] = CHAT_MEMORY[session_id][-20:]
                return CommonResponse(
                    success=True,
                    data={"reply": reply, "text": reply, "intent": "MENU_INFO", "action": "NONE", "actionData": {}},
                    meta={"sessionId": session_id},
                    requestId=f"req_chat_{int(datetime.now().timestamp())}",
                )

        # Allergen include/exclude across menus (requires per-menu detail).
        if allergen_terms and not mentioned:
            want_exclude = any(k in user_text for k in ["없는", "빼고", "제외"])
            try:
                await _ensure_all_details_cached(menu_items)
            except Exception:
                pass

            term = allergen_terms[0]
            matched = []
            for it in menu_items:
                mid = it.get("menuItemId")
                if not mid:
                    continue
                det = (_MENU_DETAIL_CACHE.get(mid) or {}).get("data") or {}
                alls = det.get("allergies") or []
                has = term in alls
                if (not want_exclude and has) or (want_exclude and not has):
                    matched.append(it.get("name"))
            matched = [m for m in matched if m]
            if not matched:
                reply = f"DB 기준으로 '{term}' 조건에 맞는 메뉴를 찾지 못했어요."
            else:
                show = matched[:8]
                suffix = f" 등 {len(matched)}개가 있어요." if len(matched) > len(show) else "가 있어요."
                reply = f"'{term}' {'없는' if want_exclude else '포함된'} 메뉴는 {', '.join(show)}{suffix}"

            CHAT_MEMORY[session_id].extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply},
            ])
            CHAT_MEMORY[session_id] = CHAT_MEMORY[session_id][-20:]
            return CommonResponse(
                success=True,
                data={"reply": reply, "text": reply, "intent": "MENU_INFO", "action": "NONE", "actionData": {}},
                meta={"sessionId": session_id},
                requestId=f"req_chat_{int(datetime.now().timestamp())}",
            )

        # Similar menu suggestion (DB-grounded, ingredient overlap).
        if _is_similarity_question(user_text) and mentioned:
            if not any(isinstance((it.get("ingredients") if isinstance(it, dict) else None), list) for it in menu_items):
                try:
                    menu_items = await _get_kiosk_menu_items_cached()
                except Exception:
                    pass
            target_id = mentioned.get("menuItemId")
            target_detail = await _get_kiosk_menu_detail_cached(target_id)
            target_ings = set((target_detail or {}).get("ingredients") or [])
            scored = []
            for it in menu_items:
                mid = it.get("menuItemId")
                if not mid or mid == target_id:
                    continue
                ings = it.get("ingredients") or []
                if not isinstance(ings, list):
                    continue
                s = set(ings)
                inter = len(target_ings & s)
                union = len(target_ings | s) or 1
                score = inter / union
                scored.append((score, it.get("name")))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = [n for _, n in scored[:3] if n]
            if top:
                reply = f"{mentioned.get('name','해당 메뉴')}과 비슷한 메뉴로는 {', '.join(top)}를 추천드려요."
            else:
                reply = "비슷한 메뉴를 DB에서 찾기 어려워요. 다른 메뉴로 다시 말씀해 주세요."

            CHAT_MEMORY[session_id].extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply},
            ])
            CHAT_MEMORY[session_id] = CHAT_MEMORY[session_id][-20:]
            return CommonResponse(
                success=True,
                data={"reply": reply, "text": reply, "intent": "MENU_INFO", "action": "NONE", "actionData": {}},
                meta={"sessionId": session_id},
                requestId=f"req_chat_{int(datetime.now().timestamp())}",
            )

        # -------------------------
        # Ordering orchestrator (OpenAI) with DB catalog constraint
        # -------------------------
        catalog = []
        for it in menu_items[:200]:
            mid = it.get("menuItemId")
            name = it.get("name")
            if mid and name:
                catalog.append(
                    {
                        "menuItemId": mid,
                        "name": name,
                        "categoryId": it.get("categoryId"),
                        "price": it.get("price"),
                    }
                )
        catalog_json = json.dumps(catalog, ensure_ascii=False)
        state_json = json.dumps(state, ensure_ascii=False)

        system_prompt = (
            "너는 패스트푸드 키오스크 음성 주문 오케스트레이터다.\n"
            "반드시 JSON만 출력해라. 스키마:\n"
            "{"
            "\"speech\": string, "
            "\"action\": \"NONE|NAVIGATE|ADD_MENU|REMOVE_MENU|CHANGE_QTY|CHECK_CART|CHECKOUT|SELECT_PAYMENT|CONTINUE_ORDER\", "
            "\"actionData\": object"
            "}\n"
            "절대 규칙:\n"
            "1) menuItemId는 제공된 메뉴 카탈로그에 있는 값만 사용한다. 없으면 action=NONE으로 두고 정확한 메뉴명을 되묻는다.\n"
            "2) action은 명확할 때만 설정한다. 모호하면 action=NONE으로 두고 speech에서 되묻는다.\n"
            "3) ADD/REMOVE/CHANGE_QTY는 반드시 menuItemId를 actionData에 넣는다.\n"
            "4) CHANGE_QTY는 quantity(정수>=1)를 넣는다.\n"
            "5) SELECT_PAYMENT는 method(CARD|POINT|SIMPLE)를 넣는다.\n"
            "6) 한국어 존댓말, 한 문장 또는 두 문장 이내.\n"
            f"현재 주문 타입: {order_type}\n"
            f"현재 상태(JSON): {state_json}\n"
            f"메뉴 카탈로그(JSON): {catalog_json}\n"
        )

        # Use frontend-provided history if available; fallback to server-side memory.
        history_msgs: List[Dict[str, str]] = []
        for m in request.messages[-10:]:
            role = (m.role or "").strip()
            content = (m.content or "").strip()
            if role in ("user", "assistant") and content:
                history_msgs.append({"role": role, "content": content})
        if not history_msgs:
            history_msgs = CHAT_MEMORY[session_id][-12:]

        # Ensure last user message exists once.
        if not history_msgs or history_msgs[-1].get("role") != "user":
            history_msgs.append({"role": "user", "content": user_text})

        messages = [{"role": "system", "content": system_prompt}] + history_msgs[-12:]

        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            messages=messages,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            raw = "{\"speech\":\"잘 들었어요. 원하시는 메뉴를 한 번 더 말씀해 주세요.\",\"action\":\"NONE\",\"actionData\":{}}"

        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            # Try to extract JSON block if model wrapped text around it.
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(raw[start:end + 1])
                except Exception:
                    parsed = None

        if not isinstance(parsed, dict):
            parsed = {
                "speech": raw if raw else "원하시는 메뉴를 다시 말씀해 주세요.",
                "action": "NONE",
                "actionData": {},
            }

        reply = str(parsed.get("speech") or "").strip() or "원하시는 메뉴를 다시 말씀해 주세요."
        action = str(parsed.get("action") or "NONE").upper()
        allowed = {"NONE", "NAVIGATE", "ADD_MENU", "REMOVE_MENU", "CHANGE_QTY", "CHECK_CART", "CHECKOUT", "SELECT_PAYMENT", "CONTINUE_ORDER"}
        if action not in allowed:
            action = "NONE"
        action_data = parsed.get("actionData") if isinstance(parsed.get("actionData"), dict) else {}

        CHAT_MEMORY[session_id].extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ])
        CHAT_MEMORY[session_id] = CHAT_MEMORY[session_id][-20:]

        return CommonResponse(
            success=True,
            data={
                "reply": reply,
                "text": reply,
                "intent": "GENERAL",
                "action": action,
                "actionData": action_data,
            },
            meta={"model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), "sessionId": session_id},
            requestId=f"req_chat_{int(datetime.now().timestamp())}"
        )
    except Exception as e:
        return CommonResponse(success=False, error={"code": "LLM_FAILED", "message": str(e)})


@app.post("/api/v2/llm/chat")
async def llm_chat_v2(request: LlmChatRequest):
    if not client:
        return CommonResponse(success=False, error={"code": "NO_API_KEY", "message": "OpenAI API Key not found"})

    request_id = f"req_chat_v2_{int(datetime.now().timestamp())}"
    started = time.perf_counter()
    try:
        session_id = request.sessionId or (request.context.sessionId if request.context else None) or "default"

        pack = await v2_orchestrator.run(
            request=request,
            openai_client=client,
            request_id=request_id,
        )
        result = pack.get("result", {})
        trace = pack.get("trace", [])
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "[v2][llm_chat] requestId=%s sessionId=%s elapsedMs=%s action=%s orchestrator=%s trace=%s",
            request_id,
            session_id,
            elapsed_ms,
            str(result.get("action") or "NONE"),
            str(result.get("orchestrator") or "unknown"),
            json.dumps(trace, ensure_ascii=False),
        )

        return CommonResponse(
            success=True,
            data=result,
            meta={
                "model": os.getenv("OPENAI_CHAT_MODEL_V2", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")),
                "sessionId": session_id,
                "version": "v2",
                "elapsedMs": elapsed_ms,
                "trace": trace,
            },
            requestId=request_id,
        )
    except Exception:
        # Keep v2 safe during migration by falling back to the proven v1 path.
        logger.exception("[v2][llm_chat] requestId=%s fallback_to_v1", request_id)
        return await llm_chat(request)


@app.post("/api/v2/debug/llm/chat")
async def llm_chat_v2_debug(request: LlmChatRequest):
    if not client:
        return CommonResponse(success=False, error={"code": "NO_API_KEY", "message": "OpenAI API Key not found"})

    request_id = f"req_chat_v2_dbg_{int(datetime.now().timestamp())}"
    started = time.perf_counter()
    try:
        session_id = request.sessionId or (request.context.sessionId if request.context else None) or "default"
        state = (request.context.state if request.context else None) or {}
        inferred_stage = v2_orchestrator._infer_stage(state if isinstance(state, dict) else {})

        pack = await v2_orchestrator.run(
            request=request,
            openai_client=client,
            request_id=request_id,
        )
        result = pack.get("result", {})
        trace = pack.get("trace", [])
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        return CommonResponse(
            success=True,
            data={
                "result": result,
                "trace": trace,
                "debug": {
                    "inferredStage": inferred_stage,
                    "resultStage": result.get("stage"),
                    "hasVectorCandidatesInState": bool(
                        isinstance(state, dict) and isinstance(state.get("vectorCandidates"), list) and len(state.get("vectorCandidates")) > 0
                    ),
                    "messageCount": len(request.messages or []),
                    "elapsedMs": elapsed_ms,
                    "requestId": request_id,
                },
            },
            meta={
                "model": os.getenv("OPENAI_CHAT_MODEL_V2", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")),
                "sessionId": session_id,
                "version": "v2",
                "debug": True,
            },
            requestId=request_id,
        )
    except Exception as e:
        logger.exception("[v2][llm_chat_debug] requestId=%s failed", request_id)
        return CommonResponse(success=False, error={"code": "LLM_V2_DEBUG_FAILED", "message": str(e)})


@app.post("/api/v2/vector/sync-menus")
async def v2_vector_sync_menus(request: V2VectorSyncRequest):
    if not qdrant_menu_store:
        return CommonResponse(
            success=False,
            error={"code": "QDRANT_DISABLED", "message": "Qdrant is not configured. Set QDRANT_URL first."},
        )

    try:
        menu_items = await _get_kiosk_menu_items_cached(force=bool(request.forceRefresh))
        if not isinstance(menu_items, list):
            menu_items = []
        if request.size > 0:
            menu_items = menu_items[: int(request.size)]

        result = await qdrant_menu_store.upsert_menu_items(
            menu_items=menu_items,
            detail_provider=_get_kiosk_menu_detail_cached,
        )
        return CommonResponse(
            success=True,
            data=result,
            meta={"version": "v2", "vectorDb": "qdrant"},
            requestId=f"req_v2_vec_sync_{int(datetime.now().timestamp())}",
        )
    except Exception as e:
        logger.exception("[v2][vector_sync] failed")
        return CommonResponse(success=False, error={"code": "VECTOR_SYNC_FAILED", "message": str(e)})


@app.post("/api/v2/vector/search-menus")
async def v2_vector_search_menus(request: V2VectorSearchRequest):
    if not qdrant_menu_store:
        return CommonResponse(
            success=False,
            error={"code": "QDRANT_DISABLED", "message": "Qdrant is not configured. Set QDRANT_URL first."},
        )

    try:
        result = await qdrant_menu_store.search_menus(
            query=request.query,
            top_k=request.topK,
            category_id=request.categoryId,
            include_allergens=request.includeAllergens or [],
            exclude_allergens=request.excludeAllergens or [],
            min_price=request.minPrice,
            max_price=request.maxPrice,
        )
        return CommonResponse(
            success=True,
            data=result,
            meta={"version": "v2", "vectorDb": "qdrant"},
            requestId=f"req_v2_vec_search_{int(datetime.now().timestamp())}",
        )
    except Exception as e:
        logger.exception("[v2][vector_search] failed")
        return CommonResponse(success=False, error={"code": "VECTOR_SEARCH_FAILED", "message": str(e)})

# 4. NLU
@app.post("/api/v1/nlu/parse")
@app.post("/api/v2/nlu/parse")
async def nlu_parse(request: NluRequest):
    return CommonResponse(
        success=True,
        data={
            "intent": "ORDER",
            "slots": {"menu": [{"name": "Mock Item", "quantity": 1}]}
        },
        requestId="req_nlu_01"
    )

# 5. Suggest
@app.post("/api/v1/llm/suggest")
@app.post("/api/v2/llm/suggest")
async def llm_suggest(request: LlmSuggestRequest):
    return CommonResponse(
        success=True,
        data={"suggestion": "메뉴 선택이 어려우시면 추천 메뉴를 안내해드릴게요."},
        requestId="req_suggest_01"
    )

# 6. Summarize
@app.post("/api/v1/llm/summarize")
@app.post("/api/v2/llm/summarize")
async def llm_summarize(request: LlmSummarizeRequest):
    return CommonResponse(
        success=True,
        data={
            "summary": "Mock Summary",
            "compressedContext": {"lastTopic": "Mock"}
        },
        requestId="req_summarize_01"
    )

# 7 ~ 11. Vision (Mocks kept as is)
@app.post("/api/v1/vision/facemesh")
@app.post("/api/v2/vision/facemesh")
async def vision_facemesh(request: VisionFrameRequest):
    return CommonResponse(success=True, data={"landmarks": [], "count": 0}, requestId="req_facemesh")

@app.post("/api/v1/vision/pose")
@app.post("/api/v2/vision/pose")
async def vision_pose(request: VisionFrameRequest):
    return CommonResponse(success=True, data={"landmarks": []}, requestId="req_pose")

@app.post("/api/v1/vision/hands")
@app.post("/api/v2/vision/hands")
async def vision_hands(request: VisionFrameRequest):
    return CommonResponse(success=True, data={"hands": []}, requestId="req_hands")

@app.post("/api/v1/vision/sign-language/interpret")
@app.post("/api/v2/vision/sign-language/interpret")
async def sign_language(request: SignLanguageRequest):
    return CommonResponse(success=True, data={"commands": []}, requestId="req_sign")
    
@app.post("/api/v1/vision/hesitation")
@app.post("/api/v2/vision/hesitation")
async def hesitation(request: HesitationRequest):
    return CommonResponse(success=True, data={"score": 0.0, "level": "LOW", "signals": []}, requestId="req_hesitation")


# ============================================
# Hesitation Detection API (From YHG-pose)
# ============================================

@app.post("/api/hesitation/detect", response_model=HesitationResponse)
async def detect_hesitation_from_image(image: UploadFile = File(...)):
    """
    ?대?吏?먯꽌 留앹꽕???뺣룄 媛먯?
    
    - **image**: ?대?吏 ?뚯씪 (JPEG, PNG ??
    - Returns: 留앹꽕???덈꺼 (0-3), ?좊ː?? ?쇰꺼
    """
    try:
        # ?대?吏 ?쎄린
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # 留앹꽕??媛먯?
        from hesitationLearning.inference import detect_hesitation
        result = detect_hesitation(img, binary=False)
        
        return HesitationResponse(**result)
        
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Hesitation detection model not available. Please train the model first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hesitation/detect-base64", response_model=HesitationResponse)
@app.post("/api/v1/hesitation/detect-base64", response_model=HesitationResponse)
@app.post("/api/v2/hesitation/detect-base64", response_model=HesitationResponse)
async def detect_hesitation_from_base64(request: Base64ImageRequest):
    """
    Base64 ?몄퐫?⑸맂 ?대?吏?먯꽌 留앹꽕??媛먯?
    
    - **image**: Base64 ?몄퐫?⑸맂 ?대?吏 臾몄옄??
    - **binary**: True硫??댁쭊 遺꾨쪟 (留앹꽕??鍮꾨쭩?ㅼ엫)
    - Returns: 留앹꽕???덈꺼, ?좊ː?? ?쇰꺼
    """
    try:
        from hesitationLearning.inference import get_detector
        detector = get_detector(binary=request.binary)
        result = detector.detect_from_base64(request.image)
        if isinstance(result, dict) and result.get("error"):
            return HesitationResponse(
                hesitation_level=0,
                confidence=0.0,
                label="NORMAL",
                body_score=0.0,
                face_score=0.0,
                pose_score=0.0,
                final_raw=0.0,
                final_ema=0.0,
                status="NORMAL",
                is_hesitating=False,
                pose_features={},
                pose_points=[],
                pose_connections=[],
                error=str(result.get("error")),
            )

        return HesitationResponse(**result)
        
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Hesitation detection model not available. Please train the model first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hesitation/status")
async def hesitation_model_status():
    """留앹꽕??媛먯? 紐⑤뜽 ?곹깭 ?뺤씤"""
    from pathlib import Path
    from hesitationLearning.config import MODEL_PATH, SCALER_PATH
    
    model_exists = MODEL_PATH.exists()
    scaler_exists = SCALER_PATH.exists()
    
    return {
        "model_available": model_exists and scaler_exists,
        "model_path": str(MODEL_PATH),
        "message": "Model ready" if (model_exists and scaler_exists) else "Model not trained yet"
    }


# ============================================
# Sign Language Translation API (From YHG-pose)
# ============================================

@app.post("/api/sign-language/translate", response_model=SignLanguageResponse)
async def translate_sign_language(video: UploadFile = File(...)):
    """
    ?섑솕 鍮꾨뵒??踰덉뿭 API
    
    - **video**: 鍮꾨뵒???뚯씪 (MP4, AVI ??
    - Returns: 踰덉뿭???띿뒪??
    """
    import tempfile
    import os
    from signLanguage.inference import HandTranslator
    
    # ?꾩떆 ?뚯씪濡????
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        translator = HandTranslator()
        # 鍮꾨뵒??泥섎━
        result_text = translator.process_video(tmp_path)
        
        if result_text is None:
             raise HTTPException(status_code=400, detail="Could not process video")
             
        return SignLanguageResponse(text=result_text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # ?꾩떆 ?뚯씪 ??젣
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

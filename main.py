from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import pytz
import os
import base64
import tempfile
import json
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict

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

class NluRequest(BaseModel):
    utterance: str

class LlmSuggestRequest(BaseModel):
    hesitationLevel: float
    state: str

class LlmSummarizeRequest(BaseModel):
    history: List[ChatMessage]
    sessionId: str

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
async def meta_health():
    return {"status": "UP"}

@app.get("/api/v1/meta/models")
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

        # Keep a compact memory window per session.
        memory = CHAT_MEMORY[session_id][-12:]
        state = (request.context.state if request.context else None) or {}
        state_json = json.dumps(state, ensure_ascii=False)
        system_prompt = (
            "너는 패스트푸드 키오스크 음성 주문 오케스트레이터다.\n"
            "반드시 JSON만 출력해라. 스키마:\n"
            "{"
            "\"speech\": string, "
            "\"action\": \"NONE|NAVIGATE|ADD_MENU|REMOVE_MENU|CHANGE_QTY|CHECK_CART|CHECKOUT|SELECT_PAYMENT|CONTINUE_ORDER\", "
            "\"actionData\": object"
            "}\n"
            "규칙:\n"
            "1) action은 명확할 때만 설정한다. 모호하면 action=NONE으로 두고 speech에서 되묻는다.\n"
            "2) ADD/REMOVE/CHANGE_QTY는 반드시 menuItemId를 actionData에 넣는다.\n"
            "3) CHANGE_QTY는 quantity(정수>=1)를 넣는다.\n"
            "4) SELECT_PAYMENT는 method(CARD|POINT|SIMPLE)를 넣는다.\n"
            "5) 장바구니가 여러 개인데 사용자가 '빼줘'만 말하면 NONE + 어떤 메뉴를 삭제할지 질문한다.\n"
            "6) 한국어 존댓말, 한 문장 또는 두 문장 이내.\n"
            f"현재 주문 타입: {order_type}\n"
            f"현재 상태(JSON): {state_json}\n"
        )

        messages = [{"role": "system", "content": system_prompt}] + memory + [{"role": "user", "content": latest_user}]

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
            {"role": "user", "content": latest_user},
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

# 4. NLU
@app.post("/api/v1/nlu/parse")
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
async def llm_suggest(request: LlmSuggestRequest):
    return CommonResponse(
        success=True,
        data={"suggestion": "메뉴 선택이 어려우시면 추천 메뉴를 안내해드릴게요."},
        requestId="req_suggest_01"
    )

# 6. Summarize
@app.post("/api/v1/llm/summarize")
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
async def vision_facemesh(request: VisionFrameRequest):
    return CommonResponse(success=True, data={"landmarks": [], "count": 0}, requestId="req_facemesh")

@app.post("/api/v1/vision/pose")
async def vision_pose(request: VisionFrameRequest):
    return CommonResponse(success=True, data={"landmarks": []}, requestId="req_pose")

@app.post("/api/v1/vision/hands")
async def vision_hands(request: VisionFrameRequest):
    return CommonResponse(success=True, data={"hands": []}, requestId="req_hands")

@app.post("/api/v1/vision/sign-language/interpret")
async def sign_language(request: SignLanguageRequest):
    return CommonResponse(success=True, data={"commands": []}, requestId="req_sign")
    
@app.post("/api/v1/vision/hesitation")
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


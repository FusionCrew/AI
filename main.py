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
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import numpy as np
import cv2

# Load env from Backend folder as user specified
env_path = Path(__file__).parent.parent / "Backend" / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API")
client = OpenAI(api_key=api_key) if api_key else None

app = FastAPI(
    title="AI Kiosk - AI Server",
    description="MediaPipe Pose와 FaceMesh용 딥러닝 모델 구동 서버 + OpenAI Integration",
    version="0.2.0"
)

# CORS 설정
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

class TtsRequest(BaseModel):
    text: str
    voice: str = "alloy"
    speed: float = 1.0

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatContext(BaseModel):
    stage: Optional[str] = None
    userType: Optional[str] = None
    sessionId: Optional[str] = None
    kioskState: Optional[str] = None

class LlmChatRequest(BaseModel):
    messages: List[ChatMessage]
    context: Optional[ChatContext] = None

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
    """망설임 감지 응답 모델"""
    hesitation_level: int
    confidence: float
    label: str
    probabilities: Optional[List[float]] = None
    error: Optional[str] = None


class SignLanguageResponse(BaseModel):
    """수화 인식 응답 모델"""
    text: str
    error: Optional[str] = None


class Base64ImageRequest(BaseModel):
    """Base64 이미지 요청 모델"""
    image: str  # Base64 encoded image
    binary: bool = False  # 이진 분류 모드

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <body>
        <div>
            <h1>🤖 AI Kiosk - AI Server</h1>
            <p>FastAPI Server is running with OpenAI Integration.</p>
            <p class="tech">Python + FastAPI + MediaPipe + OpenAI</p>
            <p style="margin-top: 2rem; font-size: 0.9rem; opacity: 0.7">FusionCrew © 2025~2026</p>
        </div>
    </body>
    </html>
    """

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

    try:
        # Decode Base64 to temp file
        audio_data = base64.b64decode(request.audioBase64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name

        # Call OpenAI Whisper
        with open(temp_audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=request.language
            )
        
        # Cleanup
        os.remove(temp_audio_path)

        return CommonResponse(
            success=True,
            data={
                "text": transcript.text,
                "confidence": 0.99 
            },
            meta={"model": "whisper-1", "provider": "openai"},
            requestId=f"req_stt_{int(datetime.now().timestamp())}"
        )
    except Exception as e:
        return CommonResponse(success=False, error={"code": "STT_FAILED", "message": str(e)})

# 2. TTS (OpenAI TTS)
@app.post("/api/v1/tts")
async def tts(request: TtsRequest):
    if not client:
        return CommonResponse(success=False, error={"code": "NO_API_KEY", "message": "OpenAI API Key not found"})

    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=request.voice if request.voice in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"] else "alloy",
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

# 3. Chat (Mock for now, can perform actual LLM call if needed)
@app.post("/api/v1/llm/chat")
async def llm_chat(request: LlmChatRequest):
    return CommonResponse(
        success=True,
        data={
            "reply": f"AI 응답입니다: {request.messages[-1].content}에 대해 확인해보겠습니다.",
            "intent": "GENERAL"
        },
        requestId="req_chat_01"
    )

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
        data={"suggestion": "메뉴 선택이 어려우시면 추천을 도와드릴게요."},
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
    이미지에서 망설임 정도 감지
    
    - **image**: 이미지 파일 (JPEG, PNG 등)
    - Returns: 망설임 레벨 (0-3), 신뢰도, 라벨
    """
    try:
        # 이미지 읽기
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # 망설임 감지
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
async def detect_hesitation_from_base64(request: Base64ImageRequest):
    """
    Base64 인코딩된 이미지에서 망설임 감지
    
    - **image**: Base64 인코딩된 이미지 문자열
    - **binary**: True면 이진 분류 (망설임/비망설임)
    - Returns: 망설임 레벨, 신뢰도, 라벨
    """
    try:
        from hesitationLearning.inference import get_detector
        detector = get_detector(binary=request.binary)
        result = detector.detect_from_base64(request.image)
        
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
    """망설임 감지 모델 상태 확인"""
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
    수화 비디오 번역 API
    
    - **video**: 비디오 파일 (MP4, AVI 등)
    - Returns: 번역된 텍스트
    """
    import tempfile
    import os
    from signLanguage.inference import HandTranslator
    
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        translator = HandTranslator()
        # 비디오 처리
        result_text = translator.process_video(tmp_path)
        
        if result_text is None:
             raise HTTPException(status_code=400, detail="Could not process video")
             
        return SignLanguageResponse(text=result_text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 임시 파일 삭제
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

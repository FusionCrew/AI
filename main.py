from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import cv2

app = FastAPI(
    title="AI Kiosk - AI Server",
    description="MediaPipe Pose와 FaceMesh용 딥러닝 모델 구동 서버",
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


@app.get("/", response_class=HTMLResponse)
async def root():
    """Hello World 페이지"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AI Kiosk - AI Server</title>
        <style>
            body {
                font-family: system-ui, -apple-system, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
            }
            div {
                text-align: center;
            }
            h1 {
                font-size: 3rem;
                margin-bottom: 1rem;
            }
            p {
                font-size: 1.2rem;
                opacity: 0.9;
            }
            .tech {
                margin-top: 2rem;
                font-size: 0.9rem;
                opacity: 0.7;
            }
        </style>
    </head>
    <body>
        <div>
            <h1>🤖 AI Kiosk - AI Server</h1>
            <p>Hello World! FastAPI Server is running.</p>
            <p class="tech">Python + FastAPI + MediaPipe + OpenAI</p>
            <p style="margin-top: 2rem; font-size: 0.9rem; opacity: 0.7">FusionCrew © 2025~2026</p>
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    """헬스 체크"""
    return {"status": "ok"}


@app.get("/api/ping")
async def ping():
    """API 연결 테스트"""
    return {"message": "pong", "server": "ai-server"}


# ============================================
# Hesitation Detection API
# ============================================

class HesitationResponse(BaseModel):
    """망설임 감지 응답 모델"""
    hesitation_level: int
    confidence: float
    label: str
    probabilities: Optional[List[float]] = None
    error: Optional[str] = None


class Base64ImageRequest(BaseModel):
    """Base64 이미지 요청 모델"""
    image: str  # Base64 encoded image
    binary: bool = False  # 이진 분류 모드


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
        from hesitation.inference import detect_hesitation
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
        from hesitation.inference import get_detector
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
    from hesitation.config import MODEL_PATH, SCALER_PATH
    
    model_exists = MODEL_PATH.exists()
    scaler_exists = SCALER_PATH.exists()
    
    return {
        "model_available": model_exists and scaler_exists,
        "model_path": str(MODEL_PATH),
        "message": "Model ready" if (model_exists and scaler_exists) else "Model not trained yet"
    }


# STT, LLM 등 AI 기능은 추후 구현
# @app.post("/api/stt")
# @app.post("/api/llm")
# @app.post("/api/recommend")

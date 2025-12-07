from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="AI Kiosk - AI Server",
    description="STT, LLM, AI 추론을 담당하는 FastAPI 서버",
    version="0.1.0"
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
            <p class="tech">Python + FastAPI + OpenAI</p>
            <p style="margin-top: 2rem; font-size: 0.9rem; opacity: 0.7">FusionCrew © 2024</p>
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


# STT, LLM 등 AI 기능은 추후 구현
# @app.post("/api/stt")
# @app.post("/api/llm")
# @app.post("/api/recommend")

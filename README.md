# AI Kiosk - AI Server

FastAPI 기반 AI 추론 서버 (STT, LLM)

## 기술 스택
- Python 3.11+
- FastAPI
- OpenAI API
- Uvicorn

## 설치 방법

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 실행 방법

```bash
# 개발 모드 (자동 리로드)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 프로덕션
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API 엔드포인트

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Hello World 페이지 |
| GET | `/health` | 헬스 체크 |
| GET | `/api/ping` | API 연결 테스트 |
| GET | `/docs` | Swagger UI (자동 생성) |

## 환경 변수

`.env` 파일 생성:
```
OPENAI_API_KEY=your-api-key-here
```

## 포트
- 기본: `8000`

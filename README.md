# 🤖 AI Kiosk - AI Server

FastAPI 기반 AI 추론 서버입니다. STT(음성 인식), LLM(대규모 언어 모델) 등 AI 기능을 담당합니다.

---

## 📋 목차
- [기술 스택](#-기술-스택)
- [폴더 구조](#-폴더-구조)
- [사전 준비](#-사전-준비)
- [설치 및 실행](#-설치-및-실행)
- [API 엔드포인트](#-api-엔드포인트)
- [환경 설정](#-환경-설정)

---

## 🛠️ 기술 스택

| 구분 | 기술 | 버전 |
|------|-----|------|
| **언어** | Python | 3.11+ |
| **프레임워크** | FastAPI | 0.104.0+ |
| **ASGI 서버** | Uvicorn | 0.24.0+ |
| **AI** | OpenAI API | 1.3.0+ |
| **유효성 검사** | Pydantic | 2.5.0+ |

---

## 📁 폴더 구조

```
AI/
├── main.py              # FastAPI 애플리케이션 진입점
├── requirements.txt     # Python 패키지 의존성 목록
├── .gitignore          # Git 무시 파일 설정
└── .env                # 환경 변수 (직접 생성 필요)
```

---

## ✅ 사전 준비

### 필수 설치
- **Python 3.11** 이상
  ```bash
  # 버전 확인
  python3 --version
  # 출력 예시: Python 3.11.x
  ```

- **pip** (Python 패키지 관리자)
  ```bash
  pip3 --version
  ```

---

## 🚀 설치 및 실행

### 1. 가상환경 생성 및 활성화
```bash
# 가상환경 생성
python3 -m venv venv

# 활성화 (macOS/Linux)
source venv/bin/activate

# 활성화 (Windows)
venv\Scripts\activate
```

> 💡 가상환경이 활성화되면 터미널 앞에 `(venv)`가 표시됩니다.

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정
```bash
# .env 파일 생성
touch .env
```

`.env` 파일에 다음 내용 추가:
```env
OPENAI_API_KEY=your-openai-api-key-here
```

### 4. 개발 모드 실행 (Hot Reload)
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. 프로덕션 모드 실행
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🔗 API 엔드포인트

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Hello World 페이지 |
| GET | `/health` | 헬스 체크 |
| GET | `/api/ping` | API 연결 테스트 |
| GET | `/docs` | Swagger UI (API 문서 - 자동 생성) |
| GET | `/redoc` | ReDoc (API 문서 - 대체 UI) |

### 서버 기본 포트
- **포트**: `8000`

### 접속 확인
서버 실행 후 브라우저에서 확인:
```
http://localhost:8000/         # Hello World 페이지
http://localhost:8000/docs     # Swagger UI (API 테스트 가능)
http://localhost:8000/health   # 헬스 체크
```

---

## ⚙️ 환경 설정

### 환경 변수 (.env)
```env
# OpenAI API 키 (필수)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

> ⚠️ **주의**: `.env` 파일은 `.gitignore`에 포함되어 있어 Git에 커밋되지 않습니다. 각 개발자가 직접 생성해야 합니다.

### requirements.txt
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
openai>=1.3.0
python-dotenv>=1.0.0
pydantic>=2.5.0
```

---

## 🐛 문제 해결

### "Module not found" 에러
```bash
# 가상환경 활성화 확인
source venv/bin/activate

# 의존성 재설치
pip install -r requirements.txt
```

### 포트 충돌
```bash
# 다른 포트로 실행
uvicorn main:app --reload --port 8001
```

### OpenAI API 에러
1. `.env` 파일에 올바른 API 키가 있는지 확인
2. API 키가 활성 상태인지 OpenAI 대시보드에서 확인

---

## 🔮 추후 구현 예정 기능

- `POST /api/stt` - 음성을 텍스트로 변환
- `POST /api/llm` - 자연어 질의 처리
- `POST /api/recommend` - AI 추천

---

## 👥 팀 정보

**FusionCrew** © 2025~2026


"""
Gemini API Client for Sign Language Translation
Using Google's Generative AI SDK
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class GeminiClient:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        Gemini 클라이언트 초기화
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(model_name)
        self.chat = self.model.start_chat(history=[])
        
        # 시스템 프롬프트 설정 (페르소나 부여)
        self.system_prompt = """
        당신은 수화 번역 전문가이자 친절한 AI 점원입니다.
        사용자가 수화로 표현한 단어들이 나열되면, 이를 자연스러운 한국어 문장으로 번역하고 적절한 대답을 해주세요.
        
        예시:
        입력: "나 배고파 밥 주세요"
        출력: "배가 고프시군요. 식사 메뉴를 보여드릴까요?"
        
        입력: "화장실 어디"
        출력: "화장실을 찾으시는군요. 오른쪽 복도 끝에 있습니다."
        
        입력: "물"
        출력: "물 한 잔 드릴까요?"
        """

    def generate_response(self, sign_text: str) -> str:
        """
        동작:
        1. 수화 단어 리스트(텍스트)를 입력받음
        2. 자연스러운 문장으로 변환 및 응답 생성
        """
        prompt = f"{self.system_prompt}\n\n입력: \"{sign_text}\"\n출력:"
        
        try:
            response = self.chat.send_message(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[ERROR] Gemini API Error: {e}")
            return "죄송합니다. 잠시 후 다시 시도해주세요."

if __name__ == "__main__":
    # 테스트 코드
    try:
        client = GeminiClient()
        print("Gemini Client Initialized.")
        
        test_input = "안녕 만나다 반갑다"
        print(f"Test Input: {test_input}")
        response = client.generate_response(test_input)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")

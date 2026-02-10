
import sys
import os
import argparse

# AI 패키지 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
# AI 폴더의 상위 폴더(project)를 path에 추가
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from AI.signLanguage.preprocess import main

if __name__ == "__main__":
    main()

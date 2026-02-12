
import sys
import os
import argparse

# 현재 스크립트의 상위 폴더(project)를 파이썬 경로에 추가
# c:\Users\User\project\AI\run_inference.py -> c:\Users\User\project 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

from AI.signLanguage.inference import main

if __name__ == "__main__":
    main()


import sys
import os
import argparse

# AI 패키지 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
# AI 폴더의 상위 폴더(project)를 path에 추가
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from AI.signLanguage.train import train_sign_language_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="date/sign_language_processed")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train_sign_language_model(args.data_dir, args.epochs)

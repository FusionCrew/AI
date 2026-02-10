# -*- coding: utf-8 -*-
"""
망설임 감지 모델 테스트 스크립트
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from pathlib import Path

print("=== 망설임 감지 모델 테스트 ===")
print()

# 1. 경로 확인
from hesitationLearning.config import MODEL_PATH, SCALER_PATH, DAISEE_DIR
print(f"모델 경로: {MODEL_PATH}")
print(f"모델 존재: {MODEL_PATH.exists()}")
print(f"스케일러 존재: {SCALER_PATH.exists()}")
print(f"DAiSEE 경로: {DAISEE_DIR}")
print(f"DAiSEE 존재: {DAISEE_DIR.exists()}")
print()

# 2. 테스트 라벨 로드
test_labels_path = DAISEE_DIR / "Labels" / "TestLabels.csv"
print(f"라벨 파일: {test_labels_path}")
print(f"라벨 존재: {test_labels_path.exists()}")

if not test_labels_path.exists():
    print("[ERROR] 테스트 라벨 파일 없음")
    sys.exit(1)

df = pd.read_csv(test_labels_path)
print(f"테스트 라벨 총 샘플: {len(df)}")
print(f"Confusion 분포:")
print(df['Confusion'].value_counts().sort_index().to_string())
print()

# 3. 모델 로드
print("모델 로드 시도...")
try:
    from hesitationLearning.inference import HesitationDetector
    detector = HesitationDetector(binary=True)
    print("[OK] 모델 로드 성공")
except Exception as e:
    print(f"[FAIL] 모델 로드 실패: {e}")
    print("\n[INFO] 모델이 없습니다. 먼저 다음 명령어로 학습을 진행해주세요:")
    print("  1. 특징 추출 (시간 소요됨):")
    print("     python -m hesitationLearning.run_feature_extraction")
    print("  2. 모델 학습:")
    print("     python -m hesitationLearning.run_training --binary")
    sys.exit(1)

# 4. 테스트 비디오 5개로 추론
print()
print("=== 추론 테스트 (5개 샘플) ===")
print()

test_count = 0
results = []

for _, row in df.iterrows():
    if test_count >= 5:
        break
    
    clip_id = row["ClipID"].strip()
    true_confusion = row["Confusion"]
    
    clip_name = clip_id.replace(".avi", "")
    user_id = clip_name[:6]
    video_path = DAISEE_DIR / "DataSet" / "Test" / user_id / clip_name / clip_id
    
    if not video_path.exists():
        continue
    
    print(f"[{test_count+1}] {clip_id}")
    print(f"    실제 Confusion: {true_confusion}")
    
    try:
        result = detector.detect_from_video(str(video_path))
        print(f"    예측: {result['label']} (level={result['hesitation_level']}, conf={result['confidence']:.3f})")
        if result.get('probabilities'):
            print(f"    확률: {[round(p, 3) for p in result['probabilities']]}")
        results.append({
            'true': true_confusion,
            'pred': result['hesitation_level'],
            'confidence': result['confidence'],
            'label': result['label']
        })
    except Exception as e:
        print(f"    [ERROR] {e}")
    
    test_count += 1
    print()

# 5. 결과 요약
if results:
    print("=== 결과 요약 ===")
    correct = 0
    for r in results:
        true_label = "hesitating" if r['true'] >= 2 else "not_hesitating"
        pred_label = "hesitating" if r['pred'] == 1 else "not_hesitating"
        match = "O" if true_label == pred_label else "X"
        if match == "O":
            correct += 1
        print(f"  [{match}] 실제: {true_label}({r['true']}) -> 예측: {pred_label}({r['pred']}) conf={r['confidence']:.3f}")
    print(f"\n  정확도: {correct}/{len(results)} ({correct/len(results)*100:.0f}%)")

if hasattr(detector, 'close'):
    detector.close()
print("\n테스트 완료.")

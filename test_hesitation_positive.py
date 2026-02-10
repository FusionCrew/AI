# -*- coding: utf-8 -*-
"""망설임 양성 샘플 테스트"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from pathlib import Path
from hesitationLearning.config import DAISEE_DIR
from hesitationLearning.inference import HesitationDetector

df = pd.read_csv(DAISEE_DIR / 'Labels' / 'TestLabels.csv')
hesitating = df[df['Confusion'] >= 2]
print(f'망설임 샘플 수: {len(hesitating)}')

detector = HesitationDetector(binary=True)
correct = 0
total = 0

for _, row in hesitating.head(5).iterrows():
    clip_id = row['ClipID'].strip()
    clip_name = clip_id.replace('.avi', '')
    user_id = clip_name[:6]
    video_path = DAISEE_DIR / 'DataSet' / 'Test' / user_id / clip_name / clip_id
    if not video_path.exists():
        continue
    total += 1
    result = detector.detect_from_video(str(video_path))
    pred = 'hesitating' if result['hesitation_level'] == 1 else 'not_hesitating'
    is_match = result['hesitation_level'] == 1
    tag = 'O' if is_match else 'X'
    if is_match:
        correct += 1
    probs = result.get('probabilities', [])
    prob_str = str([round(p, 3) for p in probs])
    print(f'[{tag}] {clip_id} | true: hesitating({row["Confusion"]}) -> pred: {pred} (conf={result["confidence"]:.3f}, prob={prob_str})')

print(f'\n망설임 샘플 정확도: {correct}/{total}')
detector.close()

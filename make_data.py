import os
import pickle
import numpy as np
import pandas as pd
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --- 기본 설정 ---
MAP_COL = 'waferMap'
LABEL_COL = 'label'
TARGET_SIZE = (64, 64)
SAVE_DIR = './data/processed_data/'
os.makedirs(SAVE_DIR, exist_ok=True)

print("🔄 데이터셋 로드 시작...")
df = pd.read_pickle("data/wafer_filtered.pkl")

# 데이터 분할 (기존과 동일하게 유지)
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df[LABEL_COL], random_state=42)

def resize_wafer(wafer, size=TARGET_SIZE):
    return resize(wafer, size, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

def normalize_wafer(wafer):
    return wafer / 2.0

# --- v6 재가공 및 가중치 계산 ---
print("⚙️ [v6 재가공] 클래스 가중치용 데이터셋 생성 중...")

# v2와 동일한 방식으로 v6 생성 (3.8GB 용량 확보)
train_v6 = train_df.copy()
train_v6[MAP_COL] = train_v6[MAP_COL].apply(lambda x: normalize_wafer(resize_wafer(x)))
train_v6.to_pickle(os.path.join(SAVE_DIR, 'train_v6.pkl'))

print("⚖️ [손실함수 가중치] Balanced Class Weight 재계산 중...")
y_train = train_v6[LABEL_COL].values
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, weights))

with open(os.path.join(SAVE_DIR, 'class_weights.pkl'), 'wb') as f:
    pickle.dump(class_weights_dict, f)

print("✅ [완료] train_v6가 3.8GB로 정상 복구되었으며 가중치 파일도 갱신되었습니다!")

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ==========================================
# ⚙️ [실험 인자 및 하이퍼파라미터 설정]
# ==========================================
DATA_VERSION = sys.argv[1] if len(sys.argv) > 1 else 'train_v2'

BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = './results/'
os.makedirs(SAVE_DIR, exist_ok=True)

# 보고서용 로그 파일 생성
log_file_path = os.path.join(SAVE_DIR, f"report_{DATA_VERSION}.txt")
log_file = open(log_file_path, "w", encoding="utf-8")

def log_print(message):
    print(message)
    log_file.write(message + "\n")

log_print("="*60)
log_print(f"📋 [실험 리포트] ResNet18 + 전처리 버전: {DATA_VERSION}")
log_print(f"💻 연산 장치(Device): {DEVICE}")
log_print("="*60)

# ==========================================
# 📦 [1. 데이터셋 및 데이터로더 구축]
# ==========================================
class WaferDataset(Dataset):
    def __init__(self, df, label_mapping=None):
        self.X = torch.tensor(np.stack(df['waferMap'].values), dtype=torch.float32).unsqueeze(1)
        if label_mapping is None:
            unique_labels = sorted(df['label'].unique())
            self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_mapping = label_mapping
        self.y = torch.tensor(df['label'].map(self.label_mapping).values, dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_path = f'./data/processed_data/{DATA_VERSION}.pkl'
if not os.path.exists(train_path):
    log_print(f"❌ 에러: {train_path} 파일이 없습니다.")
    sys.exit()

train_df = pd.read_pickle(train_path)
val_df = pd.read_pickle('./data/processed_data/val.pkl')

train_dataset = WaferDataset(train_df)
val_dataset = WaferDataset(val_df, label_mapping=train_dataset.label_mapping)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#log_print(f"🔹 [데이터 셋 설명]")
#log_print(f"   - 학습 데이터(Train) 수: {len(train_dataset)}개")
#log_print(f"   - 검증 데이터(Validation) 수: {len(val_dataset)}개")
#log_print(f"   - 타겟 클래스 개수: {len(train_dataset.label_mapping)}개")
#log_print(f"   - 클래스 매핑 가이드: {train_dataset.label_mapping}")

# ==========================================
# 🧠 [2. 모델 정의 및 구조적 특징 추출]
# ==========================================
def get_resnet18(num_classes=8):
    model = resnet18(weights=None)  # 처음부터 학습 (Scratch)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

model = get_resnet18(num_classes=8)

# 모델 파라미터 수 계산 (보고서 필수 항목)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

#log_print(f"\n🔹 [사용 모델 설명 (ResNet18)]")
#log_print(f"   - 학습 가중치 방식: From Scratch (처음부터 학습)")
#log_print(f"   - 총 파라미터 수 (Total Params): {total_params:,}개")
#log_print(f"   - 학습 가능한 파라미터 수: {trainable_params:,}개")
#log_print(f"   - 입력 구조 변형: 3채널 컬러 ➡️ 1채널 단색(웨이퍼 격자) 지정")

# ==========================================
# ⚖️ [3. 손실 함수 및 옵티마이저]
# ==========================================
if DATA_VERSION == 'train_v6':
 #   log_print("   - 특이사항: Loss 함수에 클래스 균형 가중치(Class Weight) 주입 완료")
    with open('./data/processed_data/class_weights.pkl', 'rb') as f:
        weights_dict = pickle.load(f)
    ordered_weights = [weights_dict[label] for label in sorted(train_dataset.label_mapping.keys())]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(ordered_weights, dtype=torch.float32).to(DEVICE))
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 🔥 [4. 훈련 및 다양한 Metric 측정 루프]
# ==========================================
log_print("\n" + "="*40 + " [학습 로그 시작] " + "="*40)
best_val_f1 = 0.0
total_train_start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    
    # --- 훈련 세션 ---
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)

    # --- 검증 및 실행 시간(Inference Time) 측정 세션 ---
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    inference_start_time = time.time()  # 추론 시간 측정 시작
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    inference_end_time = time.time()  # 추론 시간 측정 종료
    
    # 지표 계산
    epoch_duration = time.time() - epoch_start_time
    total_inf_time = inference_end_time - inference_start_time
    inf_time_per_batch = (total_inf_time / len(val_loader)) * 1000  # ms 단위
    
    val_loss /= len(val_loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='macro')

# 수정할 라인:
    log_print(f"Epoch [{epoch:02d}/{EPOCHS}] | Loss: {train_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Time: {epoch_duration/60:.2f} min")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        model_save_path = os.path.join(SAVE_DIR, f'best_resnet18_{DATA_VERSION}.pth')
        torch.save(model.state_dict(), model_save_path)

total_train_duration = time.time() - total_train_start_time
log_print("="*60)
log_print(f" 총 학습 소요 시간: {total_train_duration/60:.2f}분")
log_print(f" 최종 검증 최고 F1-Score: {best_val_f1:.4f}")
log_print("="*60)

log_file.close()

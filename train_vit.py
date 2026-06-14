import torch
import torch.nn as nn
import torch.optim as optim
import timm
import time
import argparse
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument('version', type=str, help='데이터 버전 (예: train_v2)')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_VERSION = args.version
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4 

class WaferDataset(Dataset):
    def __init__(self, data_path, label_mapping=None):
        self.data = pd.read_pickle(data_path)
        self.label_mapping = label_mapping if label_mapping else {label: i for i, label in enumerate(self.data['label'].unique())}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        wafer = self.data.iloc[idx]['waferMap']
        label = self.label_mapping[self.data.iloc[idx]['label']]
        return torch.tensor(wafer, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)

train_path = os.path.join('./data/processed_data/', f'{DATA_VERSION}.pkl')
val_path = './data/processed_data/val.pkl'

train_dataset = WaferDataset(train_path)
val_dataset = WaferDataset(val_path, label_mapping=train_dataset.label_mapping)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = timm.create_model(
    'vit_tiny_patch16_224', 
    pretrained=False, 
    num_classes=len(train_dataset.label_mapping), 
    in_chans=1
)

model.img_size = 64

model.patch_embed.img_size = (64, 64)
model.patch_embed.patch_size = (8, 8)
model.patch_embed.grid_size = (8, 8)
model.patch_embed.num_patches = 64
model.patch_embed.proj = nn.Conv2d(1, 192, kernel_size=(8, 8), stride=(8, 8))

model.pos_embed = nn.Parameter(torch.randn(1, 65, 192) * .02)

model = model.to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print(f" 학습 시작: 모델=ViT-Tiny, 데이터={DATA_VERSION}")
print("-" * 85)

total_start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    
    # Train
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 지표 계산
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    epoch_acc = 100 * (np.array(all_preds) == np.array(all_labels)).mean()
    epoch_loss = running_loss / len(train_loader)
    epoch_duration = (time.time() - epoch_start_time) / 60
    
    print(f"Epoch [{epoch:02d}/{EPOCHS}] | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | F1: {epoch_f1:.4f} | Time: {epoch_duration:.2f} min")

print("-" * 85)
print(f" 학습 완료! 총 소요 시간: {(time.time() - total_start_time)/60:.2f}분")

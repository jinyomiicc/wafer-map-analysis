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

DATA_VERSION = sys.argv[1] if len(sys.argv) > 1 else 'train_v2'

BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = './results/'
os.makedirs(SAVE_DIR, exist_ok=True)

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

train_df = pd.read_pickle(train_path)
val_df = pd.read_pickle('./data/processed_data/val.pkl')

train_dataset = WaferDataset(train_df)
val_dataset = WaferDataset(val_df, label_mapping=train_dataset.label_mapping)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

def get_resnet18(num_classes=8):
    model = resnet18(weights=None) 
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

model = get_resnet18(num_classes=8)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

if DATA_VERSION == 'train_v6':
    with open('./data/processed_data/class_weights.pkl', 'rb') as f:
        weights_dict = pickle.load(f)
    ordered_weights = [weights_dict[label] for label in sorted(train_dataset.label_mapping.keys())]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(ordered_weights, dtype=torch.float32).to(DEVICE))
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

log_print("\n" + "="*40 + " [학습 로그 시작] " + "="*40)
best_val_f1 = 0.0
total_train_start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    
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

    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    inference_start_time = time.time()  
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    inference_end_time = time.time()  
    
    epoch_duration = time.time() - epoch_start_time
    total_inf_time = inference_end_time - inference_start_time
    inf_time_per_batch = (total_inf_time / len(val_loader)) * 1000  # ms 단위
    
    val_loss /= len(val_loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='macro')

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

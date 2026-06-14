# 🚀 Wafer-map-analysis

반도체 제조 공정에서 수집된 웨이퍼 맵(Wafer Map)의 결함 패턴을 딥러닝 모델을 통해 자동으로 분류하는 프로젝트

---

## 📋 프로젝트 개요
* **목표**: WM-811K 데이터셋을 활용한 웨이퍼 맵 결함 유형(10종) 분류
* **기간**: 2026.05 - 2026.06
* **개발 환경**: AWS EC2 GPU Instance

## ⚙️ 학습 하이퍼파라미터 상세
모델 학습에 사용된 주요 설정값은 다음과 같습니다.

| 모델 | Batch Size | Learning Rate | Optimizer | epoch | 
| :--- | :---: | :---: | :---: | :---: |
| ResNet18 | 128 | 0.001 | AdamW | 15 |
| EfficientNet-B0 | 32 | 0.001 | AdamW | 15 |
| ViT-Tiny | 32 | 0.0001 | AdamW | 15 |

## ⚙️ 전처리 단계 (Preprocessing)

| 단계 | 설명 |
| :--- | :--- |
| **P1** | 불필요 클래스('Unknown', 'Near-full') 제거 |
| **P2** | 이미지 리사이징(64x64) 및 정규화(Normalization) |
| **P3** | Median Filter를 통한 노이즈 제거 |
| **P4** | 데이터 증강(Augmentation) 수행 |
| **P5** | 손실 함수(Loss Function)에 Balanced Class Weight 적용 |

## 📊 전처리 성능 비교
다양한 전처리 조합을 실험한 결과, 클래스 불균형을 고려한 P5 단계에서 가장 우수한 F1-Score를 기록하였습니다.

| 단계 | 기법 요약 | Accuracy | F1-Score |
| :--- | :--- | :---: | :---: |
| P1 | 클래스 제거 | 
| P2 | 단순 Resize/정규화 | 0.9720 | 0.8506 | 최고성능
| P3 | Median Filter 적용 | 0.9524 | 0.7133 |
| P4 | Augmentation 적용 | 0.9690 | 0.8331 |
| P5 | Class Weight 적용 | 0.9590 | 0.8162 |

## 🧠 모델 성능 비교
모델 성능은 **F1-Score**를 최우선 지표로 하여 비교·분석하였습니다.

| 모델 | Accuracy | F1-Score | Inference Time (min) |
| :--- | :---: | :---: | :---: |
| **ResNet18** | **0.9717** | **0.8537** | **11.80** |
| **EfficientNet-B0** | 0.9721| 0.8437 | 35.20 |
| **ViT-Tiny** | 0.9663 | 0.7725 | 37.24 |

## 🏗 프로젝트 구조
```text
wafer-map-analysis/
├── data/               # 원본 및 전처리 데이터
├── models/             # 모델 아키텍처 정의
├── notebooks/          # 실험용 Jupyter Notebook
├── train.py            # 메인 학습 스크립트
├── make_data.py        # 데이터 전처리 스크립트
└── README.md

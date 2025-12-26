---
layout: post
title: "[CLIP] Zero-shot Prediction & Linear Probe "
date: 2025-12-02 14:00:00 +0900
categories: Paper_Review
tags: [
    Zero_shot,Linear_Probe, computer_vision, Vision_Language_Models, paper_review , VLM, Zero-shot, NLP
]
---



# CLIP_Zeroshot_Prediction & Linear Probing Implementation

## Learning Transferable Visual Models From Natural Language Supervision(2021)

![](/assets/paper_review/VLM_CLIP_code/1.png)

### **CLIP (Contrastive Leanguage-Image Pre-training)**

- 실제로 논문에서 강조하고 있는 CLIP 모델의 zero-shot prediction을 실제로 확인해보고 싶었다.
- 또한, Linear Probing을 통해서도 모델의 성능을 확인해보고 싶었다.

**CLIP 모델**

- Feature Extraction (특징 추출)
    - 이미지와 텍스트를 각각의 인코더(이미지 인코더, 텍스트 인코더)에 통과시켜 벡터로 생성
- Joint Embedding Space (하나의 임베딩 공간을 활용)
    - 특징 추출 단계에서 추출한 이미지, 텍스트 각각의 벡터를 하나의 임베딩 공간에 위치시킴
- Contrastive Learning (대조 학습)
    - Positive Pair → 유사도 높게, 가깝게
    - Negative Pair → 유사도 낮게, 멀게

### **Zero-shot Prediction?**

- Supervised Learning (기존의 방식)
    - 다른 task에 모델을 적용하기 위해서는 Fine-Tuning이 필요함
- Zero-shot (CLIP)
    - 사용자의 의도(프롬프트)에 맞게 특정 task에 따로 Fine-Tuning 같은 과정 없이 예측하는 능력
    - 특정 데이터 셋에 과적합되지 않으며, 높은 일반화(Generalization) 성능과, 강건함(Robustness)을 가짐
    
    **→ 사용자의 의도에 맞게 파라미터 업데이트(Gradient Update)가 필요 없음**
    
- 그만큼 모델이 광범위하게 강건하다..는 뜻이 아닐까 (Robustness)
    - 엄청난 크기의 데이터셋을 통해 학습을 했기 때문에 가능

**활용한 데이터셋**

- CIFAR100

Model & Data Preparation → Prompt Engineering → Feature Coding → Similarity Calculation

**Prompt Engineering**

→ 단순히 단어만을 주는 것이 아니라, a photo of a snake 와 같은 문장 (prompt)

- CLIP은 문장 단위로 학습함

**Feature Encoding**

→ model.encode_image(), model.encode_text() 를 통해 이미지, 텍스트 각각 벡터로 변환

**Similarity Calculation (Zero-shot)**

→ Cosine Similarity 

```python
import os
import clip
import torch
from torchvision.datasets import CIFAR100

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
```

### Linear Probe?

- 이해도를 평가하는 척도
- 모델의 파라미터를 모두 얼리고(Freeze), 오직 마지막 분류기 (Linear Head)만을 학습시켰을 때의 성능을 확인

**→ 단순한 선형 분류기만을 추가해도 높은 성능을 가진다면, 사전 학습된 모델이 이미 데이터를 선형적으로 구분** 

**→ OpenAI의 CLIP (ViT-B/32) 모델을 Feature Extractor로 사용하고, scikit-learn 의 Logistic Regression을 사용하여 CIFAR-100 데이터셋에 대한 Linear Probing을 구현**

```python
import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Load the dataset
root = os.path.expanduser("~/.cache")
train = CIFAR100(root, download=True, train=True, transform=preprocess)
test = CIFAR100(root, download=True, train=False, transform=preprocess)

def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")

accuracy : 78.31%
```

### Results

- 실험 결고, CLIP의 Image Encoder를 재학습 (Fine-Tuning) 하지 않고 단순한 선형 분류기만 붙였음에도 불구하고 준수한 정확도를 보여줌
- CLIP은 대규모 데이터셋으로 학습되며, 이미지 내의 시각적 특징을 매우 강력하고 일반화된 형태(Robust & Generalizable)로 학습했음을 확인

→ Fine-tuning 없이도, CLIP의 Feature를 활용하면 다양한 Downstream Task를 효율적으로 수행 가능

### References

Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", 2021
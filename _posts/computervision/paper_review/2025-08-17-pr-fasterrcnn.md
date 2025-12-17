---
layout: post
title: "PR [Faster R-CNN:Towards Real-Time Object Detection with Region Proposal Networks] "
date: 2025-8-10 14:00:00 +0900
categories: Paper_Review
tags: [
    computer_vision, Object_Detection, paper_review , CNN, R_CNN
]
---



# Faster R-CNN: Towards Real-Time Object
Detection with Region Proposal Networks

![](/assets/paper_review/faster_r_cnn/1.png)


**💡 Faster R-CNN : Towards Real-Time Object Detection with Region Proposal Networks (2016, Jan, 06)**

저자 : SHaoqing Ren, Kaiming He, Ross Girshick, Jian Sun


## Abstract

- 그 당시의 객체 탐지 기술 : Region Proposal Algorithms를 기반으로 물체의 위치를 예측하곤 했었다.
    - **Region Proposal (영역 제안**) 이란?
        - 객체가 있을 만한 후보 영역들을 찾아주는 과정
        - 이미지의 픽셀을 모두 확인하는 것이 아니라, 객체가 있을만한 후보 영역을 먼저 제안하고, 그 부분만 CNN이 인식하게 만드는 구조
        - 후보 영역마다 CNN을 각각 따로 적용
        - 연산량을 줄일 수 있다.
- SPPnet, Fast R-CNN
    - CNN 연산을 공통 feature map으로 공유하게 해서 속도를 향상
        - feature map
            - CNN이 추출한 시각적 정보의 요약
            - feature map 에 ROI를 적용하여 영역 제안
    - CNN 연산을 많이 진행 X

해당 논문에서는 **Region Proposal Network (RPN)** 을 소개하고자 한다.

- full - image convolutional features를 탐지 네트워크와 공유하며, cost-free regional proposal 이 가능하다.
    - full-image convolutional features
        - 전체 이미지에 대해 한 번 추출한 convolutional feature map
    - 추가적인 계산 비용 없이 영역 제안을 얻을 수 있다.
- 이미지 전체에 대해 CNN을 한번 적용하여 feature map을 산출하고, RPN을 적용하여 영역제
- **Region Proposal Network (RPN)**
    - Fully convolutional network
    - 각 위치마다 객체의 경계와 객체일 확률을 동시에 예측
    - **end-to-end 방식으로 학습**
        - 입력부터 최종 출력까지 전체 네트워크를 하나의 그래디언트  흐름으로 연결하여 오차를 기반으로 모든 파라미터를 동시에 학습
    
    RPN + Fast R-CNN 이렇게 2가지를 하나의 네트워크로 합친다.
    
    ⇒ convolutional features 공유!
    
    - Convolutional Features 란?
        - CNN이 입력 이미지에 합성곱 연산을 적용하여 추출한 출력값
    - “attentions”에 관해서는 RPN 구성요소들이 통합된 네트워크 ( RPN + Fast R-CNN )에 어디에 집중할지 알려준다.
    - Frame rate : 5fps (GPU)

| **단계** | **Fast R-CNN** | **Faster R-CNN** |
| --- | --- | --- |
| **1. CNN (feature map 생성)** | **✅** | **✅** |
| **2. Region Proposal** | **❌ 외부 Selective Search** | **✅ CNN 기반 RPN** |
| **3. RoI Pooling** | **✅ 사용** | **✅ 사용 (동일)** |
| **4. Classification + bbox regression** | **✅** | **✅** |
| **5. 학습 방식** | **❗️부분만 end-to-end** | **✅ 전체 end-to-end** |

# 1 Introduction

객체 탐지 (Object Detection)의 기본

**⇒ Region Proposal Algorithm(영역제안) + CNN**

기존의 Fast R-CNN

- Region Proposal (영역제안) 시간을 제외한다면, 아주 깊은 신경망을 사용해도 거의 실시간 속도를 이룬다.
- CNN의 백본(BackBone)을 다르게 사용할 수 있다. (논문에선 VGG-16)

**객체 탐지 시스템 ⇒ 영역 제안 과정이 테스트 단계에서 속도를 가장 느리게 만드는 병목이 되었다.**

### 기존 영역 제안 기법

- 빠르지만 단순한 특징에 의존, 학습 불가능한 계산 효율 위주의 방식 (사람이 정해놓은 규칙에만 의존)

**Selective Search**

- 많이 쓰이는 영역 제안 알고리즘
- Greedy Merge (비슷해보이는 superpixel끼리 하나씩 차례로 계속 합침)
    - 픽셀들을 하나씩 합치면서 영역을 생성
    - superpixel
        - 작고 비슷한 색/텍스처 덩어리로 분할
- 사람이 설계한 저수준 특징들 (색상, 질감 등)에 기반한다.

⇒ 효율적인 객체 탐지 신경망에 비하면, Selective Search 는 CPU 구현환경에서 이미지 당 2초가 걸리는 엄청 느린 속도를 가지고 있다.

**Edge Boxes**

- 현재로써 최고의 균형을 보여줌 (영역 제안의 질과 속도)
    - 이미지당 0.2초
- 그래도 객체 탐지 신경망의 전체 시간만큼의 시간이 걸린다.

객체 탐지 네트워크는 GPU 구현인데, Region Proposal은 CPU 구현환경이니까 비교가 불공평하지 않나?

 **Region Proposal 을 GPU 환경에서도 작동될 수 있도록 구현**

⇒ GPU 환경으로 옮기는 것은 효과적일 순 있지만, 후속 탐지 네트워크를 고려하지 않기 때문에 계산을 공유하는 기회를 놓치게 된다.

RPN과 Detection Network가 같이 계산을 공유하는 방식이 더 좋다.

따라서 Faster R-CNN : 전체 이미지에 대해 CNN 후, 얻은 feature map을 RPN과 Detection Network가 계산을 공유

**따라서 이 논문에서는, 깊은 CNN를 사용해 영역 제안을 수행하는 방식이 효율적인 해결을 제공하고, cost-free하게 가능하다는 점을 보여준다.**

**CNN 으로 Region Proposal을 수행 ⇒ detection 연산을 그대로 활용 ⇒ 효율적 (cost-free)**

가장 활용성이 높은 객체 탐지 네트워크들과 계산을 공유하는 RPN을 도입한다.

⇒ 영역 제안의 비용은 매우 작다

또한, Fast R-CNN과 같이, 지역 탐지를 할 때 사용되는 feature map은 지역 제안을 할때에도 사용이 가능하다는 것을 확인하였다.

⇒ 이전의 feature map : ‘탐지’에만 사용되었으며, 영역 제안은 다른 regional proposal algorithm을 사용 (Selective Search) 하지만, feature map이 regional proposal을 할때에도 사용이 가능하다는 것을 알아내었다.

논문에서 제안하는 “RPN”

- 해당 합성곱적 특성 위에, 해당 논문에서는 합성곱 층을 여러개 쌓은 RPN을 제안한다.
    - 여러 개의 합성곱 층은 각각의 위치에서 regular grid 위에 region bounds와 objectness scores를 동시에 예측한다.
    - 해당 RPN은 Fully Convolutional Network (FCN)이다.
        - Fully Convolutional Network ⇒ 합성곱 연산으로만 이루어진 계층을 의미한다.
        - 특히 탐지 제안을 하는 목표에 대해 end-to-end 방식으로 학습가능하다.
- 다양한 크기와 종횡비에 대해 예측이 가능하도록 효율적으로 디자인되었다.
    
    ![](/assets/paper_review/faster_r_cnn/2.png)
    
- 기존의 객체 탐지 방식 (a)(b)
    - 다양한 크기의 객체 탐지를 위해 이미지와 필터를 다양한 크기로 재조정하여 처리했다.
        - 각각의 크기에 대해 CNN을 일일이 돌려야했음
- **논문에서 제안하는 “’anchor’ boxes”** : 다양한 크기와 종횡비의 객체들을 탐지 가능하다.
    - anchor boxes : CNN을 통해 산출된 feature map에 여러 크기의 box들을 깔아두고, 객체의 위치를 예측한다.
    - 다양한 크기에 대해 이미지, 필터들을 계산할 필요 없이 깔아둔 anchor box를 통해 객체의 위치를 예측한다.
        - 박스의 크기가 얼마나 조정되어야하는지?
        - 박스 안에 객체가 있을 확률?
    - 해당 모델은 단일 크기로 학습되고 테스트를 하였을때 성능이 좋고, 속도도 빠르다.

RPN 과 Fast R-CNN 객체 탐지 신경망과 합치는 방식

- RPN과 Fast R-CNN을 통합하여 end-to-end 방식으로 학습하기 위해서는 지역 제안(RPN)과 객체 탐지에 대해서 번갈아가며 fine-tuning하는 방식을 선택
    - 객체 탐지에 대해 fine-tuning을 할때에는 지역 제안 (Regional Proposal)을 고정한채로 진행

PASCAL VOC Detection Benchmarks를 통해 위의 방식을 평가

- **RPN + Fast R-CNN** 이 **Selective Search + Fast R-CNN**보다 더 좋은 성능을 나타냈다.
- 또한, Fast R-CNN과 RPN을 통합한 방식은 기존의 Selective Research를 통해 발생하는 계산량을 피할 수 있기 때문에, 시간적인 측면에서도 우수하다.
- CNN의 백본으로 굉장히 깊은 신경망 모델을 사용해도, 해당 모델은 frame rate : 5fps 로 속도, 정확도 측면에서 실용적인 객체 탐지 시스템이다.
- 3D 객체 탐지, 부분 객체 탐지, 객체 탐지 세분화, 이미지 캡션에서도 사용된다.

**⇒ RPN + Fast R-CNN은 효율적일 뿐만 아니라, 실용적인 방식이며, 객체 탐지의 정확도를 높이는 효율적인 방식이다.**

# 2 Related Work

### Object Proposals

- super-pixels를 그룹화하는 방식을 기반으로 한다.
    - Selective Research
- Sliding Windows를 기반으로 하는 방식
    - EdgeBoxes
- Object Proposal 방식들은 탐지 (Detectors)와 독립적인 외부 모듈로 사용되었다.
    
    ⇒ 탐지기와 계산 공유 X
    

### **Deep Networks for Object Detection**

- R-CNN 방식은 CNN들을 제안된 지역을 객체 or 배경으로 분류하기 위해  end-to-end 방식으로 학습시킨다.
- R-CNN은 주로 분류 역할을 수행하며, 객체 경계를 예측하지는 않는다. 
(박스 예측을 통해 재조정하는 것은 제외)
    - 분류 역할 정확도 ⇒ 지역 제안 (Region Proposal) 모듈에 따라 달라진다.
- 여러 논문들에서 딥러닝을 이용하여 객체의 bounding box를 예측하는 방식을 사용해왔다.
    - OverFeat 방식에서는 완전연결계층을 학습시켜 객체의 bounding box를 예측했다.
    - 이후 완전연결계층은 다중클래스 객체를 탐지하는 합성곱 계층으로 바뀌었다.
        - OverFeat 의 ‘single-box’를 확장
        
        ⇒ **Multibox 방식**은 네트워크의 마지막 완전연결계층으로부터 다수의 class-agnostic boxes를 동시에 예측하여 Regional proposal을 생성한다.
        
        - 이 class-agnostic boxes는 R-CNN의 제안에 사용된다.
        - MultiBox 네트워크는 단일 이미지 혹은 다수의 큰 이미지에 적용되며, 이는 논문의 합성곱으로만 이루어진 방식과 대비된다.
        - Proposal과 detection 네트워크 사이에 특징들을 공유하지 않는다.
            - 합성곱의 연산을 공유하는 것은 효율과 정확한 시각 인식을 위해 주목받고 있다.
    
    OverFeat 방식
    
    - 이미지를 여러 사이즈로 만든 후, 각각의 사이즈에 대해 모두 CNN을 적용하는 방식
    
    SPP 방식
    
    - 이미지의 Feature Map에 여러 사이즈의 anchor boxes를 두고 객체를 탐지한다.
    - 효율적으로 지역 기반 객체 탐지가 가능하다.
    
    Fast R-CNN
    
    - 공유된 합성곱적 특징에서 end-to-end 탐지 학습이 가능하며, 속도와 정확도가 모두 우수하다.
    
    # **3 Faster R-CNN**
    
    ![](/assets/paper_review/faster_r_cnn/3.png)
    
- Faster R-CNN은 2가지 모듈로 이루어져 있다.
    1. 깊은 합성곱 네트워크 → 지역을 제안
    2. Fast R-CNN detector → 제안된 지역을 사용
- 전체적인 시스템은 하나로 통합되어 있다.
- RPN 모듈이 Fast R-CNN에게 어디에 “Attention”을 둘지 알려준다.

### 3.1 Region Proposal Networks

- 이미지를 input으로 받으며, 출력으로는 직사각형의 객체/지역 제안을 가진다.
- 위와 같은 과정을 합성곱 계층으로만 구현하며, 해당 **모델의 목표**는 **계산(이미지→feature map)**을 Fast R-CNN과 공유하기 위해서이다.
    - 2개의 네트워크는 동일한 합성곱 계층을 공유한다.
        
        ![](/assets/paper_review/faster_r_cnn/4.png)
        

⇒ 해당 논문의 실험에서는 **Zelier and Fergus Model (공유 가능한 합성곱 계층 : 5개), Simonyan and Zisserman Model (공유 가능한 합성곱 계층 : 13개)**

Region Proposal

- 마지막 합성곱 계층의 출력인 Feature Map 위에 small network 를 slide 시킨다
- small network
    - feature map위에서 nxn 크기의 부분적 윈도우를 입력으로 가진다.
    - 각각의 슬라이딩 윈도우는 저차원 특징들로 매핑되며 (ZF 네트워크 - 256 차원, VGG - 512차원) 이후 ReLU 함수로 전달된다.

⇒ 특징들은 이후 **box-regression layer(reg)**와 **box-classification layer(cls)**로 전달된다.

- n=3으로 설정 (하나의 위치에 대해 3개의 scale을 적용)

![](/assets/paper_review/faster_r_cnn/5.png)

<aside>
💡

**CNN의 마지막 계층에서 나온 feature map**

 **→ sliding window (3x3)을 이동시키면서, 윈도우에 대응하는 feature들을 256차원으로 축소시킴** 

**→ cls, reg 계층으로 이동시킴** 

**→ 각 위치마다 anchor boxes들을 적용하여 객체 탐지** 

</aside>

### 3.1.1 Anchors

- Sliding Window 위치에서, 동시에 다수의 지역 제안을 하고, 그 개수는  k로 정의되어있다.
- reg layer (위치를 얼마나 조정할 것인지)
    - 4k의 출력 (x,y,w,h)
- cls layer (해당 객체가 전경인지 배경인지)
    - 2k의 출력 (object, not-object)
- 3개의 scales, 3개의 aspect ratios를 적용하여 총 9개의 anchors를 사용한다. (논문에서)
- 총 anchor의 개수
    - feature map이 WxH면, 총 anchor의 개수 = WxHxk이다.
        - **sliding window 는 feature map 각각의 위치에서 모두 적용하기 때문이다.**

**Translation-Invariant Anchors**

- 해당 논문의 중요한 부분은 translation invariant 이다.
    - 만약 객체의 위치가 조금 바뀌더라도 각 위치에 적용되는 anchor들과 객체를 탐지하는 함수들로 객체가 탐지되게끔 하는 것이다.
    - 만약 이미지 내에서 어떤 객체를 인식하였다면, 동일한 함수가 해당 객체를 다른 위치에서도 예측할 수 있어야한다.
        - Faster R-CNN은 이것이 가능하다! (Multibox는 K-means를 사용하기 때문에 불가능하다.)
    - Translation-Invariant 는 또한 모델의 크기를 줄인다.
    - Multibox의 경우, parameters의 개수가 6.1x10^6개 존재하며, 해당 논문의 모델의 경우엔 parameters의 개수가 (512 x (4+2) x 9)개이다. (VGG-16 모델을 CNN 백본으로 사용하였을때)
    
    ⇒ parameter들의 개수도 줄이며, 소규모 데이터셋에 대해서도 과적합의 위험을 낮출 수 있다.
    

Multi-Scale Anchors as Regression References

- Faster R-CNN의 방식에서는 다양한 이미지 크기에 대한 객체 탐지가 가능하며, 해당 방식에는 2가지가 있다.
1. image/feature pyramids를 사용 (이미지를 조작)
    - 이미지들이 다양한 크기로 재조정되며, 재조정된 사이즈에 대해 feature map 이 각각 적용되는 방식이다.
2. Sliding Windows를 Feature map에 대해 다양한 크기로 사용하는 방식
    - 서로 다른 종횡비를 가진 객체 탐지 모델들은 각각 서로 독립적으로 다양한 크기의 필터들로 훈련된다.
    - “pyramid of filters”라고 생각하자!
    
    ⇒ 해당 방식은 1번의 방식과 함께 사용되곤 한다.
    
- 위의 2가지 방식과 비교하여, anchor를 사용하는 방식은 효율적인 ‘Pyramids of anchors”의 구조를 가진다.

![](/assets/paper_review/faster_r_cnn/6.png)

- 다양한 크기의 anchors 사용하기 때문에, feature map과 image는 단 하나의 크기만을 가진다.
- 해당 방식으로 다양한 크기를 가진 객체들을 탐지할 수 있다.

### 3.1.2 Loss Function

- 학습하는 RPN을 위해서는, 이진분류를 위한 클래스 라벨을 부여한다. (객체인지? 아닌지?)
- 2가지의 anchor들을 위해 긍정적인 라벨을 부여한다.
    1. ground-truth box 와 가장 높은 영역 겹침 비율을 가진 anchor
    2. ground-truth box 와 영역이 겹치는 비율이 0.7 이상인 anchor
    
    ***ground-truth box? : 정답 박스, 정답 기준, 사람이 부여한 라벨***
    
    ***하나의 ground-truth box가 정답 라벨을 다수의 anchor에게 할당할 수 있다.***
    
    ⇒ 2번째 방식이 정답 샘플들을 결정하는데 효율적이지만, 해당 논문에서는 첫번째 방식을 사용한다. (가끔 드물게 2번째 방식이 정답 라벨을 탐지하지 못하는 경우를 위해)
    
- 만약 IoU (정답 라벨과 겹치는 비율)이 0.3보다 작으면, Negative label(배경)을 부여한다.
- Positive/Negative가 아닌 anchor들은 학습에 관여하지 않는다!

| 라벨 | 설명/사용 |
| --- | --- |
| Positive Label | Ground-Truth box와 겹치는 비율(IoU)가 0.7이 넘을때 (학습시 객체 라벨로 사용) |
| Negative Label | Ground-Truth box과 겹치는 비율 (IoU)가 0.3보다 작을때 (학습시 배경 라벨로 사용) |
| 0.3<IoU<0.7 | 학습시 사용 X |

위와 같은 방식으로 Fast R-CNN에서의 객체 탐지 (멀티 테스크) 손실을 최소화한다.

![](/assets/paper_review/faster_r_cnn/7.png)

- 위의 식에서 i는 미니배치에서의 anchor의 인덱스로 사용된다.
- p_i는 i번째 anchor가 객체로 예측될 확률을 의미한다.
- p_i* (ground-truth label)은 anchor가 positive(객체이면), 1, 객체가 아니면(배경이면) 0이된다.
- t_i 와 t^*_i 는 둘 다 **Anchor를 기준으로 파라미터화된 위치 보정값** 벡터이며, 하나는 모델 예측값, 다른 하나는 그 anchor에 연결된 정답 박스 값
    - 왜 파라미터화 하는가? → anchor를 기준으로 한 상대적 변화량으로 표현하기 때문!
- Classification_Loss : 객체/배경 2가지의 클래스의 log를 씌운 형태이다.
- Regression Loss : Robus Loss Function을 사용하여 학습 안정성을 높인다.
    - p*_i L_reg는 회귀 손실이 정답 anchor들을 위해서만 활성화되고, 배경 anchor들에 한해서는 활성화되지 않는다.
- cls, reg : cls 는 지금 현재 미니배치 크기(256)에 의해 정규화되었으며, reg는 anchor의 개수에 의해 정규화되었다.
    - cls : 객체 vs 배경 분류
    - reg : bbox 위치 보정
- balancing parameter (λ) 는 Classification term과 Regression term 값의 스케일이 다르기 때문에, 그냥 합치면 한 쪽이 너무 크게/작게 작용하여 학습이 한쪽으로 치우칠 수 있다. 이를 방지하기 위해 2개의 loss의 비중으로 조절하는 가중치이다.

⇒ 이와 같은 정규화 과정과 balancing parameter의 값은 간략화시킬 수 있다.

bounding box 회귀를 위해서는, 다음과 같은 파라미터화 과정을 거친다.

![](/assets/paper_review/faster_r_cnn/8.png)

→ x,y,높이, 너비 방향으로의 변화량을 비율로 나타내서 예측하는 방식이다.

위와 같은 bounding box regression 방식이 이전 RoI 기반 방법들과 어떻게 다른가?

- 이전의 RoI 기반 방식 ([1],[2] = R-CNN, Fast R-CNN)
    1. Region Proposal (Selective Search 등) → 크기 제각각인 RoI를 생성
    2. 각 RoI에서 feature pooling (RoI Pooling)
    3. 같은 Regression weight를 모든 RoI 크기에 대해 사용
    
    문제점 : RoI 크기가 다 다르지만, regression 파라미터(가중치)는 하나라서 큰 물체, 작은 물체 모두 똑같은 방식으로 보정 → 최적화 어려움
    

| 기존 RoI 기반 방식 | Faster R-CNN (RPN 방식 |
| --- | --- |
| 다양한 크기의 RoI pooling | feature map에서 고정 크기(3x3) 영역 사용 |
| 모든 RoI에 동일한 regression weight | scale/ratio 별로 별도 regressor 사용 |
| RoI feature마다 크기 차이 반영 어려움 | anchor design으로 크기/비율 차이를 해결 |

→ Faster R-CNN이 좋은 이유

- 고정된 feature 크기 → 학습 안정성이 좋고, 구현을 단순화할 수 있다.
- scale/ratio 별 별도 regressor → 작은 물체/큰 물체 각각에 특화된 weight 학습 가능
- Anchor 덕분에, 고정 feature size에서도 다양한 크기의 bbox 예측 가능
    
    ![](/assets/paper_review/faster_r_cnn/9.png)
    

### 3.1.3 Training RPNs

- RPN은 end-to-end back propagation 방식으로 학습된다.
    - Faster R-CNN 전체 구조가 하나의 연결된 네트워크처럼 작동하여 입력 이미지 → 최종 loss 까지의 모든 경로가 역전파로 동시에 학습!
- Fast R-CNN에서의 ‘image-centric” 샘플링 전략을 사용한다.
    - 각각의 미니배치는 하나의 이미지에서 다수의 객체/배경 anchor들을 가진다.
    - 모든 anchor들에 대해서 손실함수를 최적화 (손실함수의 값을 가능한한 작게 만듦)하는 것이 가능하지만, negative sample(배경 샘플)이 더 우세하기 때문에, 학습결과가 negative sample 방향으로 치우칠 수도 있다.
    
    → 그렇기 때문에 무작위로 256개의 anchor를 뽑아서 미니배치의 손실함수를 계산하기 위해 사용한다. (여기서 샘플들은 negative와 positive의 비율이 1:1을 가진다.)
    
    ***만약 객체 샘플이 하나의 이미지에 128개보다 적게 있을 경우엔, negative anchor로 채운다.***
    

## 3.2 Sharing Features for RPN and Fast R-CNN

- RPN과 Fast R-CNN은 독립적으로 학습되며, 그들의 ConV 계층을 서로 다른 방식으로 수정할 것이다.

⇒ 그렇기 때문에 ConV 계층을 2가지의 네트워크에서 공유할 수 있도록 하는 기법이 필요하다.

***Faster R-CNN의 핵심!!!! ⇒ Region Proposal(RPN) 네트워크와 Fast R-CNN이 Feature Map을 공유한다!!***

1. Alternating Training
    - 해당 솔루션에서는, 첫번째로 RPN을 학습시키고, 지역 제안을 Fast R-CNN을 학습시키는데 사용한다.
    - Fast R-CNN으로 tuned 된 네트워크는 RPN을 시작할때 사용되며, 이와 같은 과정이 반복된다.
2. Approximate joint training (한번의 forward/backward로 학습)
    - RPN과 Fast R-CNN 네트워크가 학습시에 하나의 네트워크로 병합된다.
    - 한 번의 forward pass
        - Backbone CNN이 feature map 생성
        - RPN이 proposal 생성
        - proposal을 바로 Fast R-CNN Detector에 넣어서 Classification & bbox regression 수행
        - 학습과정
            - Forward pass
                - RPN이 Proposal 생성
                - Fast R-CNN detector가 이 proposal을 입력으로 받아 Loss 계산
            - Backward pass
                - RPN loss와 Fast R-CNN loss 모두 공유된 Conv Layer까지 Gradient 전파
                - 공유된 layer에서는 2개의 gradient를 합쳐서 업데이트
            
            → RPN이 내놓은 proposal 좌표에 대한 gradient는 계산하지 않는다!
            
3. Non-approximate joint Training
    - RPN에 의해 예측된 경계 박스들은 함수(입력 이미지 → RPN → bounding box)들의 input으로 사용된다.
    - Fast R-CNN안에 있는 RoI pooling layer는 input으로써 convolutional feature들과, 예측된 경계박스들을 input으로 받는다.
    - 2번 방식 (approximate joint Training)에서는 RPN이 생성한 proposal box 좌표를 고정값처럼 취급하여 좌표에 대한 gradient는 계산을 하지 않았지만, 이론적으로 완전한 joint training에서는 box 좌표도 네트워크의 출력이므로, loss를 box좌표까지 역전파해야한다.
    
    ⇒ 완전한 joint training을 하기 위해서는 box 좌표까지 gradient를 전파해야하는데, 이를 위해서는 box 좌표에 대해 미분 가능한 RoI Pooling이 필요하다.
    

### 4-Step Alternating Training

- 공유된 features들을 학습하기 위해서는 4-단계의 학습 알고리즘을 사용한다.
1. RPN을 학습시킨다. (3.1.3에서 소개한 방식으로!)
    - 이 네트워크는 ImageNet-pre-trained 모델로부터 시작되었으며, 지역 제안을 위해 fine Tuning 되었다.
2. RPN에서 제안한 지역제안을 활용하여 Fast R-CNN에서 나온 detection network를 학습시킨다.
    - detection network 또한 ImageNet-pre-trained 모델이다.
    - 해당 과정에서 2개의 네트워크는 ConV 계층을 공유하지 않는다.
3. Detector 네트워크를 사용하여 RPN 학습
    - Fast R-CNN (Detector Network)이 학습한 백본 가중치를 RPN 초기화에 사용하고, 백본은 고정한 채 RPN 전용 레이어만 미세 조정한다.
    
    ⇒ 이제 2개의 네트워크는 convolution layer를 공유한다.
    

![](/assets/paper_review/faster_r_cnn/10.png)

## 3.3 Implementation Detatils

- 해당 논문에서는 region proposal과 object detection network를 동시에 학습하고 검증하고 있다. (하나의 크기에서)
- 이미지들을 짧은 변 길이를 600 픽셀로 리사이즈한다.
- 긴 변들은 비율에 맞게 자동 조정하고, 종횡비 (aspect ratio)는 유지시킨다.
- 크기가 조정된 이미지들은, ZF와 VGG net에서 마지막 Conv 계층은 16 픽셀을 가지고 있으며, Stride를 16으로 설정해도 충분히 좋은 결과가 나왔다.
    - ZF - net, VGG - net
        - 대표적인 CNN 백본
    - Stride
        - CNN에서 필터가 한 번에 얼마나 건너뛰면서 이동하는지 나타내는 값
- Anchors
    - 3개의 스케일과 3개의 종횡비을 사용한다. (128^2, 256^2, 512^2), (1:1, 1:2, 2:1)
    - 해당 논문에서의 방식은 image pyramid나 filter pyramid를 사용하지 않는다.

***이미지의 경계를 넘나드는 anchor box들에 대해***

- 학습시에는 이미지의 경계를 넘나드는 anchor box들을 무시한다.
    - 손실에 기여할 수 있기 때문!
- 1000 x 600의 전형적인 이미지는 약 20000개의 anchor들을 가질 수 있다.
    - 하지만 만약 이미지의 경계를 넘나드는 anchor box들을 제거한다면, 6000개의 anchor들만이 남는다. (하나의 이미지당)

![](/assets/paper_review/faster_r_cnn/11.png)

- 몇 개의 RPN proposal들은 overlap되는 경우도 있다.
- 중복성을 줄이기 위해서, 해당 논문에서는 non-maximum suppression 방식을 proposal regions의 cls(각 anchor가 객체인지 아닌지) 점수에 도입한다.

# 4 Experiments

### 4.1 Experiments on PASCAL VOC

- Faster R-CNN 방식은 PASCAL VOC 2007 탐지 벤치마크를 통해 평가하였다.
- 해당 데이터셋은 약 5천개의 학습/검증 이미지와, 5천개의 테스트 이미지들로 구성된다. (20개 이상의 객체 카테고리를 가짐)

ImageNet pre-trained network

- ZF net
    - “fast” version 사용
    - 5 convolutional layers
    - 3 fully-connected layers
- VGG-16 model
    - 13 convolutional layers
    - 3 fully-connected layers

⇒ mAP(mean Average Precision) 을 사용 (객체 탐지에서 표준으로 사용되는 최종 성능 지표)

mAP (mean Average Precision)

- 클래스별 Average Precision을 계산, 이를 모든 클래스에 대해 평균 낸 값
    
    ![](/assets/paper_review/faster_r_cnn/12.png)
    
- 위의 테이블에서는 Fast R-CNN이 다양한 region proposal methods를 통해 학습되고 평가된 결과를 보여준다.
- 테이블을 살펴보면, SS (Selective Search)는 약 58.7%의 mAP를 기록하며, RPN with Fast R-CNN은 59.9%의 성능을 보이고 있다.
- RPN을 사용하는 것이 SS 혹은 EB를 사용하는 것보다 훨씬 더 빠르며, 그 이유는 convolutional 계산을 공유하기 때문이다.

### Ablation Experiments on RPN

RPN을 proposal method로 사용하였을때 차이를 보기 위해서는, 다양한 ablation study를 진행하였다.

*Ablation Study?*

![](/assets/paper_review/faster_r_cnn/13.png)

→ **Ablation Study는 모델의 성능에 가장 큰 영향을 미치는 요소를 찾기 위해 모델의 구성요소 및 feature들을 단계적으로 제거 하거나 변경해가며 성능의 변화를 관찰하는 방법**

1. RPN과 Fast R-CNN 감지 네트워크가 convolutional layer를 공유할때의 영향
    - 4-step 학습 과정에서 2번째 단계 이후에 학습을 멈춘다.
    - 분리된 네트워크를 사용하면, 결과가 58.7%로 조금 감소한다.
    
    ⇒ 관찰 결과 3번째 단계에서 dector에 의해 tuning된 피처들이 RPN을 미세조정하기 위해 사용될때 성능이 오른다는 것을 알게되었다.
    
2. Fast R-CNN 탐지 네트워크에서 RPN의 영향을 풀어보았다.
    - 해당 과정의 의도는 Fast R-CNN 모델을 2000 SS proposals 와 ZF net을 이용하여 학습시켰다.
    - 해당 탐지기를 고치고, proposal regions를 바꾸며 mAP를 통해 평가하였다.
    - 해당 ablation 과정에서는, RPN이 detector와 features을 공유하지 않는다.
- Selective Search를 300 RPN Proposals로 대체하였을때, mAP가 56.8%가 되었으며, mAP가 줄어든 이유는 training/testing proposal의 불일치 때문이었다.
- RPN은 여전히 경쟁적인 결과 (55.1%)를 top-ranked 100proposals를 사용하였을때 보이고 있으며, 이는 top-ranked RPN proposals가 정확하다는 것을 의미한다.

**CLS 출력 역할 분석**

- cls layer 제거 = proposal 점수 없음
    
    ⇒ NMS (non-max suppression)이나 ranking 불가
    

결과 :

- N=1000 → mAP 거의 동일 (55.8%)
    - 상위 제안 수가 많으면 ranking이 크게 필요하지 않다.
- N=100 → mAP 급락 (44,6%)
    - 적은 수의 제안을 쓸 경우, cls 점수 기반 ranking이 정확도 유지에 중요

**⇒ cls score는 “상위 순위 proposal”의 정확도에 큰 영향을 미친다.**

**reg 출력 역할 분석**

- reg layer 제거 = anchor box 그대로 사용

결과 :

- mAP 55.8% → 52.1% 하락
- 여러 scale/aspect ratio의 anchor box만으로도 정확도 부족
- bbox regression이 위치 보정에 필수

**⇒ reg는 제안 영역의 정밀도를 높이는 핵심**

**백본 (Backbone) 변경 효과**

- ZF-net 사용 → mAP = 56.8%
- VGG-16 사용 → mAP=59.2%
- Detector는 동일하게 SS(Selective Search) + ZF 사용

결과:

- 백본이 더 강력해질수록 RPN 제안 품질도 향상
- RPN+ZF가 이미 SS와 비슷한 성능

⇒ RPN+VGG는 SS보다 더 좋을 가능성이 높음

**Perfomance of VGG-16**

Table 3는 VGG-16의 proposal, detection의 결과를 모두 보여준다.

![](/assets/paper_review/faster_r_cnn/14.png)

- RPN+VGG를 사용하였을때, feature 를 서로 공유하지 않고도 68.5%의 mAP를 기록하고, SS baseline보다 약간 더 높았다.
- feature-shared (feature를 서로 공유)했을 경우엔, 결과가 69.9%였다.
- PASCAL VOC 2007, 2012 데이터 셋으로 더 RPN과 Detection Network를 더 학습시켰을 때, mAP는 73.2%였다.
- Table 4 에서는 PASCAL VOC 2012 test set에 대한 성능을 보이고 있으며, Table 6,7에서는 자세한 내용을 담고있다.

Table 5

- 실행속도 비교
    - Selective Search(SS): 1~2초 (매우 느림)
    - Fast R-CNN + VGG-16 :
        - SS proposals 2000개 사용 시 : 320ms
        - SVD 최적화 적용 시 : 223ms
    - Faster R-CNN (RPN+VGG-16)
        - 전체 : 198ms
        - Conv feature 공유 덕분에 RPN 자체는 10ms만 소요
    - ZF-Net backbone
        - 17fps 속도 달성

**⇒ feature 공유와 proposal 수 축소 덕분에 SS 기반 대비 큰 속도 향상**

![](/assets/paper_review/faster_r_cnn/15.png)

**Anchor 설정 실험 (Table 8)**

- 기본값 : 3 scales x 3 aspect ratios → mAP = 69.9%
- 앵커 1개만 사용 : mAP 3~4% 하락
- 3 scales + 1 aspect ratio : 69.8% (거의 동일)
- 1 sclale + 3 aspect ratios : mAP 상승 (1 anchor 대비)

⇒ scale, aspect ratio 모두 다양하게 사용하는 것이 좋다.

**λ 값 영향  (Table 9)**

- 기본값 λ =10 → cls term과 reg termdl 정규화 후 비슷한 크기
- **λ의 범위를 1~100까지 변경 → 성능 변화 약 1% 수준**

⇒ λ 값에 광범위하게 둬도 성능에 민감 X  

![](/assets/paper_review/faster_r_cnn/16.png)

**Recall-to-IoU 분석**

- Recall-to-IoU metric
    - 특정 IoU 기준 이상에서 제안이 정답 박스를 얼마나 많이 커버하는지?
- 해당 지표는 최종 mAP와 약한 상관관계만 있음 → proposal 품질 진단용
- 실험 결과 (Figure 4)
    - 비교 대상 : RPN, SS, Edgeboxes(EB)
    - proposal개수를 2000 → 1000 → 300으로 줄였을 때

**⇒ RPN은 proposal 수가 적어도 recall이 안정적 → 효율적**

![](/assets/paper_review/faster_r_cnn/17.png)

**One-stage vs Two-stage**

- **One-stage (OverFeat 스타일)**:
    - class-specific detection을 한 번에 수행 (sliding window 기반)
    - 한 단계에서 위치 + 클래스 예측 동시에
    - 실험 세팅:
        - Dense sliding window (3 scales × 3 aspect ratios)
        - Fast R-CNN이 직접 클래스 점수와 bbox regression
        - 5-scale image pyramid 버전도 테스트
- **Two-stage (Faster R-CNN)**:
    - Stage 1: class-agnostic RPN → proposal 생성
    - Stage 2: Fast R-CNN이 proposal 기반으로 클래스 + bbox 예측
    - RoI Pooling으로 proposal 위치에 맞게 feature를 잘 추출

**결과 (Table 10, ZF backbone)**:

- One-stage: mAP = 53.9%
- Two-stage: mAP = 58.7% (**+4.8% 향상**)
- 속도: one-stage가 proposal 수가 많아서 오히려 느림
- 기존 연구([2], [39])에서도 SS 대신 sliding window 쓰면 약 -6% 성능 하락 보고됨

→ two-stage 구조 (proposal → 정밀 분류/보정)가 sliding window 기반 one-stage보다 정확하고 효율적

**MS-COCO 데이터셋 사용**

데이터셋 구성

- MS COCO : 80개 객체 카테고리
- 사용데이터
    - Train : 80k 이미지
    - val : 40k 이미지
    - Test-dev : 20k 이미지
- 평가지표 : mAP

결과 (Table 11)

mAP@0.5 vs mAP@[.5,.95]

| mAP@0.5 | mAP@[.5,.95] |
| --- | --- |
| IoU 임계값 =0.5 | IoU임계값을 0.5 ~ 0.95까지 0.05 간격으로 변화시키며 평균 |
| PASCAL VOC에서 사용하던 전통적인 방식 | COCO 표준 평가 방식 |

![](/assets/paper_review/faster_r_cnn/18.png)

- Fast R-CNN baseline
    - mAP@.5 = 39.3%
    - mAP@[.5,.95] = 19.3% (기존의 Fast R-CNN과 유사)
- Faster R-CNN (Train set 학습)
    - mAP@0.5 = 42.1%
    - mAP@[.5,.95] = 21.5%
- Faster R-CNN (train+val set 학습)
    - mAP@0.5 = 42.7%
    - mAP@[.5,.95] = 21.9%

📌 **핵심 요약**

1. COCO 실험에서 앵커 scale 확대와 negative sample 범위 확장으로 성능 향상
2. Faster R-CNN은 Fast R-CNN 대비 mAP@0.5에서 +2.8%, mAP@[.5, .95]에서 +2.2% 개선
3. RPN은 특히 **높은 IoU 기준의 localization 정확도** 개선에 효과적

**만약 Faster R-CNN의 백본이 강력한 네트워크로 교체된다면?**

성능 비교 (VGG-16 → ResNet-101)

- 데이터셋 : MS COCO val set

| 백본 | 성능 (mAP@0.5) | 성능 (mAP@[.5,.95]) |
| --- | --- | --- |
| VGG-15 | 41.5% | 48.4% |
| ResNet-101 | 21.2% | 27.2% |

**COCO 데이터로 학습한 모델의 PASCAL VOC 성능에 미치는 영향**

배경

![](/assets/paper_review/faster_r_cnn/19.png)

- MS COCO는 PASCAL VOC 보다 훨씬 크고, 클래스 수도 많다.
- COCO의 카테고리는 VOC의 카테고리를 포함하는 Superset

⇒ COCO 모델을 VOC에 직접 적용 가능

실험 1 : COCO 모델 → VOC 직접 평가 (Fine-Tuning X)

- 결과 : VOC 2007 test mAP = 76.1%
- 비교 : VOC 07 + 12 데이터로만 학습 : 73.2%

**⇒ VOC 데이터를 전혀 쓰지 않아도, COCO로만 학습한 모델이 VOC 데이터로 학습한 모델보다 성능이 높다**  

실험 2.  COCO 모델을 VOC에 Fine-Tuning

- COCO 모델을 ImageNet-pretrained 모델 대신 초기 가중치로 사용
- VOC 데이터로 Fine-Tuning 진행 (Faster R-CNN 방식)
- 결과 : VOC 2007 test mAP = 78.8%

# 5 Conclusion

![](/assets/paper_review/faster_r_cnn/20.jpeg)

- 해당 논문에서는 RPN를 통해 효율적이고 정확한 지역 제안을 제시하고있다.
- Convolution Features을 공유함으로써, 지역 제안을 하는 단계에서의 비용을 절감할 수 있다.
- 논문에서의 방식은 딥러닝 기반의 객체 탐지 시스템으로써, near-real-time 탐지가 가능하다.

<aside>
💡

**정리**

**Faster R-CNN은 Region Proposal Network(RPN)를 도입해 백본 CNN의 특징 맵을 검출 단계와 공유하여 제안 영역 생성 속도를 크게 높이고, 적은 연산으로도 높은 정확도의 객체 탐지를 실시간에 가깝게 수행하는 방법**

</aside>
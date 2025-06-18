---
layout: post
title: "DeepLearning Study Chapter 3~5"
date: 2025-04-10 14:00:00 +0900
categories: Study,DeepLearning
tags : DeepLearning,Python
---

# Chapter_3

신경망 → Chapter_2 에서 가중치를 설정하는 작업을 우리 인간이 정하는 과정을 해결

### 신경망

- 가중치 매개변수의 적절한 값을 데이터로부터 자동으로 학습하는 능력

### 퍼셉트론 - 신경망

신경망의 그림 

- 입력층, 출력층, 은닉층으로 나뉜다.
- 은닉층의 뉴런은 사람 눈에는 보이지 않는다. (입력층이나 출력층과 달리)
    
    ![](/assets/bamboo_1/chapter345/1.png)
    

퍼셉트론 복습

![](/assets/bamboo_1/chapter345/image.png)

![](/assets/bamboo_1/chapter345/image1.png)

위의 그림은 x1 과 x2 라는 2개의 신호를 입력받아 y를 출력하는 퍼셉트론이고, 해당 퍼셉트론을 수식으로 나타낸 것이다.

b : 편향을 나타내는 매개변수, 뉴런이 얼마나 쉽게 활성화되는지 제어

w1 , w2 : 각 신호의 가중치를 나타내는 매개변수, 각 신호의 영향력 제어

![](/assets/bamboo_1/chapter345/image2.png)

- 위의 그림과 같이 가중치가 b이고 입력이 1인 뉴런을 추가함으로써 편향을 추가할 수 있다.
- x1 , x2 , 1 이라는 3개의 신호가 뉴런에 입력되어, 각 신호에 가중치를 곱한 후, 다음 뉴런에 전달된다.

![](/assets/bamboo_1/chapter345/image3.png)

- 위의 식은 3개의 신호에 각각의 가중치를 곱한 값을 모두 더하여 조건 분기의 동작을 하나의 함수로 나타냄.
    - 함수 : h(x)
- 조건 분기의 동작?
    - 각 신호에 가중치를 곱한 값들을 모두 더하여 0을 넘으면 1을 출력, 그렇지 않으면 0을 출력

### 활성화 함수

위에서 정의한 h(x) 함수, 즉 입력 신호의 총합을 출력신호로 변환하는 함수를 **활성화 함수**라고 함.

위의 식을 다시 정리해보면, 각각의 신호에 가중치가 곱해진 입력신호의 총합을 계산 

⇒ 그 합을 활성화 함수에 입력해 결과를 도출

아래의 식으로 표현해 볼 수 있다.

![](/assets/bamboo_1/chapter345/image4.png)

![](/assets/bamboo_1/chapter345/image5.png)

다음 과정을 그림으로 표현해보면 다음과 같다.

위의 사진에서 원으로 표현된 것들을 **노드**라고 표현, 뉴런과 같은 의미로 사용.

### 계단함수

- 위에서 살펴본 h(x) 와 같은 형태를 띄고 있다. 임계값을 경계로 출력이 바뀜.

계단함수 구현하기

```python
def step_function(x):
    if x>0:
        return 1
    else:
        return 0
```

```python
def step_function(x): # numpy 배열도 인수로 받아주기 위해
    y=x>0
    return y.astype(np.int)
```

- 넘파이 배열도 지원하도록 수정

왼쪽의 함수는 인수로 실수(부동소수점)만 받아들인다. 오른쪽의 코드와 같이 넘파이 배열도 인수로 전달할 수 있도록 수정한다.

```python
import numpy as np
x=np.array([-1.0,1.0,2.0])
x
>>> array([-1.,  1.,  2.])
-------------------------------------------
y=x>0
y
>>> array([False,  True,  True])
```

- 배열 각각의 원소가 0보다 크면 True, 0 이하면 False를 반환하는 새로운 배열 y 생성

해당 배열 y 는 현재 bool 형 이므로, 형 변환을 통해 int로 바꿔줘야한다.

```python
y=y.astype(int) ## NumPy 1.20 버전부터 np.int가 제거되었기 때문에 아래와 같이 형변환
y
>>> array([0, 1, 1])
```

```python
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x>0, dtype=int)
x=np.arange(-5.0,5.0,0.1)
y=step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()
```

![](/assets/bamboo_1/chapter345/2.png)

- 그래프를 통해 계단 함수는 0을 경계로 출력이 0에서 1(또는 1에서 0)으로 바뀌는 것을 확인할 수 있다.

---

### 시그모이드 함수

![](/assets/bamboo_1/chapter345/image6.png)

![](/assets/bamboo_1/chapter345/3.png)

- 신경망에서 자주 이용하는 활성화 함수
- 시그모이드 함수를 이용하여 신호를 변환, 변환된 신호를 다음 뉴런에 전달

시그모이드 함수 구현하기

```python
def sigmoid(x):
    return 1/(1+np.exp(-x))
```

- 실제 넘파이 배열을 제대로 처리하는지 확인

```python
x=np.array([-1.0,1.0,2.0])
sigmoid(x)

>>> array([0.26894142, 0.73105858, 0.88079708])
```

- 어느 크기의 배열도 처리해줄 수 있는 이유 → **브로드캐스트 ( 배열의 크기를 자동으로 맞춰줌 )**
    - **DL_Chapter_1 브로드캐스트 참고**

시그모이드 함수 그리기

```python
x=np.arange(-5.0,5.0,0.1)
y=sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()
```

![](/assets/bamboo_1/chapter345/4.png)

---

시그모이드 함수 VS 계단 함수

![](/assets/bamboo_1/chapter345/image7.png)

위의 그래프의 비교를 통해 다른 점을 찾을 수 있다. 우선, 가장 눈에 띄는 차이점은 ‘매끄러움’이다.

계단 함수는 0과 1 중 하나의 값만 돌려주는 반면, 시그모이드 함수는 실수를 돌려준다는 점도 다르다.

| **계단함수** | **시그모이드 함수** |
| --- | --- |
| **0 or 1** | **실수 (연속적인 실수)** |

공통점 : 

1. 2개의 그래프 모두 0과 1 사이의 값을 출력
2. 입력이 작을 때의 출력은 0에 가깝고, 입력이 커지면 출력이 1에 가까워지는 비슷한 형태

---

### 비선형 함수

위에서 살펴본 공통점 이외에 중요한 공통점으로는 2개의 함수 모두 **‘비선형 함수’** 라는 것이다.

신경망 

- 활성화 함수 : 비선형 함수를 사용해야한다.
    - 선형함수를 활성화 함수로 사용한다면? → 신경망의 층을 깊게 하는 의미가 사라지게 된다.

ex) 만약 활성화 함수로  h(x)=cx 라는 함수를 사용한다면, 해당 신경망을 3층으로 쌓아도 결국 c를 3제곱한 식, 즉 은닉층이 존재하지 않는 신경망으로 표현됨.

---

### ReLU 함수

ReLU 함수는 입력이 0을 넘으면 그 입력을 그대로 출력, 0 이하이면 0을 출력하는 함수

![](/assets/bamboo_1/chapter345/image8.png)

![](/assets/bamboo_1/chapter345/image9.png)

```python
def relu(x):
	return np.maximum(0,x)
```

maximum은 두 입력 중 큰 값을 선택해 반환한다.

---

### 다차원 배열의 계산

NumPy의 다차원 배열을 사용한 계산법 → 신경망 구현

다차원 배열

- ‘숫자의 집합’ 즉, 숫자가 차원에 상관없이, n차원으로 나열하는 것을 통틀어 다차원 배열이라고 한다.

-1차원 배열

```python
import numpy as np
A=np.array([1,2,3,4])
print(A)

>>> [1 2 3 4]
------------------------------------
np.ndim(A) # 배열의 차원 수 확인

>>> 1

A.shape # 배열의 형상 확인
>>> (4,)

A.shape[0] 
>>> 4
```

-2차원 배열

```python
B=np.array([[1,2],[3,4],[5,6]])
print(B)

>>>
[[1 2]
 [3 4]
 [5 6]]
 
 np.ndim(B)
 >>> 2
 
 B.shape
 >>> (3,2)
```

- 위와 같은 2차원 배열은 ‘3 X 2 배열’임. 처음 차원에는 원소가 3개, 다음 차원에는 원소가 2개 있다는 의미이다. (0차원, 1차원 ..)
- 2차원 배열은 행렬 (Matrix) 라고 부름.
- 가로방향 - 행 , 세로방향 - 열

### 행렬의 곱

![](/assets/bamboo_1/chapter345/5.png)
위의 그림과 같이 행렬의 곱셈은 왼쪽 행렬의 행 (가로)와 오른쪽 행렬의 열(세로)을 원소별로 곱하고 그 값들을 더해서 계산한다.

```python
A=np.array([[1,2],[3,4]])
A.shape
>>> (2,2)

B=np.array([[5,6],[7,8]])
B.shape
>>> (2,2)

np.dot(A,B)
>>> array([[19, 22],
       [43, 50]])
```

<aside>

2개의 행렬의 곱 : 넘파이 함수 np.dot() 으로 계산 가능

**np.dot(A,B) ≠ np.dot(B,A)**

</aside>

- np.dot()은 입력이 1차원 배열이면 벡터를, 2차원 배열이면 행렬 곱을 계산
- 또한, 행렬 A의 1번째 차원의 원소수 (열 수) 와 행렬 B의 0번째 차원의 원소 수(행 수)가 같아야 한다.

![](/assets/bamboo_1/chapter345/image10.png)

- A 행렬이 2차원이고 B가 1차원 배열일 때도 똑같이 적용

![](/assets/bamboo_1/chapter345/image11.png)

### 신경망에서의 행렬 곱

- 넘파이 행렬을 이용하여 신경망 구현

![](/assets/bamboo_1/chapter345/image12.png)

- 다음과 같은 신경망을 가정.
- 입력이 2개이고, 그에 상응하는 가중치가 6개가 존재한다.
- 위와 같은 신경망을 행렬의 곱으로 나타내보자.

![](/assets/bamboo_1/chapter345/image13.png)

```python
X=np.array([1,2])
X.shape
>>> (2,)

W=np.array([[1,3,5],[2,4,6]])
print(W)
>>> [[1 3 5]
		 [2 4 6]]
		 
W.shape
>>> (2,3)

Y=np.dot(X,W)
print(Y)
>>> [ 5 11 17 ]
```

### 3층 신경망 구현하기

![](/assets/bamboo_1/chapter345/image14.png)

- 입력층 (0층) : 2개
- 첫번째 은닉층 (1층) : 3개
- 두 번째 은닉층 (2층) : 2개
- 출력층 (3층) : 2개의 뉴런

![](/assets/bamboo_1/chapter345/image15.png)

- 위의 사진과 같이 입력층에서 1층으로 전달하는 과정을 살펴보자
- 위의 그림에서는 편향을 뜻하는 뉴런이 추가되었다.
- 편향은 앞 층의 편향 뉴런이 하나뿐이기 때문에 편향의 오른쪽 아래 인덱스는 비어있다.

![](/assets/bamboo_1/chapter345/image16.png)

위와 같은 식으로 1층 첫번째 뉴런을 표현할 수 있다.

- 여기에 행렬의 곱을 사용하면 1층의 ‘가중치 부분’을 다음 식처럼 간소화 할 수 있다.

![](/assets/bamboo_1/chapter345/6.png)
위의 필기와 같이 각각의 행렬을 표현할 수 있고,

![](/assets/bamboo_1/chapter345/image17.png)

다음과 같은 식으로 표현할 수 있다.

각각의 행렬의 크기를 살펴보면 행렬간의 곱셈이 가능하다는 것도 확인할 수 있다.

```python
X=np.array([1.0,0.5])
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1=np.array([0.1,0.2,0.3])
A1=np.dot(X,W1)+B1
A1
>>> array([0.3, 0.7, 1.1])
```

![](/assets/bamboo_1/chapter345/7.png)

- 은닉층에서의 가중치 합 (가중 신호 + 편향의 총합)을 a로 표기하고 활성화 함수 h()로 변환된 신호를 z로 표기.

```python
Z1=sigmoid(A1)
Z1
>>> array([0.57444252, 0.66818777, 0.75026011])

```

- 활성화 함수를 통하여 변환된 값은 그 다음층의 입력값으로 사용된다.

![](/assets/bamboo_1/chapter345/image18.png)

![](/assets/bamboo_1/chapter345/8.png)

- **1층에서 2층으로 가는 과정**

```python
W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])

print(Z1.shape) # (3,)
print(W2.shape) # (3,2)
print(B2.shape) # (2,)

A2=np.dot(Z1,W2)+B2
Z2=sigmoid(A2)
print(Z2)

>>> (3,)
(3, 2)
(2,)
[0.62624937 0.7710107 ] ## 이게 이제 다음층의 입력값이 됨!
```

- **2층에서 출력층으로의 신호 전달**

![](/assets/bamboo_1/chapter345/image19.png)

![](/assets/bamboo_1/chapter345/9.png)

```python
def identity_function(x): ## 항등함수 정의
    return x

W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])

A3=np.dot(Z2,W3)+B3
Y=identity_function(A3)
Y
>>> array([0.31682708, 0.69627909])
```

출력층의 활성화 함수는 ‘시그마’로 표현.

### 구현 정리

```python
def init_network():
    network={}
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']=np.array([0.1,0.2,0.3])
    network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']=np.array([0.1,0.2])
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b3']=np.array([0.1,0.2])
    return network

def forward(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=identity_function(a3)

    return y
network=init_network()
x=np.array([1.0,0.5])
y=forward(network,x)
print(y)

>>> [0.31682708 0.69627909]
```

- init_network() 라는 함수를 통해 network라는 딕셔너리 안에 각 층의 가중치와 편향을 저장하고, forward() 라는 함수를 통해 행렬곱을 진행하고, 그 값을 활성화 함수로 전달, 그 출력값을 다음층의 입력값으로 전달하는 과정.

### 출력층 설계하기

신경망 - 분류, 회귀

- 어떤 문제를 해결하느냐에 따라 출력층에서 사용하는 활성화 함수가 달라진다.
- 회귀 : 항등함수
- 분류 : 소프트맥스 함수

항등 함수 (신경망이 회귀문제를 해결할때 사용)

- 입력을 그대로 출력

![](/assets/bamboo_1/chapter345/image20.png)

소프트맥스 함수 (신경망이 분류문제를 해결할때 사용)

![](/assets/bamboo_1/chapter345/image21.png)

- exp(x)는 지수함수, e는 자연상수를 뜻함.
- n = 출력층의 뉴런 수
- yk = 그 중 k 번째 출력

![](/assets/bamboo_1/chapter345/image22.png)

```python
a=np.array([0.3,2.9,4.0])

exp_a=np.exp(a)
print(exp_a)
>>> [ 1.34985881 18.17414537 54.59815003]

sum_exp_a=np.sum(exp_a)
print(sum_exp_a)
>>> 74.1221542101633

y=exp_a/sum_exp_a
print(y)
>>> [0.01821127 0.24519181 0.73659691]

def softmax(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
```

- 위에서 구현한 소프트맥스 함수는 지수 함수가 아주 큰 값을 반환할 확률이 높고, 큰 값끼리의 나눗셈을 수행하면 결과 수치가 불안정해진다.

→ 이 문제를 개선하기 위한 수식

![](/assets/bamboo_1/chapter345/image23.png)

위의 식과 같이 분모와 분자에 임의의 숫자 C를 곱한다.

C를 지수 함수 안으로 옮겨 log C 를 만든다.

지수 함수 안으로 옮긴 log C 를 C’ 라는 새로운 기호로 바꿔준다.

이렇듯 지수 함수에 어떠한 수를 더하거나 빼도 그 값은 변하지 않는다.

```python
a=np.array([1010,1000,990])
np.exp(a) / np.sum(np.exp(a))
>>> array([nan,nan,nan])
```

위의 코드와 같이 큰 값을 exp() 지수 함수에 전달했을때 값이 너무 커 오버플로우가 발생하여 출력이 nan으로 나온다.

→ 이를 해결하기 위해 앞서 살펴본 것처럼 어떠한 수를 더하거나 빼도 그 결과는 바뀌지 않는다는 것을 활용하여, 입력 신호 중 최댓값을 이용한다.

```python
c=np.max(a)
a-c
>>> array([  0, -10, -20])

np.exp(a-c)/np.sum(np.exp(a-c))
>>> array([9.99954600e-01, 4.53978686e-05, 2.06106005e-09])
```

이렇게 입력 신호 중 최댓값을 빼주면 올바르게 계산할 수 있다.

이를 앞서 만든 소프트맥스 함수에 적용

```python
def softmax(a):
    c=np.max(a) ## 입력신호 중 최댓값을 추출
    exp_a=np.exp(a-c) ## 최댓값을 빼준다.
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
```

소프트맥스 함수의 특징

```python
a=np.array([0.3,2.9,4.0])
y=softmax(a)
print(y)
>>> [0.01821127 0.24519181 0.73659691]

np.sum(y)
>>> np.float64(1.0)
```

<aside>

위의 코드와 같이 소프트맥스 함수의 출력은 0에서 1.0 사이의 실수이며, 소프트맥스 함수 출력의 총합은 1이다.

</aside>

위의 출력을 살펴보면

y[0] → 0.018 (1.8%)

y[1] → 0.245 (24.5%)

y[2] → 0.736 (73.6%)

이렇게 해석할 수 있으며, 2번째 원소의 확률이 가장 높으니 답은 2번째 클래스다.

or 74%의 확률로 2번째 클래스, 25%의 확률로 1번째 클래스, 1%의 확률로 0번째 클래스다. 라고 결론을 낼수도 있다.

소프트맥스 함수에서 지수 함수는 단조 증가 함수이기 때문에, 결국 a의 원소들 사이의 대소 관계가 y의 원소들 사이의 대소 관계로 그대로 이어진다.

→ 소프트맥스 함수를 적용해도 출력이 가장 큰 뉴런의 위치는 달라지지 않는다. (소프트맥스 함수를 생략하기도 함)

**출력층의 뉴런 수 정하기**

→ 분류에서는 분류하고 싶은 클래스 수로 출력층의 수를 설정하는 것이 일반적이다.

![](/assets/bamboo_1/chapter345/image24.png)

위의 사진에서 출력층 뉴런은 위에서부터 차례로 숫자 0~9를 의미하며, y2뉴런이 가장 큰 값을 출력했다면, 입력이미지를 숫자 2로 판단한다.

## 손글씨 숫자 인식

추론 과정만 구현

→ 신경망의 **순전파**

MNIST 데이터셋을 활용.

훈련 이미지들을 사용하여 모델 학습 → 학습한 모델로 시험 이미지들을 얼마나 정확하게 분류하는지 평가

mnist.py 파일에 정의된 load_mnist() 함수를 이용하여 MNIST 데이터를 가져온다.

load_mnist() 함수의 인수로는 normalize, flatten, one_hot_label 3가지를 설정할 수 있다.

3개의 인수 모두 bool 값이다.

| normalize | 입력 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 
정규화 할지를 정함 |
| --- | --- |
| flatten | 입력 이미지를 평탄하게, 즉 1차원 배열로 만들지 정함
False 설정 : 입력이미지를 3차원 배열로 저장 |
| one_hot_label | 범주형 데이터를 숫자에 대응시키는 방식 |

```python
import sys,os
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train),(x_test,t_test) =\
    load_mnist(flatten=True,normalize=False)

img=x_train[0]
label=t_train[0]
print(label)

print(img.shape)
img=img.reshape(28,28)
print(img.shape)

img_show(img)

>>>
5
(784, )
(28, 28)
```

![](/assets/bamboo_1/chapter345/10.png)

- 위의 코드에서 flatten = True 로 지정했기 때문에 읽어들인 이미지는 1차원 배열로 저장되어있다.
- reshape() 메서드에 원하는 형상을 인수로 지정하면 넘파이 배열의 형상을 다시 바꿀 수 있다.

### 신경망의 추론 처리

입력층 뉴런 : 784개 → 이미지 크기가 28x28 = 784개

출력층 뉴런 : 10개 → 문제가 0~9까지의 숫자를 구분하는 문제

1번째 은닉층 뉴런 : 50개

2번째 은닉층 뉴런 : 100개

```python
import pickle
def get_data():
    (x_train,t_train),(x_test,t_test)=\
        load_mnist(normalize=True,flatten=True, one_hot_label=False)
    return x_test,t_test
    
def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network=pickle.load(f)
    return network

def predict(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=softmax(a3)

    return y
```

- init_network() 에서는 pickle 파일인 sample_weight.pkl에 저장된 ‘학습된 가중치 매개변수’를 읽는다.
    - 해당 파일엔 가중치화 편향 매개변수가 딕셔너리 변수로 저장되어 있음
- normalize=True , 즉 정규화, 데이터 전처리 과정도 진행

<aside>

3개의 함수를 활용해 신경망에 의한 추론을 수행, 정확도 평가 진행

</aside>

```python
x,t=get_data()
network=init_network()

accuracy_cnt=0
for i in range(len(x)):
    y=predict(network,x[i])
    p=np.argmax(y)
    if p==t[i]:
        accuracy_cnt+=1
print("Accuracy:"+str(float(accuracy_cnt)/len(x)))

>>> Accuracy:0.9352
```

추론 과정 흐름:

1. MNIST 데이터셋을 얻고 네트워크 생성
2. for 문을 통해 x에 저장된 이미지 데이터 1장씩 predict()로 전달
3. predict() 함수는 각 레이블의 확률을 배열로 반환
    1. [0.1,0.3,0.2 … ] 와 같이 전달
    2. ‘0’일 확률이 0.1 , ‘1’일 확률이 0.3 과 같이 해석
4. np.argmax() 함수로 이 배열에서 값이 가장 큰 원소의 인덱스 추출
5. 예측한 답변과 정답 레이블을 비교하여 맞은 숫자는 accuracy_cnt 에 +1
6. 전체 이미지 숫자로 나눠 정확도 계산

## 배치 처리

```python
x, _ =get_data()
network=init_network()
W1,W2,W3=network['W1'],network['W2'],network['W3']

x.shape
>>> (10000,784)
x[0].shape
>>> (784,)
W1.shape
>>> (784, 50)
W2.shape
>>> (50,100)
W3.shape
>>> (100,10)
```

다차원 배열의 대응하는 차원의 원소 수가 일치 → 원소가 10개인 1차원 배열 출력

![](/assets/bamboo_1/chapter345/image25.png)

만약 이미지 여러 장을 한꺼번에 입력하는 경우엔?

![](/assets/bamboo_1/chapter345/image26.png)

- 출력은 100x10의 형상을 가진다.
- 100장 분량 입력 데이터의 결과가 한 번에 출력

<aside>

묶은 데이터 ⇒ 배치

</aside>

```python
x,t=get_data()
network=init_network()
batch_size=100 #배치 크기
accuracy_cnt=0

for i in range(0,len(x),batch_size):
    x_batch=x[i:i+batch_size]
    y_batch=predict(network,x_batch)
    p=np.argmax(y_batch,axis=1)
    accuracy_cnt+=np.sum(p==t[i:i+batch_size])

print("Accuracy:"+str(float(accuracy_cnt)/len(x)))
```

100장 씩 묶어나가면서 그룹을 형성한다고 생각!

## Chapter_3 정리

<aside>

신경망 순전파

각 층의 뉴런 → 다음 층의 뉴런으로 신호 전달

신경망에서 사용하는 활성화 함수 (매끄럽게 변화하는 함수)

sigmoid 함수

ReLU 함수

넘파이의 다차원 배열 → 신경망을 효율적으로 구현 가능

기계 학습 문제 - 회귀, 분류

출력층의 활성화 함수

회귀 : 항등 함수 (출력값 그대로 전달)

분류 : softmax() 함수 → 출력층의 뉴런 수 == 클래스 수

입력 데이터를 묶은 것 : 배치

배치 단위로 진행 → 속도 **↑**

</aside>


# Chapter_4

### 신경망 학습

- 데이터를 보고 학습할 수 있다.

기계학습 (Machine Learning) → 데이터에서 답을 찾고, 데이터에서 패턴을 발견하고, 데이터로 이야기를 만든다. (데이터가 중요!)

신경망 & 딥러닝 → 사람의 개입을 더욱 배제할 수 있도록 해준다.

ex) 숫자 5를 분류하는 프로그램

- 이미지에서 특징 추출 → 특징을 활용하여 기계학습 기술로 학습
    - 이미지의 특징 : 보통 벡터, SIFT, SURF, HOG 등의 특징을 많이 사용
- 규칙을 찾아내는 역할 : 기계
    - 이미지를 벡터로 변환할 때 사용하는 특징 → ‘사람’이 설계

![](/assets/bamboo_1/chapter345/chapter4/image.png)

→ 첫번째 방식과 두번째 방식 모두 사람이 알고리즘을 만들거나, 이미지의 특징을 전달해줘야함

### 신경망

- 이미지를 ‘있는 그대로’ 학습
    - 즉, 이미지에 포함된 중요한 특징까지 ‘기계’가 스스로 학습

### 훈련 데이터 & 시험 데이터

- 훈련 데이터 사용 → 최적의 매개변수 도출
- 시험 데이터를 사용 → 앞서 훈련한 모델의 성능 평가

**모델의 범용성, 범용 능력을 제대로 평가하기 위함 (과적합 , Overfitting 방지)**

## 손실함수

신경망 → ‘하나의 지표’를 기준으로 최적의 매개변수 값을 탐색

일반적으로 **오차제곱합, 교차 엔트로피 오차** 사용

### 손실함수_오차제곱합_SSE

![](/assets/bamboo_1/chapter345/chapter4/image1.png)

```python
def sum_squares_error(y,t):
    return 0.5 * np.sum((y-t)**2)
```

yk : 신경망의 출력

tk : 정답 레이블

k : 데이터의 차원 수

![](/assets/bamboo_1/chapter345/chapter4/image2.png)

위의 사진에서 y : 소프트맥스 함수의 출력, 따라서 확률로 해석 가능, 해당 이미지가 ‘0’일 확률이 0.1, ‘1’일 확률이    0.05

위의 사진에서 t : 정답을 가리키는 위치의 원소, 즉 해당 레이블이 1이면 정답을 알 수 있음

이렇듯, 한 원소만 1로 하고, 그 외는 0으로 나타내는 표기법 : One - hot encoding

### 교차 엔트로피 오차

![](/assets/bamboo_1/chapter345/chapter4/image3.png)

log : 밑이 e인 자연로그

yk : 신경망의 출력

tk : 정답 레이블 (정답 : 1, 나머지 : 0)

![](/assets/bamboo_1/chapter345/chapter4/image4.png)

x 가 1일때, y는 0, x가 0에 가까울수록 y는 점점 작아진다.

```python
def cross_entropy_error(y,t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta)) # 아주 작은 값인 delta를 더하여 마이너스 무한대가 발생하지 않도록 설정
```

![](/assets/bamboo_1/chapter345/chapter4/1.png)

![](/assets/bamboo_1/chapter345/chapter4/2.png)

- 첫번째 정답일때의 출력이 0.6인 경우 교차 엔트로피 오차는 약 0.51인 반면, 정답일때의 출력이 0.1인 경우엔 교체 엔트로피가 무려 2.3까지 올라감.
- 즉, 오차값이 더 작은 첫 번째 추정이 정답일 가능성이 높다고 판단

### 미니배치 학습

기계학습 문제 → 훈련 데이터를 활용해 학습

훈련 데이터에 대한 손실 함수 값 → 최대한 줄여주는 방향으로 매개변수를 고쳐나감.

**훈련 데이터 모두에 대한 손실 함수의 합을 구하는 방법**

![](/assets/bamboo_1/chapter345/chapter4/image5.png)

ynk : 신경망의 출력

tnk : 정답 레이블

- 마지막에 N으로 나눠줌으로써 정규화하고있음

만약 데이터가 엄청 많다면??

→ 데이터 일부를 추려 전체의 ‘근사치’로 이용

신경망 학습 : 훈련데이터로부터 일부만 골라 학습 수행 **(미니배치)**

ex) 60,000개의 데이터 중에서 100장만을 추려서 사용

![](/assets/bamboo_1/chapter345/chapter4/3.png)

![](/assets/bamboo_1/chapter345/chapter4/4.png)

→ 훈련 데이터 중에서 무작위로 10개의 이미지만 추려냄!

### **(배치용) 교차 엔트로피 오차 구현하기**

![](/assets/bamboo_1/chapter345/chapter4/5.png)

- 만약 y가 1차원이라면, 즉 데이터 하나당 교차 엔트로피 오차를 구하는 경우는 reshape 함수로 데이터의 형상을 바꿔줌. → 이미지 1장당 평균의 교차 엔트로피 오차 계산

정답 레이블이 one_hot_label이 아니라 ‘2’나 ‘7’과 같이 숫자 레이블로 주어졌을 때의 교차 엔트로피 오차

![](/assets/bamboo_1/chapter345/chapter4/6.png)

차이점 : one_hot_label에서 t가 0인 원소는 교차 엔트로피 오차도 0 이므로, 무시 가능

np.log(y[np.arrange(batch_size),t] → batch_size -1 크기의 배열을 생성

현재 t=[2,7,0,9,4], 위의 코드는 각 데이터의 정답 레이블에 해당하는 신경망의 출력을 추출

### 손실 함수 설정 이유?

**“미분”**

- 최적의 매개변수를 찾을때 손실 함수의 값을 가능한 작게 만드는 매개변수의 값을 찾는다.

→ 이러한 과정에서 매개변수의 미분을 계산, 그 미분 값을 단서로 매개변수의 값을 갱신하는 과정 반복

**손실 함수에서의 매개변수의 미분?**

→ 손실함수에서 가중치 매개변수의 값을 아주 조금 움직였을때 손실 함수가 어떻게 변하는가?

미분 값 == 0 : 가중치 매개변수를 어느 쪽으로 움직여도 손실 함수의 값은 줄어들지 않는다.

→ 가중치 매개변수 갱선 멈춤.

<aside>

신경망을 학습할 때 정확도를 지표로 삼아서는 안된다. 정확도를 지표로 하면 매개변수의 미분이 대부분의 장소에서 0이 되기 때문이다.

</aside>

왜 정확도를 지표로 삼으면 매개변수의 미분이 대부분의 장소에서 0이 될까?

매개변수를 조금 변화시키더라도  정확도는 크게 변하지 않기 때문 → 그 후의 학습에 영향을 미침.

**즉, 매개변수의 작은 변화가 주는 영향을 제대로 받아들이지 않고 결국 손실 함수의 값에는 아무런 변화가 나타나지 않음.**

- 계단 함수가 사용되지 않는 이유도 이에 해당됨.

![](/assets/bamboo_1/chapter345/chapter4/image6.png)

계단함수는 순간적인 변화를 일으키지만, 시그모이드 함수의 미분은 연속적으로 변함

→ 즉 시그모이드 함수의 미분은 어느 장소라도 0이 되지 않음. → 올바른 학습 가능

### 수치 미분

![](/assets/bamboo_1/chapter345/chapter4/image7.png)

미분이란? → ‘특정 순간’의 변화량

결국 x의 ‘작은 변화’가 함수 f(x)를 얼마나 변화시키느냐를 의미

![](/assets/bamboo_1/chapter345/chapter4/7.png)

진정한 미분 → x위치의 함수의 기울기

- 위와 같이 구현하면 저 h 의 값에 반올림 오차를 일으켜 그냥 0으로 출력됨
- 위의 구현에서 진정한 미분 즉, 특정 위치에서의 기울기를 구하는 것이 아니라, x+h 와 x사이의 기울기를 구하는 것임. 진정한 미분과는 차이가 있다.
- h를 무한히 0으로 좁히는 구현이 불가능하여 생기는 한계

![](/assets/bamboo_1/chapter345/chapter4/image8.png)

위와 같은 한계 2가지를 극복하기 위한 방법 :

1. 미세한 값 h를 10-4승 으로 이용 → 좋은 결과를 얻는다고 알려짐
2. x+h, x-h 일때의 함수의 차분을 계산하는 방법 사용 → x를 중심으로 그 전 후를 계산하는 방식

![](/assets/bamboo_1/chapter345/chapter4/8.png)

### 수치 미분의 예

![](/assets/bamboo_1/chapter345/chapter4/image9.png)

다음과 같은 수식을 파이썬으로 구현

![](/assets/bamboo_1/chapter345/chapter4/9.png)

![](/assets/bamboo_1/chapter345/chapter4/10.png)

해당 그래프에서 x=5일때, 10일때의 함수를 미분 계산

![](/assets/bamboo_1/chapter345/chapter4/11.png)

위의 식을 미분하여 계산해보면 각각 0.2,0.3이 나오는 것을 알 수 있음

→ 우리가 정의한 함수를 통해 계산한 값과 오차가 거의 없는 것을 알 수 있다.

### 편미분

![](/assets/bamboo_1/chapter345/chapter4/image10.png)

→ 변수가 2개이 경우!

![](/assets/bamboo_1/chapter345/chapter4/12.png)

각 원소를 제곱하고 더하는 수식

![](/assets/bamboo_1/chapter345/chapter4/image11.png)

- 다음과 같은 3차원으로 그려진다.
- 위의 그래프를 보면 변수가 2개이다 (x1,x0)
- 이렇게 변수가 여러개인 함수에 대한 미분 : 편미분

![](/assets/bamboo_1/chapter345/chapter4/image12.png)

![](/assets/bamboo_1/chapter345/chapter4/image13.png)

![](/assets/bamboo_1/chapter345/chapter4/image14.png)

![](/assets/bamboo_1/chapter345/chapter4/13.png)

→ 변수가 1개인 함수를 정의하고 그 함수를 미분하는 형태로 구현

편미분 : 변수가 1개인 미분과 마찬가지로 특정 장소의 기울기를 구함

### 기울기

앞서 편미분은 2가지의 변수를 따로따로 계산, 하지만 변수들의 계산을 동시에 하고싶다면?

양쪽의 편미분을 묶어서 계산 → 모든 변수의 편미분을 벡터로 정리한 것 **기울기**

![](/assets/bamboo_1/chapter345/chapter4/14.png)

f는 함수, x 는 넘파이 배열, 넘파이 배열 x의 각 원소에 대해서 수치 미분을 구함

![](/assets/bamboo_1/chapter345/chapter4/15.png)

여기서 기울기가 의미하는 건 무엇일까?

기울기의 결과에 마이너스를 붙인 벡터를 그려봄

![](/assets/bamboo_1/chapter345/chapter4/image15.png)

화살표들이 한 점을 향하고 있음, 가장 낮은 곳에서 멀어질수록 화살표의 크기가 커짐

기울기가 가리키는 쪽 → 각 장소에서 함수의 출력 값을 가장 크게 줄이는 방향

### 경사법 (경사 하강법)

→ 현 위치에서 기울어진 방향으로 일정 거리만큼 이동, 이동한 곳에서도 기울기를 구하고 또 그 기울어진 방향으로 나아가기를 반복

- 신경망 → 최적의 매개변수를 학습시에 찾아야함
- 최적 → 손실함수가 최솟값이 될 때의 매개변수 값
- 이렇게 함수의 최소값을 찾는 방식 → 경사법
- 각 지점에서 함수의 값을 낮추는 방안을 제시하는 지표 : 기울기

![](/assets/bamboo_1/chapter345/chapter4/image16.png)

여기서 학습률이 등장, 위의 eta 기호는 학습률을 의미

학습률 → 한번의 학습으로 얼마만큼 학습해야 할지, 즉 매개변수 값을 얼마나 갱신하느냐를 정하는 것

경사하강법 구현

![](/assets/bamboo_1/chapter345/chapter4/16.png)

lr : 학습률

init_x : 초기값

f : 최적화하려는 함수

**미분하고, 미분한 값에 학습률을 곱하여 갱신하는 처리를 step_num 번 반복**

![](/assets/bamboo_1/chapter345/chapter4/image17.png)

![](/assets/bamboo_1/chapter345/chapter4/17.png)

- 초기값을 -3.0,4.0으로 설정, 경사법을 사용하여 최솟값 탐색 시작
- 최종 결과는 거의 0에 가까움 (경사법으로 최소값을 찾음)

![](/assets/bamboo_1/chapter345/chapter4/image18.png)

x0,x1 이 각각 초기값인 -3.0,4.0에서 점점 가장 낮은 장소인 원점에 가까워지고 있음

학습률이 너무 큰 경우엔 큰 값으로 발산해버린다.

### 신경망에서의 기울기

![](/assets/bamboo_1/chapter345/chapter4/image19.png)

- 예를 들어 신경망에 다음과 같은 가중치, 손실함수가 존재한다고 가정
- 모든 가중치에 대한 경사를 구해야한다.
- 즉, 각각의 원소에 대한 편미분을 진행.

간단한 신경망을 예로 실제 기울기를 구하는 코드

![](/assets/bamboo_1/chapter345/chapter4/18.png)

- simpleNet 클래스 : 2x3 가중치 매개변수 하나를 인스턴스 변수로 갖음
- predict(x) : 예측 수행
- loss(x,t) : 손실 함수의 값을 구함
- x : 입력 데이터, t : 정답 레이블

![](/assets/bamboo_1/chapter345/chapter4/19.png)

- dW : 2x3의 2차원 배열
    - 첫번째 원소의 기울기 : 대략 0.09
        - 즉, w11을 h만큼 늘리면 손실 함수의 값은 0.09h 만큼 증가
            - 따라서, 음의 방향으로 갱신
    - 마지막 원소의 기울기 : 대략 -0.7
        - 즉, w23을 h만큼 늘리면 손실 함수의 값은 0.7h만큼 감소
            - 따라서, 양의 방향으로 갱신

### 학습 알고리즘 구현하기

전제

신경망에는 적응 가능한 가중치와 편향이 있고, 이 매개변수들을 훈련데이터에 적응하도록 조정하는 과정

→ 학습

1단계 - 미니배치

훈련 데이터 중 일부를 무작위로 가져옴 → 미니배치

미니배치의 손실 함수 값을 줄이는 것이 목표

2단계 - 기울기 산출

미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구함

→ 기울기는 손실 함수의 값을 줄이는 방향으로 제시

3단계 - 매개변수 갱신

가중치 매개변수를 기울기 방향으로 조금씩 갱신

4단계 - 반복

1~3단계 반복

미니배치를 무작위로 선정 → 확률적 경사 하강법

위의 과정을 MNIST 데이터셋을 활용하여 시도

![](/assets/bamboo_1/chapter345/chapter4/20.png)

![](/assets/bamboo_1/chapter345/chapter4/image20.png)

<aside>

TwoLayerNet 클래스:

- params, grad (딕셔너리) 를 인스턴스 변수로 받음
    - params 변수에는 신경망에 필요한 매개변수가 모두 저장
</aside>

예시 :

![](/assets/bamboo_1/chapter345/chapter4/21.png)

예측 처리 :

![](/assets/bamboo_1/chapter345/chapter4/22.png)

![](/assets/bamboo_1/chapter345/chapter4/23.png)

numerical_gradient() 메서드를 사용하여 기울기를 계산 → grads 변수에 기울기 정보가 저장

![](/assets/bamboo_1/chapter345/chapter4/24.png)

<aside>

TwoLayerNet

- __init__(self,input_size,hidden_size,output_size)
    - 클래스를 초기화
        - 입력층의 뉴런 수, 은닉층 개수, 가중치 매개변수도 초기화
- loss(self,x,t)
    - 손실 함수의 값을 계산
- numerical_gradient(self,x,t)
    - 각 매개변수의 기울기를 계산
    - 수치 미분 방식으로 각 매개변수의 손실 함수에 대한 기울기 계산
</aside>

### 미니배치 학습 구현하기

![](/assets/bamboo_1/chapter345/chapter4/25.png)

- 미니배치 크기 : 100
    - 60,000개의 훈련 데이터에서 임의로 100개의 데이터를 추려냄
    - 100개의 미니배치를 대상으로 확률적 경사 하강법을 수행 → 매개변수 갱신
    - 갱신 횟수 : 10,000번

![](/assets/bamboo_1/chapter345/chapter4/image21.png)

- 손실 함수 값의 추이를 살펴보면 반복 횟수가 증가할때마다 손실 함수의 값이 줄어드는 것을 볼 수 있다.

## 시험 데이터로 평가하기 (Test_data)

위의 그래프를 통해 학습을 반복함으로써 손실 함수의 값이 서서히 내려가는 것을 확인

→ ‘훈련데이터’의 미니배치에 대한 손실 함수

다른 데이터셋에서도 비슷한 실력을 발휘하는지 확인해야함!

![](/assets/bamboo_1/chapter345/chapter4/26.png)

- 1에폭 (전체 데이터 학습 횟수)마다 훈련 데이터와 시험 데이터에 대한 정확도 계산

![](/assets/bamboo_1/chapter345/chapter4/image22.png)

- 학습데이터 정확도는 실선, 시험 데이터 정확도는 점선으로 표현
- 에폭(전체 데이터 학습)이 진행될수록 2개의 그래프 모두 정확도가 좋아지고 있음, 정확도 차이 없음
- 과적합 (Overfitting) 이 일어나고 있지 않음!

<aside>


## Chapter_4 정리

- 기계학습에서 훈련 데이터/ 시험 데이터 나누어 사용
- 시험 데이터로 학습한 모델의 범용 능력을 확인 (과적합 방지)
- 손실 함수를 지표로, 손실 함수의 값이 작아지는 방향으로 가중치 매개변수를 갱신
- 가중치 매개변수 갱신
    - 가중치 매개변수의 기울기 사용, 기울어진 방향으로 가중치의 값을 갱신하는 작업 반복
- 아주 작은 값을 주었을 때의 차분으로 미분하는 것 → 수치 미분
- 수치 미분을 이용해 가중치 매개변수의 기울기를 구할 수 있다.
</aside>


# Chapter_5

### 오차역전파법

- 가중치 매개변수의 기울기를 효율적으로 계산할 수 있음
- 수식을 통한 이해
- 계산 그래프를 통한 이해

## 5.1 계산그래프

### 1. 계산 그래프로 풀기

- 문제 1
    - 슈퍼에서 1개에 100원인 사과 2개를 샀다. 이때 지불 금액을 구해보자 (소비세10%)

![](/assets/bamboo_1/chapter345/chapter5/image.png)

- 문제 2
    - 슈퍼에서 사과 2개, 귤을 3개 샀다. 사과는 1개에 100원, 귤은 1개에 150원이다. 소비세 10%

![](/assets/bamboo_1/chapter345/chapter5/image1.png)

- 덧셈 노드인 ‘+’가 새로 등장하여 사과와 귤의 금액을 합산

<aside>

계산 그래프를 이용한 문제풀이

1. 계산 그래프를 구성한다.
2. 그래프에서 계산을 왼쪽에서 오른쪽으로 진행 **(순전파) 그것을 반대로 오른쪽에서 왼쪽 → (역전파)**
</aside>

### 2. 국소적 계산

계산 그래프의 특징

- 국소적 계산을 전파함으로써 최종 결과를 얻음
- 국소적?
    - 자신과 직접 관계된 작은 범위

![](/assets/bamboo_1/chapter345/chapter5/image2.png)

- 해당 그림에서 4000원이 나온 과정은 국소적 계산을 통해 산출됨!

즉, 각 노드는 자신과 관련한 계산 외에는 아무것도 신경 쓸 것이 없음!

전체 계산이 아무리 복잡하여도, 각 단계에서 하는 일은 해당 노드의 ‘국소적 계산’이다.

### 3. 왜 계산 그래프로 푸는가?

계산 그래프의 이점 → ‘국소적 계산’

- 전체적인 계산이 복잡해도 각 노드에서는 단순한 계산에 집중하여 문제를 단순화할 수 있다.
- 계산 그래프는 중간 계산 결과를 모두 보관할 수 있다.

<aside>

실제 계산 그래프를 사용하는 가장 큰 이유

**→ 역전파를 통해 ‘미분’을 효율적으로 계산할 수 있음!**

</aside>

**‘사과 가격에 대한 지불 금액의 미분’**

→ 문제에 적용해보면, 사과 값이 어떻게 변할때 내가 내야하는 최종 금액은 어떻게 변하는가?

![](/assets/bamboo_1/chapter345/chapter5/image3.png)

역전파는 ‘국소적 미분’을 전달하고 그 미분 값은 화살표 아래에 적는다.

‘시과 가격에 대한 지불 금액의 미분’ = 2.2

즉, 사과가 1원 오르면 최종 금액은 2.2원 오른다.

## 5.2 연쇄법칙

역전파 → ‘국소적인 미분’(오른쪽에서 왼쪽으로 전달)

**연쇄법칙**

- 국소적 미분을 전달하는 원리

![](/assets/bamboo_1/chapter345/chapter5/image4.png)

다음과 같은 y=f(x)라는 계산의 역전파를 그려본다.

- 신호 E에 노드의 국소적 미분을 곱한 후 다음 노드로 전달
- 국소적 미분 : 순전파 떄의 y=f(x)계산의 미분을 구한다는 것, 이는 x에 대한 y의 미분을 구한다는 뜻(dy/dx)
- 다음과 같은 국소적인 미분을 상류에서 전달된 값에 곱해 앞쪽 노드로 전달

### **연쇄법칙이란?**

합성함수부터 알아야한다.

합성함수

- 여러 함수로 구성된 함수

![](/assets/bamboo_1/chapter345/chapter5/image5.png)

- z = (x+y)^2 이라는 식은 다음의 2개의 식으로 구성되어있다.
- 합성 함수의 미분은 합성 함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다.

<aside>

(x에 대한 z의 미분) = (t에 대한 z의 미분) * (x에 대한 t의 미분)

</aside>

![](/assets/bamboo_1/chapter345/chapter5/image6.png)

![](/assets/bamboo_1/chapter345/chapter5/image7.png)

![](/assets/bamboo_1/chapter345/chapter5/image8.png)

- 3개의 그림 순서와 같이 연쇄법칙을 사용하여 미분 dz/dx를 구할 수 있다.
- 위의 2개의 미분을 곱하여 구하고 싶은 dz/dx를 구한다.

![](/assets/bamboo_1/chapter345/chapter5/image9.png)

### 연쇄법칙과 계산 그래프

![](/assets/bamboo_1/chapter345/chapter5/image10.png)

- 다음의 사진과 같이 계산 그래프의 역전파는 오른쪽에서 왼쪽으로 신호를 전파한다.
- 노드로 들어온 입력 신호에 그 노드의 국소적 미분(편미분)을 곱한 후 다음 노드로 전달

<aside>

마지막 노드인 **2 노드에서의 역전파 예시

- 입력은 dz/dz, 여기에 국소적 미분인 dz/dt를 곱하고 다음 노드로 넘긴다.
- 여기서 왜 dz/dt냐면, 해당 노드의 입력은 t이고, 출력은 z이기 때문이다. (t가 변할때 z가 어떻게 변하는지?)

맨 왼쪽 역전파

- 연쇄법칙에 따른 계산
    
    ![](/assets/bamboo_1/chapter345/chapter5/image11.png)
    
- 즉 ‘x에 대한 z의 미분이 된다’

→ 역전파 = 연쇄법칙의 원리

![](/assets/bamboo_1/chapter345/chapter5/image12.png)

![](/assets/bamboo_1/chapter345/chapter5/image13.png)

</aside>

앞서 살펴본 식을 해당 계산 그래프에 대입

![](/assets/bamboo_1/chapter345/chapter5/image14.png)

- dz/dx는 2(x+y)임을 구할 수 있다.

## 5.3 역전파

덧셈 노드의 역전파

z=x+y라는 식을 대상으로 살펴본다.

![](/assets/bamboo_1/chapter345/chapter5/image15.png)

여기서 각각의 미분값은 모두 1이 된다.

![](/assets/bamboo_1/chapter345/chapter5/image16.png)

- 상류에서 전해진 미분 값을 dL/dz라고 하고, 이는 L이라는 값을 출력하는 큰 계산 그래프를 가정
- 하류로 각각 dL/dx와 dL/dy를 전달한다.

![](/assets/bamboo_1/chapter345/chapter5/image17.png)

만약 상류에서 1.3이라는 값이 흘러온다면,

이렇게 1.3이라는 값이 다음 노드로 전달된다.

**곱셈 노드의 역전파 (z=xy)**

해당 식의 미분 :

![](/assets/bamboo_1/chapter345/chapter5/image18.png)

![](/assets/bamboo_1/chapter345/chapter5/image19.png)

- 위의 그림을 살펴보면 순전파때는 x를 곱했던 노드가 역전파때는
서로 바꾼 값인 y로 바뀐 것을 확인할 수 있다.

### 사과 쇼핑의 예

변수 : 사과의 가격, 사과의 개수, 소비세

즉, 사과 가격에 대한 지불 금액의 미분, 사과 개수에 대한 지불 금액의 미분, 소비세에 대한 지불 금액의 미분

![](/assets/bamboo_1/chapter345/chapter5/image20.png)

- 사과 가격의 미분 : 2.2
- 사과 개수의 미분 : 110
- 소비세의 미분 : 200
- 즉, 소비세와 사과 가격이 같은 양만큼 오르면, 최종 금액에는 소비세가 200의 크기로, 사과 가격이 2.2크기로 영향을 준다.

사과와 귤 쇼핑 예시

![](/assets/bamboo_1/chapter345/chapter5/image21.png)

## 5.4 단순한 계층 구현하기

‘사과 쇼핑’의 예시를 파이썬으로 구현

곱셈 노드 : mulLayer

덧셈 노드 : AddLayer

- 모든 계층은 forward() 와 backward()라는 공통의 메서드를 갖도록 구현
- forward()는 순전파, backward()는 역전파

![](/assets/bamboo_1/chapter345/chapter5/1.png)

__init __() : 인스턴스 변수인 x,y 초기화

forward () : x와 y를 인수로 받고 두 값을 곱해서 반환

backward () : 상류에서 넘어온 미분 (dout)에 순전파 떄의 값을 ‘서로 바꿔’ 곱한 후 하류로 흘림

![](/assets/bamboo_1/chapter345/chapter5/image22.png)

- 각 변수에 대한 미분은 backward()에서 구할 수 있다.

![](/assets/bamboo_1/chapter345/chapter5/2.png)

- backward() 호출 순서는 forward() 때와 반대이다.
- backward() 가 받는 인수는 **순전파의 출력에 대한 미분**이다.
- mul_apple_layer : 순전파 때는 apple_price 출력, 
                               역전파 때는 apple_price의 미분 값인 dapple_price 출력

### 덧셈 계층

![](/assets/bamboo_1/chapter345/chapter5/3.png)

- 덧셈 계층에서는 초기화가 필요 없음 ( __init __() : pass)
- 덧셈 계층의 forward() : 입력받은 두 인수 x,y를 더해서 반환
- 덧셈 계층의 backward() : 상류에서 내려온 미분(dout)을 그대로 하류로 흘림

![](/assets/bamboo_1/chapter345/chapter5/image23.png)

![](/assets/bamboo_1/chapter345/chapter5/4.png)

- 필요한 계층을 만들고, 순전파 메서드인 forward()를 적절한 순서로 호출
- 순전파와 반대 순서로 역전파 메서드인 backward()를 호출하면 원하는 미분 도출 가능

## 5.5 활성화 함수 계층 구현하기

ReLU , Sigmoid

**ReLU()**

![](/assets/bamboo_1/chapter345/chapter5/image24.png)

![](/assets/bamboo_1/chapter345/chapter5/image25.png)

- 순전파 때의 입력인 x가 0보다 크면 역전파는 상류의 값을 그대로 하류로 흘린다.
- 순전파 때 x 가 0 이하면 역전파 때는 하류로 신호를 보내지 않는다. (0을 보낸다.)

![](/assets/bamboo_1/chapter345/chapter5/image26.png)

ReLU 계층 구현

![](/assets/bamboo_1/chapter345/chapter5/image27.png)

- Relu 클래스는 mask라는 인스턴스 변수를 가진다.
- mask : True/False 로 구성된 넘파이 배열, 순전파의 입력인 x의 원소 값이 0 인하인 인덱스는 True, 그 외는 False로 유지
- 순전파 때의 입력값이 0 이하 → 역전파 때의 값은 0
- 역전파 때는 순전파 때 만들어둔 mask를 써서 mask의 원소가 True인 곳에는 상류에서 전파된 dout을 0으로 설정

**Sigmoid 계층**

![](/assets/bamboo_1/chapter345/chapter5/image28.png)

![](/assets/bamboo_1/chapter345/chapter5/image29.png)

**1 단계 ( / 노드 )**

y=1/x를 미분하면 

![](/assets/bamboo_1/chapter345/chapter5/image30.png)

다음 식이 되고, 

- 역전파 때는 상류에서 흘러온 값에 -y^2(순전파의 출력을 제곱한 후 마이너스를 붙인 값)을 곱하여 
하류로 전달

![](/assets/bamboo_1/chapter345/chapter5/image31.png)

**2 단계 ( + 노드 )**

- 상류의 값을 여과 없이 하류로 보냄

![](/assets/bamboo_1/chapter345/chapter5/image32.png)

**3 단계 ( ‘exp’ 노드, y=exp(x) 수행 )**

![](/assets/bamboo_1/chapter345/chapter5/image33.png)

![](/assets/bamboo_1/chapter345/chapter5/image34.png)

계산 그래프에서는 상류의 값에 순전파 떄의 출력을 곱해 하류로 전파

**4 단계 ( ‘x’ 노드 )**

- 순전파 떄의 값을 ‘서로 바꿔’ 곱한다

![](/assets/bamboo_1/chapter345/chapter5/image35.png)

- 최종 출력 값인 dL/dy(y^2exp(-x)) 이 하류 노드로 전파된다.
- 순전파의 입력 x와 출력 y만으로 계산할 수 있다.

![](/assets/bamboo_1/chapter345/chapter5/image36.png)

![](/assets/bamboo_1/chapter345/chapter5/image37.png)

위의 2개의 사진과 같이 간소화해서 표현할 수 있다.

즉, sigmoid 계층의 역전파는 순전파의 출력(y)만으로 계산할 수 있다.

- sigmoid 계층 python 구현

![](/assets/bamboo_1/chapter345/chapter5/image38.png)

## 5.6 Affine/Softmax 계층 구현하기

### Affine 계층

신경망의 순전파 → 가중치 신호의 총합을 계산하기 위해 **행렬의 곱** 사용

- 가중치의 합 : Y = np.dot( X, W ) + B
- 가중치의 합을 구할때 대응하는 차원의 원소 수를 일치시키는게 핵심!

![](/assets/bamboo_1/chapter345/chapter5/image39.png)

- 지금까지의 계산은 노드에 스칼라값이 흘렀지만, 지금은 ‘행렬’이 흐르고 있다.

![](/assets/bamboo_1/chapter345/chapter5/image40.png)

- Wt는 전치행렬을 뜻하고, 해당 식을 바탕으로 계산 그래프의 역전파를 구해보면
    
    ![](/assets/bamboo_1/chapter345/chapter5/image41.png)
    
- 계산 그래프의 형상을 주의해서 살펴본다

![](/assets/bamboo_1/chapter345/chapter5/image42.png)

### 배치용 Affine 계층

여태까지의 Affine 계층 → 입력데이터로 X 하나만을 고려한 것

- 데이터 N개를 묶어 순전파하는 경우 ( 배치용 Affine 계층에 대해 생각 )

![](/assets/bamboo_1/chapter345/chapter5/image43.png)

기존과 다른 부분 

- 입력인 X의 형상이 (N,2)가 됨
- 역전파 때 행렬의 형상에 주의하면 dL/dX, dL/dW는 이전과 같이 도출 가능
- 편향의 역전파
    - N의 개수에 따라 편향은 각각 더해주어야한다.
    - N개의 데이터에 대한 미분을 데이터마다 더해서 구함

Affine 구현

![](/assets/bamboo_1/chapter345/chapter5/5.png)

Softmax-with-Loss 계층

소프트맥스 함수 → 입력 값을 정규화하여 출력

![](/assets/bamboo_1/chapter345/chapter5/image44.png)

- 입력 이미지가 Affine, ReLU 계층을 통과하여 변환
- 마지막 Softmax 계층에 의해서 10개의 입력이 정규화된다.

손실 함수인 교차 엔트로피 오차도 포함하여 ‘Softmax-with-Loss 계층’ 이라는 이름으로 구현

![](/assets/bamboo_1/chapter345/chapter5/image45.png)

- 소프트맥스 함수 : Softmax 계층
    - 입력 a1,a2,a3 를 정규화하여 y1,y2,y3 출력
- 교차 엔트로피 오차 : Cross Entropy Error
    - Softmax의 출력 y1,y2,y3와 정답 레이블 t1,t2,t3를 받고 손실 L 출력
- 3클래스 분류를 가정하고 이전 계층에서 3개의 입력을 받는다.

Softmax 계층의 역전파

- y1-t1, y2-t2, y3-t3 라는 결과를 내어줌
    - Softmax 출력 - 정답 레이블

→ 신경망의 현재 출력과 정답 레이블의 오차를 드러냄

![](/assets/bamboo_1/chapter345/chapter5/image46.png)

## **5.7 오차역전파법 구현하기**

<aside>

신경망 학습의 전체 과정

전제

- 신경망에는 적응 가능한 가중치와 편향이 존재
- 해당 가중치와 편향이 훈련 데이터에 적응하도록 조정하는 과정 : ‘학습’

**1 단계 - 미니배치**

- 훈련 데이터 중 일부를 무작위로 가져옴 ⇒ 미니배치
- 미니배치의 손실 함수 값을 줄이는 것이 목표

**2 단계 - 기울기 산출**

- 미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기 구함
- 기울기 : 손실 함수의 값을 가장 작게 하는 방향을 제시

**3 단계 - 매개변수 갱신**

- 가중치 매개변수를 기울기 방향으로 아주 조금 갱신

**4 단계 - 반복**

- 1~3단계 반복
</aside>

오차역전파법 → 2단계 (기울기 산출)

| 인스턴스 변수 | 설명 |
| --- | --- |
| params | 딕셔너리 변수, 매개변수 보관
params[’W1’]= 1번째 층의 가중치
params[’b1’]= 1번째 층의 편향 |
| layers | 순서가 있는 딕셔너리 변수, 신경망의 계층 보관 |
| lastLayers | 신경망의 마지막 계층 ( SoftmaxWithLoss 계층 ) |

| **메서드** | 설명 |
| --- | --- |
| __init__(self,input_size,hidden_size,output_size,weight_init_std) | 초기화 수행
입력층 뉴런 수, 은닉층 뉴런 수, 출력층 뉴런 수,
 가중치 초기화 시 정규분포의 스케일 |
| predict(self,x) | 예측 수행 |
| loss(self,x,t) | 손실 함수의 값을 구함 |
| accuracy(self,x,t) | 정확도를 구함 |
| numerical_gradient(self,x,t) | 가중치 매개변수의 기울기를 수치 미분 방식으로 구함 |
| pradient(self,x,t) | 가중치 매개변수의 기울기를 오차역전파법으로 구함 |

![](/assets/bamboo_1/chapter345/chapter5/6.png)

![](/assets/bamboo_1/chapter345/chapter5/7.png)

- 신경망의 계층을 OrderedDict에 보관하는 과정이 중요
    - OrderedDict → 순서가 있는 딕셔너리
    - ‘순서가 있다’ : 딕셔너리에 추가한 순서를 기억한다는 의미
- 역전파 → 계층을 반대 순서로 호출하기만 하면된다.
- 따라서, 계층을 올바른 순서로 연결한 다음 순서대로 (혹은 역순으로) 호출해주면 끝이다.

### 오차역전파법으로 구한 기울기 검증하기

기울기를 구하는 방법

1. 수치 미분을 써서 구하는 방법
    - 수치 미분은 오차역전파법을 정확히 구현했는지 확인하기 위해 필요
2. 해석적으로 수식을 풀어 구하는 방법
    - 오차역전파법을 이용하여 매개변수가 많아도 효율적으로 계산가능

수치 미분의 결과 - 오차역전파법의 결과

⇒ 비교하여 오차역전파법을 제대로 구현했는지 검증

→ 기울기 확인 (2가지 방식의 기울기가 일치하는지 확인하는 작업)

![](/assets/bamboo_1/chapter345/chapter5/image47.png)

- MNIST 데이터셋을 읽고, 훈련 데이터 일부를 수치 미분으로 구한 기울기와 오차역전파법으로 구한 기울기의 오차 확인

### 오차역전파법을 사용한 학습 구현

![](/assets/bamboo_1/chapter345/chapter5/8.png)

## 5.8 정리

<aside>

- 계산 그래프를 이용하면 계산과정을 시각적으로 파악할 수 있다.
- 계산 그래프의 노드는 국소적 계산으로 구성된다.
- 계산 그래프의 순전파는 통상의 계산을 수행한다.
- 신경망의 구성 요소를 계층으로 구현하여 기울기를 효율적으로 계산할 수 있다. (오차역전파법)
- 수치 미분과 오차역전파법의 결과를 비교하여 오차역전파법의 구현에 잘못이 없는지 확인할 수 있다. (기울기 확인)
</aside>
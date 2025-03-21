---
layout: post
title: "DeepLearning_스터디_Chapter 1&2"
date: 2025-03-07 14:00:00 +0900
categories: Study,DeepLearning
tags : DeepLearning,Python
---




# Chapter_1

파이썬은 과학 분야, 특히 기계학습과 데이터 과학 분야에서 널리 쓰인다.

- 딥러닝 프레임워크 쪽에서도 파이썬 사용 (파이썬용 API 제공)
    - 카페 Caffe
    - 텐서플로 TenserFlow
    - 체이너 Chainer
    - 테아노 Theano

## Python

- Python은 동적 언어 프로그래밍 언어이다.
    - 동적?
        - 변수의 자료형을 상황에 맞게 자동으로 결정

| 항목      | 설명 |
|----------|------|
| **산술 연산** | `+`, `-`, `*`, `/`, `**` 등 |
| **자료형 (클래스)** | `type()`을 사용하여 자료형 확인 <br> (정수, 실수, 문자열 등) |
| **변수** | 변수를 사용하여 계산 가능, 다른 값 대입 가능 <br> `x = 10`, `print(x)` |
| **리스트** | 여러 데이터를 리스트로 정리 <br> `a[0]`처럼 각 원소에 접근 <br> **슬라이싱**으로 원하는 부분 추출 가능 <br> → `a[0:2]` (인덱스 0부터 2까지, `{2는 포함 X}`) |
| **딕셔너리** | `(키 key)`와 `(값 value)`을 한 쌍으로 저장 |
| **bool** | `True (1) / False (0)` <br> (`and`, `or`, `not` 연산자 사용 가능) |
| **if 문** | 조건에 따라 처리 수행 |
| **for 문** | 반복(`루프`) 처리 |
| **함수** | 명령들을 묶어 함수로 정의 가능 <br> - 인수를 전달하면 함수는 매개변수로 받아서 처리 |

### 클래스

- 클래스 내의 전용 함수(메서드), 속성
    - 메서드의 첫 번째 인수로 자신을 나타내는 self 사용
- 생성자
    - 클래스를 초기화하는 방법 정의

```python
class Man:
    def __init__(self, name): # 생성자 (클래스 초기화 방법 정의)
        self.name=name
        print("Initialized!")
    
    def hello(self): # 메서드 1
        print("Hello"+self.name+"!")
    
    def goodbye(self): # 메서드 2
        print("Good-bye "+self.name+"!")
m=Man("David")
m.hello()
m.goodbye()

>>> 
$ python man.py
Initialized!
Hello David!
Good-bye David!
```

- m이라는 객체를 생성하고, Man 클래스에 David라는 name을 인수로 전달
- 생성자 내에서 인스턴스 변수인 self.name을 초기화
- 클래스 내에 있는 각 메서드를 호출

### Numpy __ import numpy as np

- 사용하기 편리한 메서드가 많이 준비되어 있는 배열 클래스
- 수치 계산용 라이브러리

| 항목 | 설명 |
|---|---|
| **넘파이 배열 생성** | `x = np.array([1.0, 2.0, 3.0])` <br> `y = np.array([2.0, 4.0, 6.0])` <br> `type(x)  # <class ‘numpy.ndarray’>` |
| **넘파이의 산술 연산** | 배열의 원소 수가 같을 때, 원소별 연산 수행 <br> `x + y  → array([3., 6., 9.])` <br> `x - y  → array([-1., -2., -3.])` <br> `x * y  → array([2., 8., 18.])` <br> `x / y  → array([0.5, 0.5, 0.5])` |
| **스칼라와의 연산 (브로드캐스트)** | 넘파이 배열과 단일 값(스칼라) 연산 시, 모든 원소에 적용됨 <br> `x / 2.0  → array([0.5, 1., 1.5])` |
| **넘파이의 N차원 배열** | `A = np.array([[1, 2], [3, 4]])` <br> `A.shape  # (2,2)` <br> 같은 형상의 행렬끼리 연산 가능하며, 브로드캐스트도 적용 |
| **원소 접근** | `X[0][1]`과 같이 각 원소의 위치로 접근 가능 <br> `for` 문을 이용한 원소 접근 가능 <br> `x.flatten()`을 사용해 1차원 배열로 변환 (평탄화) |

**브로드캐스트**

📌 차원이 다른 배열간에도 연산이 가능하도록 자동으로 형태를 맞춰주는 기능

- 만약 앞서 만든 2X2 배열에 스칼라값 10을 곱하고 싶다면 2x2 배열과 스칼라값 1개가 곱해지는 것이 아닌,
숫자 ‘10’으로 구성된 2x2배열이 생성되어 연산이 이루어지는 것이다.

![](/assets/bamboo_1/1.png)

- 아래의 그림처럼 2차원 배열과 1차원 배열을 곱할때에도 마찬가지로 브로드캐스팅이 작동한다.

![](/assets/bamboo_1/2.png)

### matplotlib

- 그래프를 그려주는 라이브러리
- pyplot 모듈 사용

```python
import numpy as np
import matplotlib.pyplot as plt

x=np.arange(0,6,0.1) # arange로 0부터 6까지 0.1간격으로 생성
y1=np.sin(x)
y2=np.cos(x)

plt.plot(x,y1,label='sin')
plt.plot(x,y2,linestyle="--",label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
```

![](/assets/bamboo_1/3.png)

- 이미지를 표시해주는 imshow()도 존재
- matplotlib.image 모듈의 imread() 메서드 이용

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

img=imread('cactus.png')
plt.imshow(img)
plt.show()
```


# Chapter_2

## 퍼셉트론_perceptron

퍼셉트론

- 신경망 (딥러닝)의 기원이 되는 알고리즘
- 다수의 신호를 입력으로 받아 하나의 신호를 출력
- 퍼셉트론 신호는 ‘흐른다/안 흐른다’(0 or 1)의 두 가지 값을 가질 수 있다.

![](/assets/bamboo_1/image%200.png)

- 입력으로 2개의 신호를 받은 퍼셉트론의 예시
- x1,x2는 입력 신호, w1,w2는 가중치를 뜻한다.
- 그림의 원 → 뉴런 (노드)
- 입력신호가 뉴런에 보내질 때는 각각 고유한 가중치가 곱해진다.
- 뉴런에서 보내온 신호의 총합이 정해진 한계를 넘어설 때만 1을 출력
    - 정해진 한계 → 임계값 θ

![](/assets/bamboo_1/image%201.png)

- 퍼셉트론 동작원리의 수식 표현

### 퍼셉트론을 활용한 문제

1. AND 게이트

![](/assets/bamboo_1/image%202.png)

- 만약 해당 AND 진리표를 표현하고 싶다면 각 입력값에 해당하는 가중치, 그리고 임계값을 정해야한다.
    - ex) (0.5, 0.5, 0.7), (0.5, 0.5, 0.7)과 같이 x1, x2 모두 1일때만 가중 신호의 총합이 주어진 임계값을 웃돈다.
1. NAND 게이트 

![](/assets/bamboo_1/image%203.png)

- AND 게이트에 NOT 을 붙인 것으로써, 앞서 본 AND 게이트의 진리표와 반대이다.
    - ex) (-0.5,-0.5,-0.7) 과 같은 조합이 있다. ( AND 게이트를 구현하는 매개변수의 부호를 모두 반전하면 얻을 수 있다.
1. OR 게이트

![](/assets/bamboo_1/image%204.png)

- x1 과 x2중  1개 이상이 1이면 출력이 1이 된다.
    - ex) (1,1,1)이  예시로 있다.

이렇게 AND, NAND, OR 논리 회로를 표현할 수 있다.

📌 여기서 퍼셉트론의 구조는 3가지의 게이트에서 모두 같았다.

### 퍼셉트론 구현하기

- 위에서 해본 AND 게이트 구현 함수를 작성해보고, -θ를 b(편향)로 치환하여 넘파이로 구현해본다.

```python
import numpy as np

x=np.array([0,1]) # 입력
w=np.array([0.5,0.5]) # 가중치
b=-0.7

print(np.sum(w*x)+b)
>>>-0.19999999999999996
```

‘가중치와 편향을 도입한 AND 게이트 구현’

```python
**import numpy as np

def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=-0.7
    tmp=np.sum(x*w)+b
    if tmp<=0:
        return 0
    elif tmp>0:
        return 1**
```

- 편향은 가중치 w1,w2와 기능이 다르다.
    - 가중치 w1,w2는 각 입력 신호가 결과에 주는 영향력(중요도)을 조절하는 매개변수
    - 편향(b)은 뉴런이 얼마나 쉽게 활성화하느냐를 조정하는 매개변수
        - 만약 편향이 -20.0이면 입력값에 가중치를 곱한 값들의 합이 20.0을 넘지 않으면 뉴런은 활성화되지 않는다. (정해진 임계값을 넘을때 1출력)

NAND 게이트와 OR 게이트 구현

```python
def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    tmp=np.sum(x*w)+b
    if tmp<=0:
        return 0
    elif tmp > 0:
        return 1

def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.2
    tmp=np.sum(x*w)+b
    if tmp<=0:
        return 0
    elif tmp>0:
        return 1
```

## 퍼셉트론의 한계

AND, NAND, OR 3가지 논리 회로를 구현할 수 있었다.

### XOR 게이트

- XOR 게이트는 배타적 논리합이라는 논리 회로 (배타적 → 자기 외에는 거부한다)
- x1과 x2 중 한쪽이 1일 때만 1을 출력한다.

![](/assets/bamboo_1/image%205.png)

이와 같은 XOR 게이트를 구현하기 위해선 가중치와 편향을 어떻게 구현해야할까?

![](/assets/bamboo_1/image%206.png)

- 1로 표현해야하는 값들을 세모로 표시하고, 0의 값들을 동그라미로 표현했다면 다음과 같은 그래프를 어떻게 직선 1개로 나눌 수 있을까? → 불가능하다.

만약 직선이라는 조건이 사라진다면? → 구분하는 것이 가능하다.

![](/assets/bamboo_1/image%207.png)

- 다음과 같은 곡선에서 곡선의 영역을 비선형 영역, 직선의 영역을 선형 영역이라고 한다.

### 다층 퍼셉트론_multi-layer perceptron

- 퍼셉트론으로는 XOR 게이트를 표현할 수 없었다.
- 퍼셉트론을 여러 층 쌓아 만든 다층 퍼셉트론 ( 퍼셉트론을 더 쌓아서 ) XOR게이트를 표현해볼 것이다.

앞서 해결해본 AND, NAND, OR 게이트를 조합하여 만들어 볼 것이다.

![](/assets/bamboo_1/image%208.png)

각각의 게이트를 다음과 같이 표현해볼 수 있고, 위의 기호에서 ◦ 는 출력을 반전하는 뜻을 가지고 있다.

![](/assets/bamboo_1/image%209.png)

다음 사진의 ? 에 AND, NAND, OR을 각각 1개씩 대입하여 XOR을 완성할 수 있다.

![](/assets/bamboo_1/image%210.png)

다음과 같이 S1, S2에 각각 NAND, OR 그리고 마지막에 AND 게이트를 조합하면 XOR 게이트를 구현할 수 있다.

| x1 | x2 | s1(NAND) | s2(OR) | y (AND) |
| --- | --- | --- | --- | --- |
| 0 | 0 | 1 | 0 | 0 |
| 1 | 0 | 1 | 1 | 1 |
| 0 | 1 | 1 | 1 | 1 |
| 1 | 1 | 0 | 1 | 0 |

위의 표와 같이 XOR 게이트 (배타적 논리합) 를 구현할 수 있다.

```python
def XOR(x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)
    return y
```

![](/assets/bamboo_1/image%211.png)

XOR는 다음과 같은 다층 구조의 네트워크이다.

위의 사진은 앞서 살펴본 AND, OR, NAND의 퍼셉트론 구조와는 다른걸 확인할 수 있다.

위의 사진은 2층 퍼셉트론으로써 0층에서 1층으로 신호가 전달되고, 이어서 1층에서 2층으로 신호가 전달된다.

<aside>

1. 0층의 두 뉴런이 입력 신호를 받아 1층의 뉴런으로 신호를 보낸다.
2. 1층의 뉴런이 2층의 뉴런으로 신호를 보내고, 2층의 뉴런은 y를 출력한다.
</aside>

📌  앞서 단층 퍼셉트론으로는 표현하지 못한 것을 층을 하나 늘려 구현할 수 있었다.

### Chapter_2 정리

- 퍼셉트론은 입출력을 갖춘 알고리즘, 입력을 주면 정해진 규칙에 따른 값을 출력한다.
- 퍼셉트론에서는 ‘가중치’와 ‘편향’을 매개변수로 설정
- 단층 퍼셉트론은 직선형 영역만 표현할 수 있다. (AND,NAND,OR 게이트)
- 퍼셉트론을 여러층 쌓은 다층 퍼셉트론은 비선형 영역도 표현할 수 있다. (XOR 게이트)

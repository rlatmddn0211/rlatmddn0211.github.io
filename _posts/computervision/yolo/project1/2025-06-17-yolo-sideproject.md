---
layout: post
title: "Object Detection Using YOLOv8"
date: 2025-06-19 14:00:00 +0900
categories: Object Detection
tags: [
    Python, openCV, computer_vision, object_detection, YOLOv8
]
---

# 객체_탐지_실습_YOLOv8

방학 동안 computer_vision 분야에서도 객체 탐지 (Object Detection)에 관심이 생겨서 논문들을 살펴보는 것과 동시에 실제로 내가 간단한 스몰 프로젝트를 만들어 보고자 하였다.

우선 내가 간단하게 객체 탐지 모델을 활용하고자 한 방식은 내 노트북과 연동된 핸드폰 카메라를 통하여 어떤 객체들을 감지하는지 실험해보는 프로젝트였다.

우선 내가 사용한 객체 탐지 모델은 “YOLOv8”이다.

- YOLO의 가장 큰 장점이자 무기는 바로 실시간 탐지에 있다고 생각하며, 그것을 몸소 느껴보기 위해 노트북과 연동된 핸드폰 카메라로 촬영되는 영상의 객체들을 YOLO가 실시간으로 탐지할 수 있을지 확인해보았다.

```python
import cv2
from ultralytics import YOLO

model=YOLO("yolov8n.pt")
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break
    results=model(frame)
    annotated_frame=results[0].plot()
    cv2.imshow("Object Detection",annotated_frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
```

![](/assets/object_detection/project1/image1.png)

- detect_cam.py 의 코드는 다음과 같다.
- 우선 ultralytics에서 제공하는 YOLOv8 API를 임포트해와서 YOLOv8 모델을 활용하였다.
- model = YOLO(”yolov8n.pt”) 를 통해 YOLOv8의 nano 모델 (가장 가벼운 모델)을 불러와 객체 탐지를 수행하게 하였다.
- results = model(frame) 에서는 각 프레임에서 탐지되는 객체들을 바운딩 박스와 클래스 예측을 통하여 반환하도록 설정하였다.
    - results[0].plot() 을 통하여 탐지된 객체의 바운딩 박스와 클래스 예측을 시각적으로 확인할 수 있도록 설정하였다.
- 핸드폰 카메라로 계속해서 촬영을 할 것이기 때문에 while문을 통하여 구현하였으며, cv2.imshow()를 통하여 감지한 객체를 화면에 띄우도록 설정하였다.
- cv2.waitKey(1) & 0xFF==ord("q") 그리고 종료 조건은 q를 입력받는 것으로 설정하였다.

![](/assets/object_detection/project1/image2.png)

![](/assets/object_detection/project1/image3.png)

![](/assets/object_detection/project1/image4.png)

- 이것저것 찍어보았고, 객체들을 생각보다 잘 탐지하는 것 같아서 놀랐다.
- 만약 내가 YOLOv2를 사용하였다면, 지금 탐지되는 객체들은 모두 COCO 데이터셋 + ImageNet 데이터 셋을 통하여 학습된 9000개의 물체 클래스들 중 하나겠지만, YOLOv8을 사용하였기 때문에 COCO 데이터셋에서 추가된 80개의 클래스들 중 하나였다.

## 화면의 사람 수 세어보기

- 클래스 명이 정해져있기 때문에 만약 현재 촬영 중인 화면에서 ‘person’ 이라는 클래스의 객체가 1개 이상 감지되면 그 수를 세서 화면의 사람 수를 셀 수 있을 것 같았다.

```python
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    boxes = results[0].boxes
    classes = boxes.cls.cpu().numpy().astype(int) 
    person_count = sum(1 for c in classes if c == 0)
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"Person Count: {person_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow("Object Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

![](/assets/object_detection/project1/image5.png)

- YOLOv8에서 ‘person’ 이라는 클래스는 0으로 지정되어 있었다. 따라서, 만약 예측된 박스들의 클래스 인덱스 목록이 0이라면, 해당 박스를 ‘person’으로 예측하고 있다는 의미이기 때문에 그 수를 세어준다.
- putText를 통하여 왼쪽 위에 사람 수를 출력하도록 설정하였다.

![](/assets/object_detection/project1/image6.png)

- 나 말고 다른 사람이 없어서 사진으로 대체해보았다. 사진 속 인물까지 count하여 “Person Count :2”로 잘 나오고 있었다.

## 느낀점

- 객체 탐지 분야에 첫 걸음을 내딛은 느낌이다.
- YOLO 모델을 공부한 후, 실제로 적용해보는 것까지 해보니 정말 YOLO가 잘 설계된 객체 탐지 모델이라는 생각이 들었다.
- 왜 YOLO 객체 탐지 모델이 다양한 분야에서 활용되는지 알 것 같다. (접근성이 매우 좋다.)
- 해당 실습 말고도 더 폭넓게 객체 탐지 모델들을 활용해보고싶고, 더 다양한 객체 탐지 모델들을 활용해보고싶다.
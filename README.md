# Crack Detection using YOLOv8 Instance Segmentation
 ONNX 포맷으로 export한 커스텀 트레이닝 모델을 통해 건축물의 균열을 탐지하는 프로그램입니다.

# Requirements

 * **requirements.txt** 을 통해 필요한 패키지를 다운로드합니다.

# 설치
``` python
// 아나콘다 가상환경 접속
conda activate [yourName]

// 패키지 설치
pip install -r requirements.

// 실행
```

# 주요 파일

 * **Image inference**:
 ``` python
 python image_instance_segmentation.py
 ```

 * **Webcam inference**:
 ``` python
 python webcam_instance_segmentation.py
 ```

 * **Webcam Source**:
 ``` python
 // 0: 내장 카메라, 1: 외장 카메라
cap = cv2.VideoCapture(1) // 외장 카메라 이용
 ```


 * **Video inference**:
 ``` python
 python video_instance_segmentation.py
 ```
  *Original video: https://www.youtube.com/shorts/toPt_DejlKM*
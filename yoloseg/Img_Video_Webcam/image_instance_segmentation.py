import cv2
from imread_from_url import imread_from_url

from yoloseg import YOLOSeg         # 사용자 정의 라이브러리

# 사용할 yolo 모델 정의 및 호출
model_path = '/models/yolov8n-seg.onnx'
yoloseg = YOLOSeg(model_path, conf_thres=0.2, iou_thres=0.3)
# conf_thres: 모델이 검출할 객체에 대한 정확도
# iou_thres: 동일 객체로 인식할 기준

# Read image
# img_url = "https://upload.wikimedia.org/wikipedia/commons/e/e6/Giraffes_at_west_midlands_safari_park.jpg"
# 인터넷 링크도 가능
targetImgPath = "./target.jpeg"    # 예측할 이미지 경로
img = cv2.imread(targetImgPath)     # 이미지 로드

# Detect Objects
boxes, scores, class_ids, masks = yoloseg(img)
# 모델에 한 프레임을 전달하고 모델에서 box, 예측 확률, 예측 결과, 세그먼트 정보를 가져옴

# Draw detections
combined_img = yoloseg.draw_masks(img)  # 로드한 이미지에 세그먼트 적용
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)  # 윈도우 설정
cv2.imshow("Detected Objects", combined_img)    # 세그먼트한 이미지 출력
cv2.imwrite("../../doc/img/detected_target_image.jpg", combined_img)  # 세그먼트한 이미지 저장
cv2.waitKey(0)
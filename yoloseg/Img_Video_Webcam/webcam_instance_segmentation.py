import cv2

from yoloseg import YOLOSeg

# Initialize the webcam
cap = cv2.VideoCapture(1)

model_path = "../../models/yolov8n-seg.onnx"  # 모델 경로
yoloseg = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.5)
# conf_thres: 모델이 검출할 객체에 대한 정확도
# iou_thres: 동일 객체로 인식할 기준

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)  # 윈도우 설정
while cap.isOpened():   # 웹캠 연결 이 종료될 때까지

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:     # ret은 비디오의 연결 상태를 나타냄 True면 정상
        break

    # Update object localizer
    boxes, scores, class_ids, masks = yoloseg(frame)
    # 모델에 한 프레임을 전달하고 모델에서 box, 예측 확률, 예측 결과, 세그먼트 정보를 가져옴

    combined_img = yoloseg.draw_masks(frame)
    # 세그먼트 적용된 이미지를 가져옴

    cv2.imshow("Detected Objects", combined_img)
    # 세그먼트 적용된 이미지 디스플레이

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

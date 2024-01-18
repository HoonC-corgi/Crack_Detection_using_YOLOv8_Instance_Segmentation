import cv2
from cap_from_youtube import cap_from_youtube

from yoloseg import YOLOSeg

# # Initialize video
# cap = cv2.VideoCapture("input.mp4")   # 로컬 비디오파일도 가능

videoUrl = 'https://www.youtube.com/shorts/toPt_DejlKM' # 유투브 URL
cap = cap_from_youtube(videoUrl)    # 유투브 영상 로드
start_time = 0  # skip first {start_time} seconds   # 영상 일부 스킵 시 사용
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))
# 영상의 프레임을 가져옴, 시작시간 설정 및 fps 설정

# Initialize YOLOv5 Instance Segmentator
model_path = "../../models/yolov8n-seg.onnx"  # 모델 경
yoloseg = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.3)
# conf_thres: 모델이 검출할 객체에 대한 정확도
# iou_thres: 동일 객체로 인식할 기준

# 로컬 비디오 로드
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter("output.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)      # 윈도우 설정
frame_countdown = 3
while cap.isOpened():   # 프레임이 끝날 시 종료

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    # Read frame from the video
    ret, frame = cap.read()
    if not ret:     # ret은 비디오의 연결 상태를 나타냄 True면 정상 동작
        break

    # Update object localizer
    boxes, scores, class_ids, masks = yoloseg(frame)
    # 모델에 한 프레임을 전달하고 모델에서 box, 예측 확률, 예측 결과, 세그먼트 정보를 가져옴

    combined_img = yoloseg.draw_masks(frame, mask_alpha=0.4)    # 출력 후 이미지, mask_alpha는 색칠되는 세그먼트의 투명도를 결정
    # out.write(combined_img)   # 로컬 비디오 저장
    cv2.imshow("Detected Objects", combined_img)    # 세그먼트 적용된 이미지 디스플레이

cap.release()   # 비디오 해제
# out.release() # 로컬 비디오 해제
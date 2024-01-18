import threading
import queue
import cv2
import hl2ss
from yoloseg import YOLOSeg

# HoloLens 스트림 설정
host = '172.30.1.49'
port_pv = hl2ss.StreamPort.PERSONAL_VIDEO
port_depth = hl2ss.StreamPort.RM_DEPTH_AHAT

# 공유 데이터 큐
data_queue = queue.Queue()

# 카메라 캘리브레이션 데이터 다운로드 및 초점 거리와 주점 추출
calibration_data = hl2ss.download_calibration_pv(host, port_pv, 1920, 1080, 30)
focal_length = calibration_data.focal_length
principal_point = calibration_data.principal_point


# PV 스트림 처리 함수
def process_pv_stream():
    client_pv = hl2ss.rx_decoded_pv(host, port_pv, hl2ss.ChunkSize.PERSONAL_VIDEO, hl2ss.StreamMode.MODE_1, 1920, 1080,
                                    30, hl2ss.VideoProfile.H265_MAIN, 4000000, 'bgr24')
    client_pv.open()

    while True:
        data_pv = client_pv.get_next_packet()
        if data_pv is not None and data_pv.payload.image is not None:
            data_queue.put(('pv', data_pv.payload.image))

    client_pv.close()


# 깊이 스트림 처리 함수
def process_depth_stream():
    client_depth = hl2ss.rx_decoded_rm_depth_ahat(host, port_depth, hl2ss.ChunkSize.RM_DEPTH_AHAT,
                                                  hl2ss.StreamMode.MODE_1, hl2ss.VideoProfile.RAW, 4000000)
    client_depth.open()

    while True:
        data_depth = client_depth.get_next_packet()
        if data_depth is not None and data_depth.payload.depth is not None:
            data_queue.put(('depth', data_depth.payload.depth))

    client_depth.close()


# 메인 스레드 처리 함수
def main_thread():
    yoloseg = YOLOSeg("../../models/yolov8n-seg.onnx", conf_thres=0.7, iou_thres=0.5)

    # 스레드 시작
    threading.Thread(target=process_pv_stream).start()
    threading.Thread(target=process_depth_stream).start()

    frame = None
    depth_frame = None

    while True:
        stream_type, data = data_queue.get()

        if stream_type == 'pv':
            frame = data
        elif stream_type == 'depth':
            depth_frame = data

        if frame is not None and depth_frame is not None:
            # YOLOSeg로 세그멘테이션 수행
            boxes, scores, class_ids, masks = yoloseg(frame)

            # 여기에서 마스크를 그린 이미지를 생성
            combined_img = yoloseg.draw_masks(frame)

            for box, score, class_id, mask in zip(boxes, scores, class_ids, masks):
                x1, y1, x2, y2 = map(int, box)
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)

                # 해상도 불일치 문제에 대한 처리
                x_center = min(x_center, depth_frame.shape[1] - 1)
                y_center = min(y_center, depth_frame.shape[0] - 1)

                depth_at_center = depth_frame[y_center, x_center]

                real_width = abs((x2 - x1) * depth_at_center / focal_length[0]) * 1000  # mm 단위
                real_height = abs((y2 - y1) * depth_at_center / focal_length[1]) * 1000  # mm 단위

                # 객체 정보 표시
                label = f"ID: {class_id}, Score: {score:.2f}, Width: {real_width:.2f}mm, Height: {real_height:.2f}mm"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow('HoloLens Segmentation', combined_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    client_pv.close()
    client_depth.close()
    cv2.destroyAllWindows()


# 메인 스레드 시작
main_thread()

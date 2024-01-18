import numpy as np
import cv2

class_names = ['crack', 'normal']       # 클래스 종류

# 각 클래스 별 색상 임의 지정
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

# nms 비최대 억제 처리 함수
# nms는 여러 겹친 바운딩 박스 중 가장 적합한 것 하나만을 선택하도록 함.
def nms(boxes, scores, iou_threshold):
    # 예측 신뢰도 점수에 따라 인덱스를 내림차순으로 정렬함.
    sorted_indices = np.argsort(scores)[::-1]

    # 유지할 바운딩 박스의 인덱스를 저장할 수 있는 리스트를 생성
    keep_boxes = []

    # 정렬된 인덱스가 남아있을 때까지 반복
    while sorted_indices.size > 0:
        # 가장 높은 신뢰도를 가지는 상자의 인덱스를 선택
        box_id = sorted_indices[0]
        # 임시 저장함.
        keep_boxes.append(box_id)

        # 리스트 내 다른 바운딩 박스와의 IoU 계산
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # IoU 임계값보다 낮은 박스들만 유지해야 함.
        keep_indices = np.where(ious < iou_threshold)[0]

        # 유지할 바운딩 박스들의 인덱스 업데이트
        sorted_indices = sorted_indices[keep_indices + 1]

    # 유지할 바운딩 박스 인덱스
    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # 교차 영역의 면적을 계산함. 음수일 때에는 0으로 처리.
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # 각 바운딩 박스의 면적을 계산
    # box = [ 'x1', 'y1', 'x2', 'y2']
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # 다른 바운딩 박스의 면적
    # 바운딩 박스 간의 합집합 영역의 면적을 계산함. 
    union_area = box_area + boxes_area - intersection_area

    # 교차 영역 면적 / 합집합 영역 면적
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)

    # 바운딩 박스 좌표를 복사하여 새로운 배열을 생성함
    y = np.copy(x)


    # x 중심 좌표에서 너비의 절반을 빼서 x1(왼쪽 상단 x 좌표)를 계산
    y[..., 0] = x[..., 0] - x[..., 2] / 2

    # y 중심 좌표에서 높이의 절반을 빼서 y1(왼쪽 상단 y 좌표)를 계산
    y[..., 1] = x[..., 1] - x[..., 3] / 2

    # x 중심 좌표에 너비의 절반을 더해서 x2(오른쪽 하단 x 좌표)를 계산
    y[..., 2] = x[..., 0] + x[..., 2] / 2

    # y 중심 좌표에 높이의 절반을 더해서 y2(오른쪽 하단 y 좌표)를 계산
    y[..., 3] = x[..., 1] + x[..., 3] / 2

    # 변환된 좌표를 반환합니다.
    return y


def sigmoid(x):
    # Sigmoid 활성화 함수 계산
    # 입력값 x를 받아, 0과 1 사이의 값으로 변환합니다.
    # 식은 1 / (1 + exp(-x)). exp(-x)는 자연 상수 e를 -x의 거듭제곱으로 계산
    return 1 / (1 + np.exp(-x))


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3, mask_maps=None):
    # 이미지의 높이와 너비를 불러옴
    img_height, img_width = image.shape[:2]

    # 텍스트 크기를 이미지 크기에 비례하여 설정
    size = min([img_height, img_width]) * 0.0006

    # 텍스트 두께를 이미지 크기에 비례하여 설정
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # 마스킹 함수 호출
    mask_img = draw_masks(image, boxes, class_ids, mask_alpha, mask_maps)

    # 각 바운딩 박스, 클래스 종류, 예측 신뢰도를 그림.
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        # 바운딩 박스 좌표를 정수로 변환
        x1, y1, x2, y2 = box.astype(int)

        # 박스를 그림.
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

        # 클래스 종류와 예측 신뢰도를 문자열로 결함.
        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        # 텍스트 배경을 그림.
        cv2.rectangle(mask_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)

        # 텍스트를 그림.
        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return mask_img


def draw_masks(image, boxes, class_ids, mask_alpha=0.3, mask_maps=None):
    # 원본 이미지의 복사본 생성하여 마스크 적용 시 사용함
    mask_img = image.copy()

    # 감지된 각 객체에 대하여 반복함
    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        # 클래스 종류에 따른 색상을 가져옴
        color = colors[class_id]

        # 바운딩 박스의 좌표를 정수형으로 변환
        x1, y1, x2, y2 = box.astype(int)

        # 마스크 이미지를 그림
        if mask_maps is None:
            # 마스크 맵이 없을 경우 바운딩 박스를 채워진 사각형으로 그림.
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
        else:
            # 해당 마스크에 색상 적용 후 이미지에 그림.
            crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
            crop_mask_img = mask_img[y1:y2, x1:x2]
            crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
            mask_img[y1:y2, x1:x2] = crop_mask_img

    # 마스크가 적용된 이미지와 원본 이미지를 alpha 값에 따라 합성
    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


def draw_comparison(img1, img2, name1, name2, fontsize=2.6, text_thickness=3):
    # 첫 번째 이미지의 텍스트 크기를 계산
    (tw, th), _ = cv2.getTextSize(text=name1, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=text_thickness)

    # 첫 번째 이미지에 텍스트를 배치할 위치를 계산함
    x1 = img1.shape[1] // 3
    y1 = th
    offset = th // 5

    # 첫 번째 이미지에 텍스트 배경을 그림.
    cv2.rectangle(img1, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (0, 115, 255), -1)
    
    # 첫 번째 이미지에 텍스트 배경을 그림
    cv2.putText(img1, name1,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), text_thickness)

    # 두 번째 이미지의 텍스트 크기를 계산함
    (tw, th), _ = cv2.getTextSize(text=name2, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=text_thickness)

    # 두 번째 이미지에 텍스트를 배치할 위치를 계산
    x1 = img2.shape[1] // 3
    y1 = th
    offset = th // 5

    # 두 번째 이미지에 텍스트 배경을 그림
    cv2.rectangle(img2, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (94, 23, 235), -1)

    # 두 번째 이미지에 텍스트를 그림.
    cv2.putText(img2, name2,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), text_thickness)

    # 두 이미지를 수평으로 연결하여 하나의 이미지로 합침.
    combined_img = cv2.hconcat([img1, img2])
    if combined_img.shape[1] > 3840:
        combined_img = cv2.resize(combined_img, (3840, 2160))

    return combined_img

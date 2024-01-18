import math
import time
import cv2
import numpy as np
import onnxruntime

from yoloseg.utils import xywh2xyxy, nms, draw_detections, sigmoid      # 사용자 정의 함수


class YOLOSeg:

    # 생성자: 모델 경로, 확률 임계값, IOU 임계값 및 마스크 개수 초기화
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, num_masks=32):
        self.conf_threshold = conf_thres    # 예측 신뢰도 임계치 초기화
        self.iou_threshold = iou_thres      # IOU 임계값 초기화
        self.num_masks = num_masks          # 마스크 개수

        # Initialize model
        self.initialize_model(path)         # 모델 로드

    # 객체 호출시 오브젝트 디텍션 및 세그먼트 수행
    def __call__(self, image):
        return self.segment_objects(image)

    # 모델 로드 함수 정의
    def initialize_model(self, path):
        # ONNX 런타임을 이용하여 모델을 초기화, 이때 'CPUExecutionProvider'는 모델이 CPU에서 실행될 것임을 나타냄
        self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.get_input_details()  # 입력 정보 가져오기
        self.get_output_details()  # 출력 정보 가져오기

    # 세그먼트 적용 함수
    def segment_objects(self, image):
        input_tensor = self.prepare_input(image)
        # 모델 입력을 위한 변수 초기화. 이미지를 불러와 batch, channels, height, width 4차원 텐서로 변환

        # Perform inference on the image
        outputs = self.inference(input_tensor)  # 모델에 입력하여 결과 리턴 받음

        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])
        # outputs[0]에 저장된 바운딩박스, 예측 확률, 예측 클래스, 마스크 예측 클래스를 불러옴

        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])
        # outputs[0]에서 불러온 마스크 예측 클래스를 토대로, 데이터를 처리하여 마스크 맵을 생성함.
        # 이때 마스크 맵은 적용된 세그먼트의 결과임.

        return self.boxes, self.scores, self.class_ids, self.mask_maps
        # 바운딩 박스, 예측 신뢰도, 예측 클래스, 마스크 맵을 반환

    # 입력 이미지 준비 함수
    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2] # 이미지 shape에 저장된 높이와 너비를 가져옴

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR 이미지에서 RGB이미지로 변환

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height)) # RGB 이미지 사이즈 조정

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0   # 픽셀 값 조정
        input_img = input_img.transpose(2, 0, 1)
        # 모델에 입력하기 위해 이미지 차원 변경 기존 차원(높이, 너비, 차원) -> 차원(차원, 높이, 너비)로 변환됨. (2, 0, 1)은 기존 차원의 인덱스

        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32) # 모델에 입력하기 위해 4차원 텐서로 변경함
        # 기존  3차원 텐서를 4차원으로 변환함. 이에 따라 0번 차원은 batch_size 표현함. default 1 싱글 배치여도 필요
        # 딥러닝 모델은 부동소수점 연산을 사용하므로, float32로의 데이터 타입 변환

        return input_tensor # 결과 리턴

    # 실제 추론을 실행하는 함수
    def inference(self, input_tensor):
        start = time.perf_counter() # 추론 시작 시간 체크
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        # onnx 런타임 세션으로 실행, output_names는 모델의 출력 노드, input_names는 모델의 입력 노이며 이에 input_tensor가 지정됨.
        # 해당 함수의 결과를 outputs에 저장, 결국 모델의 추론 결과가 저장됨.

        # print(f"추론 시간: {(time.perf_counter() - start)*1000:.2f} ms") # 성능평가를 위한 추론 시간 계산
        return outputs  # 추론 결과 반환

    # 바운딩 박스 처리 함수
    def process_box_output(self, box_output):
        # box_output에는 모델이 추론한 박스 좌표, 에측 정확도 및 예측 클래스 등의 정보가 저장됨.

        # 모델의 출력 배열 형태로 조정함. 바운딩 박스, 예측 신뢰도 및 예측 클래스 등의 정보를 저장
        predictions = np.squeeze(box_output).T
        # np.squeeze를 통해 불필요한 차원을 제거하여 차원 축소 후 차원을 전치하여 재배열하여 데이터를 처리함.

        # 분류된 결과 정보를 저장함.
        # box_output[1]에는 경계 상자의 x,y,w,h 4개 좌표와 예측 신뢰도, 마스크 예측 값 등이 저장되어 있음.
        # 이에 마스크 4개 좌표를 제거함. 즉 num_classes에는 모델이 예측한 클래스를 저장하게 됨.
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        # 예측 신뢰도의 최고 점수를 가져옴.
        # prediction 배열의 첫 네 개의 값 바운드박스의 좌표에 해당하므로 제외함. axis=1에 따라 행에서 가장 큰 값을 가져옴.
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)

        # 설정된 예측 신뢰도 임계치를 넘는 데이터만 가져옴
        # predictions는 오브젝트의 정보를 필터링 -> conf_threshold를 넘지 못하는 데이터 정보는 모두 삭제
        predictions = predictions[scores > self.conf_threshold, :]
        # scores는 신뢰도 점수만을 필터링 -> conf_threshold를 넘지 못하는 신뢰도 정보도 모두 삭제
        scores = scores[scores > self.conf_threshold]
        # 두 코드라인을 함께 씀으로써 두 배열이 서로 일치하는 데이터만을 가지도록 동기화 됨.

        # 필터링 된 예측 결과가 없다면 0이 채워진 배열 반환
        if len(scores) == 0:
            return [], [], [], np.array([])

        # 모델이 예측한 바운딩 박스 좌표 및 예측 신뢰도를 저장
        box_predictions = predictions[..., :num_classes+4]
        # 모델이 예측한 마스크 좌표 및 예측 신뢰도를 저장
        mask_predictions = predictions[..., num_classes+4:]

        # 행을 따라 박스 예측 신뢰도가 가장 높은 클래스만을 추출
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # 각 객체에 대한 바운딩 박스 좌표를 추출
        boxes = self.extract_boxes(box_predictions)

        # nms는 겹치거나 중복되는 바운딩 박스를 제거함으로써 여러 예측이 동일 객체를 가리키는 것을 방지함.
        # 이에 대한 기준으로 iou_threshold가 적용, 0.5로 설정하였으므로, 50%이상 겹치는 박스는 제거됨.
        indices = nms(boxes, scores, self.iou_threshold)

        # nms 처리가 이루어진 바운딩 박스, 신뢰도, 클래스 종류, 마스크 예측 결과를 반환함.
        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    # 마스크 처리 함수
    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def process_mask_output(self, mask_predictions, mask_output):

        # 모델이 예측한 마스크에 대한 정보가 없다면, 빈 배열을 반환
        if mask_predictions.shape[0] == 0:
            return []

        # 마스크 출력을 차원 축소하여 불필요한 정보를 제거함.
        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        # 마스크 출력의 차원 정보를 가져옴
        num_mask, mask_height, mask_width = mask_output.shape

        # 마스크 예측값과 모델의 출력값을 행렬 곱셈(@)하여 최종 마스크를 계산함.
        # 이를 이를 위해 모델의 출력값 차원을 재조정함. num_mask는 mask_output 배열의 첫 번째 차원이며, 채널 수를 의미함.
        # -1 매개변수를 통해 mask_output은 num_mask 매개변수를 가지는 행렬로 재배열 됨.
        # 이를 통해 mask_output은 2차원 배열이며, 각 행은 하나의 마스크를 나타내게 됨
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))

        # 계산된 마스크를 기존의 이미지 크기에 맞게 다시 재조정함. 행렬 계산을 위해 변환 후 다시 재조정한 것임.
        masks = masks.reshape((-1, mask_height, mask_width))

        # 감지된 바운딩 박스 마스크 크기에 맞게 재조정함.
        scale_boxes = self.rescale_boxes(self.boxes,
                                   (self.img_height, self.img_width),
                                   (mask_height, mask_width))

        # 마스크 맵을 초기화
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        # 블러 처리할 영역의 크기를 정함
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))

        # 각 바운딩 박스에 대한 마스크맵 생성
        for i in range(len(scale_boxes)):

            # 바운딩 박스의 스케일 조정된 좌표를 계산함.
            # math.floor는 반내림하여 마스크 경계를 명확히 하기 위함. math.ceil은 반올림
            scale_x1 = int(math.floor(scale_boxes[i][0]))   # x1좌표: 바운딩 박스의 좌상단 x좌표
            scale_y1 = int(math.floor(scale_boxes[i][1]))   # y1좌표: 바운딩 박스의 좌상단 y좌표
            scale_x2 = int(math.ceil(scale_boxes[i][2]))    # x2: 바운딩 박스의 우하단 x좌표
            scale_y2 = int(math.ceil(scale_boxes[i][3]))    # y2: 바운딩 박스의 우하단 y좌표

            # 기존 바운딩 박스의 좌표를 계산함.
            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            # 기존과 신규 좌표를 모두 계산하는 것은 스케일 조정된 마스크 좌표를 기존 바운딩 박스의 크기에 맞게 일치시키기 위함.

            # 각 객체의 스케일 조정된 바운딩 박스(마스크) 좌표 scale_y1:scale_y2, scale_x1:scale_x2를 추출
            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]

            # 추출한 각 마스크의 크기를 원본 바운딩 박스의 크기(x2 - x1, y2 - y1)에 맞게 사이즈 조정함.
            # interpolation=cv2.INTER_CUBIC는 사이즈 조정 시에 사용하는 보간법임.
            crop_mask = cv2.resize(scale_crop_mask,
                              (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_CUBIC)

            # 각 바운딩 박스 크기에 맞게 잘려진 마스크의 경계를 블러 처리하여 부드럽게 함.
            crop_mask = cv2.blur(crop_mask, blur_size)

            # 마스크 영역을 이진화 하여, 마스크 영역을 명확히 함.
            crop_mask = (crop_mask > 0.5).astype(np.uint8)

            # 각 오브젝트의 최종 마스크를 마스크 맵의 해당 위치에 할당함.
            # mask_maps[i, y1:y2, x1:x2은 원본 이미지에 대응하는 영역을 나타냄.
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        # 최종 처리된 마스크맵 반환
        return mask_maps


    # 이미지 위에 감지된 객체의 바운딩 박스, 클래스 종류, 예측 신뢰도를 그리는 함수
    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    # draw_detection에서 마스크 맵도 그리는 확장 함수
    def draw_masks(self, image, draw_scores=True, mask_alpha=0.5):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha, mask_maps=self.mask_maps)

    def get_input_details(self):
        # 모델의 입력 노드 정보를 저장
        model_inputs = self.session.get_inputs()
        # 입력 노드의 이름을 리스트로 저장함
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        # 첫 번째 입력 노드의 형태를 가져와서 입력 높이와 너비를 저장함.
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        # 모델의 출력 노드 정보를 저장
        model_outputs = self.session.get_outputs()

        # 출력 노드의 이름을 리스트로 저장
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # 입력 이미지의 형태를 이용해서 바운딩 박의 크기를 조정함. 이를 통해 바운딩 박스를 원본 이미지 크기에 맞게 조정할 수 있음.
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        # 바운딩 박의 좌표를 입력 이미지 크기로 나누어 정규화함.

        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
        # 정규화된 바운딩 박스를 다시 원본 이미지 크기에 맞게 스케일함.
        # image_shape = [ '높이', '너비' ] 바운딩 박스의 x1, x2는 너비, y1, y2는 높이에 관한 정보이므로 1, 0, 1, 0

        # 바운딩 박스 리턴
        return boxes


if __name__ == '__main__':
    # 테스팅 코드
    from imread_from_url import imread_from_url
    #
    # model_path = "../models/yolov8n-seg.onnx"
    #
    # # Initialize YOLOv8 Instance Segmentator
    # yoloseg = YOLOSeg(model_path, conf_thres=0.7, iou_thres=0.5)
    #
    # img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
    # img = imread_from_url(img_url)
    #
    # # Detect Objects
    # yoloseg(img)
    #
    # # Draw detections
    # combined_img = yoloseg.draw_masks(img)
    # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    # cv2.imshow("Output", combined_img)
    # cv2.waitKey(0)
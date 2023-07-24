import os
import cv2 as cv
import numpy as np
import time
import json
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)  # Flask 애플리케이션 생성

cors = CORS(app, resources={r"/*": {
    "origins": 'http://localhost:3000',
    "allow_headers": "*",
    "expose_headers": "*"}}, supports_credentials=True)

# YOLO 모델 설정
Conf_threshold = 0.6  # 객체를 검출하는데 사용되는 Confidence(신뢰도) 임계값
NMS_threshold = 0.4  # 겹치는 박스들을 제거하기 위한 Non-maximum Suppression(비최대 억제) 임계값

# 클래스 이름을 담을 리스트 초기화
class_name = []
with open("../model/classes_pothole.txt", "r") as f:
    class_name = [
        cname.strip() for cname in f.readlines()
    ]  # classes.txt 파일을 읽어서 리스트로 저장

# YOLOv4-tiny 모델 불러오기
net = cv.dnn.readNetFromDarknet("../model/yolov4Pothole.cfg", "../model/yolov4Pothole.weights")

# GPU 지원을 확인하고 사용 가능하면 CUDA backend를 사용하도록 설정
if cv.cuda.getCudaEnabledDeviceCount() > 0:
    print("Using GPU")
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
else:
    print("Using CPU")

classes = []
with open("../model/classes_pothole.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
        
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# cv.dnn.readNetFromDarknet
# 모델 설정
model = cv.dnn_DetectionModel(net)  # YOLO 모델을 사용하기 위한 객체 생성
model.setInputParams(
    size=(416, 416), scale=1 / 255, swapRB=True
)  # 모델에 입력될 이미지의 크기, 픽셀값 범위, 채널 순서 설정


DETECTION_URL = "/inference"  # 객체 검출 URL 경로


# 받은 이미지에 대한 객체 검출 결과 좌표를 반환
@app.route(DETECTION_URL, methods=["POST"])  # DETECTION_URL 경로에 대한 POST 요청 처리
def inference():
    if "imageFile" not in request.files:  # 전송된 파일이 없으면
        return "No file", 400  # 400 에러 반환


    #video 받기
    image_file = request.files["imageFile"]

    # 이미지 파일을 바이트 스트림으로 읽기
    image_data = image_file.read()

    # 바이트 스트림을 메모리에서 읽어 OpenCV 이미지 배열로 변환
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    imageName = image_file.filename
    
    blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    height, width, channels = img.shape

    imageName = image_file.filename


    print("{} received".format(imageName))

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    confidence_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 박스 좌표 계산
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Non-maximum suppression 수행
    indices = cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    detected_objects = []
    for i in indices:
        # i = i
        box = boxes[i]
        print(box)
        x, y, w, h = box
        label = classes[class_ids[i]]
        confidence = confidences[i]
        detected_objects.append({"label": label, "confidence": confidence, "box": (x, y, w, h)})

    return detected_objects





    # # 객체 검출 수행
    # start = time.time()  # 시간 측정 시작
    # classes, scores, boxes = model.detect(img, Conf_threshold, NMS_threshold)
    # end = time.time()  # 시간 측정 종료

    # # 검출된 객체들에 대한 개수와 좌표를 반환
    # result = {
    #     #boxes의 클래스와 좌표를 반환
    #     "classes": classes.tolist(),
    #     "boxes": boxes.tolist(),
    #     "scores": scores.tolist(),
    #     "count": len(classes),
    #     "time": end - start,
    # }

    # return jsonify(result)  # JSON 형식으로 반환

    # return "OK", 200  # 200 OK 반환



if __name__ == "__main__":
    app.run(debug=True, port=5002)
    # app.run(host="0.0.0.0", port=5000)


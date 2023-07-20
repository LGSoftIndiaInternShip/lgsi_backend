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
with open("classes.txt", "r") as f:
    class_name = [
        cname.strip() for cname in f.readlines()
    ]  # classes.txt 파일을 읽어서 리스트로 저장

# YOLOv4-tiny 모델 불러오기
net = cv.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

# GPU 지원을 확인하고 사용 가능하면 CUDA backend를 사용하도록 설정
if cv.cuda.getCudaEnabledDeviceCount() > 0:
    print("Using GPU")
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
else:
    print("Using CPU")

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


    #video 받아서 static/video/에 저장
    image_file = request.files["imageFile"]

    # 이미지 파일을 바이트 스트림으로 읽기
    image_data = image_file.read()

    # 바이트 스트림을 메모리에서 읽어 OpenCV 이미지 배열로 변환
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)


    imageName = image_file.filename

    print("{} received".format(imageName))


    # 객체 검출 수행
    start = time.time()  # 시간 측정 시작
    classes, scores, boxes = model.detect(img, Conf_threshold, NMS_threshold)
    end = time.time()  # 시간 측정 종료

    # 검출된 객체들에 대한 개수와 좌표를 반환
    result = {
        "count": len(classes),
        "boxes": boxes.tolist(),
        "time": end - start,
    }

    return jsonify(result)  # JSON 형식으로 반환

    # return "OK", 200  # 200 OK 반환



if __name__ == "__main__":
    app.run(debug=True, port=5001)
    # app.run(host="0.0.0.0", port=5000)


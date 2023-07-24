import os
import cv2 as cv
import numpy as np
import time
import json
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# YOLO 모델 설정
model = YOLO('model/best.pt')
model.predict("./static/image/IndiaRoad.jpg", conf = 0.6, device = 0)

app = Flask(__name__)  # Flask 애플리케이션 생성

cors = CORS(app, resources={r"/*": {
    "origins": 'http://localhost:3000',
    "allow_headers": "*",
    "expose_headers": "*"}}, supports_credentials=True)



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
    results = model.predict(img, conf = 0.6, device = 0)  # YOLO 모델을 사용하여 객체 검출 수행
    #GPU 사용되는지 확인
    # print("GPU 사용 여부: {}".format(results.device))
    end = time.time()  # 시간 측정 종료


    # 검출된 객체들에 대한 개수와 좌표를 반환

    boxes_data = []
    for box in results[0].boxes.xyxyn.tolist():
        boxes_data.append(round(box[1], 2))
        boxes_data.append(round(box[0], 2))
        boxes_data.append(round(box[3], 2))
        boxes_data.append(round(box[2], 2))
        
        # for coord in box:
        #     boxes_data.append(round(coord, 2))

            
        # boxes_data.append([round(coord, 2) for coord in box])
        

    # for result in results:
    ret={
        "classes": results[0].boxes.cls.tolist(),
        # "scores": result.boxes.conf.tolist(),
        "scores": [round(score, 2) for score in results[0].boxes.conf.tolist()],
        # "boxes": [[round(coord, 2) for coord in box] for box in results[0].boxes.xyxyn.tolist()],
        "boxes": boxes_data,
        # "count": len(result.boxes.cls),
        "time": end - start,
    }


    return jsonify(ret), 200  # 200 OK 반환
    # return "OK", 200  # 200 OK 반환


if __name__ == "__main__":
    app.run(debug=True, port=5002)
    # app.run(host="0.0.0.0", port=5000)




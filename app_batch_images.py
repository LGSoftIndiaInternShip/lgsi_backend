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
#첫번째로 돌릴때 시간이 오래걸려서 미리 돌려놓고 시작
model.predict("./static/image/IndiaRoad.jpg", conf= 0.6, device= 0)

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

    start = time.time()  # 시간 측정 시작
    #Image list 받기
    image_list = request.files.getlist("imageFile")

    print(image_list)

    np_image_list = []
    for image in image_list:
        image_data = image.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        np_image_list.append(img)


    # 객체 검출 수행
    results = model.predict(np_image_list, conf = 0.6, device = 0)  # YOLO 모델을 사용하여 객체 검출 수행


    # 검출된 객체들에 대한 개수와 좌표를 반환

    rets = []

    for result in results:
        boxes_data = []
        ret = []
        for box in result.boxes.xyxy.tolist():
            boxes_data.append(round(box[1], 2))
            boxes_data.append(round(box[0], 2))
            boxes_data.append(round(box[3], 2))
            boxes_data.append(round(box[2], 2))
            
        ret.append({
            "classes": result.boxes.cls.tolist(),
            "scores": [round(score, 2) for score in result.boxes.conf.tolist()],
            "boxes": boxes_data,
            
        })
        
        end = time.time()  # 시간 측정 종료
        ret.append({
            "time": end - start,
        })

        rets.append(ret)

    

    return jsonify(rets), 200  # 200 OK 반환
    # return "OK", 200  # 200 OK 반환


if __name__ == "__main__":
    app.run(debug=True, port=5002)
    # app.run(host="0.0.0.0", port=5000)




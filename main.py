from typing import Union
from fastapi import FastAPI, UploadFile, File
from starlette.requests import Request
from fastapi import FastAPI, HTTPException, UploadFile, File
from starlette.requests import Request
from starlette.responses import HTMLResponse
from ImgAnomalyDetect import predict
from PIL import Image
import numpy as np
import io
import os
import cv2
import tempfile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/input", StaticFiles(directory="input"), name="input")

# file_path에 해당하는 파일 삭제
def delete_temp_file(file_path: str):
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error deleting temp file: {e}")

# 기존에 저장된 input 폴더 아래의 모든 파일 삭제
def delete_all_files_in_input_folder():
    folder_path = "input/"
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files in input/ folder deleted.")
    except Exception as e:
        print(f"Error deleting files in input/ folder: {e}")

# 이미지를 통해 불량 탐지 모델 예측 결과 반환
@app.post("/predict")
async def predict_image(request: Request, file: UploadFile = File(...)):
    try:
        # 1. input 폴더 아래의 모든 파일 삭제
        delete_all_files_in_input_folder()
        
        # 2. 입력으로 들어온 파일(사진) 저장
        file_path = os.path.join('input/', file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read()) 

        # 3. 모델이 예측한 라벨 반환
        predicted_label = predict(file_path)

        # 4. 예측한 결과와 이미지를 반환
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "class": predicted_label.split('_')[0], "anomaly" : predicted_label.split('_')[1], "image_path": file_path.split('/')[1]}
        )

    except Exception as e:
        delete_temp_file(file_path)
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")

# 이미지를 입력받음
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})
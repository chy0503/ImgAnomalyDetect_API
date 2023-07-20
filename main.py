
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
import cv2

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 이미지 불러오기 및 예측
    image = Image.open(io.BytesIO(await file.read()))
    predicted_label = predict(image)
    return {"result": predicted_label}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return """
    <html>
        <body>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit">
            </form>
        </body>
    </html>
    """
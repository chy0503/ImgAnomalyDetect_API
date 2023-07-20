# ImgAnomalyDetect_API

본 프로젝트의 목적은 컴퓨터 비전 기술을 이용하여 생산 라인에서 불량 제품을 자동으로 탐지하는 시스템을 개발하는 것입니다. 생산 과정에서 불량 제품을 신속하게 감지하고 분류하는 것은 생산 효율을 향상시키고 제품 품질을 보장하는데 매우 중요합니다.<br>
생산 과정의 효율성을 높이고, 제품 품질을 향상시키며, 생산비용을 줄일 수 있습니다. 또한 이 시스템은 다양한 생산 라인에서 불량 제품 탐지에 활용될 수 있으며, 컴퓨터 비전 및 인공지능 기술의 산업 분야에서의 응용 가능성을 확대시킵니다.<br><br>

## 이미지를 입력받아 불량 판단하는 API
```
ImgAnomalyDetect_API
├──model
├──mvtec_anomaly_detection
│  ├── bottle
│  │   ├── ground_truth
│  │   │   ├── broken_large
│  │   │   ├── broken_small
│  │   │   └── contamination
│  │   ├── test
│  │   │   ├── broken_large
│  │   │   ├── broken_small
│  │   │   ├── contamination
│  │   │   └── good
│  │   └── train
│  │       └── good
    ...
```
0. mvtec dataset을 다운로드하고 후 ImgAnomalyDetect_API/mvtec_anomaly_detection이 되도록 위치
1. 각 클래스의 test 폴더를 돌며 data 전처리 및 라벨링
    - 15종의 제품 데이터에 대해 good/bad으로 분류
    - ex. bootle_good, pill_bad
2. 데이터를 train, test, validation으로 분할
3. 모델 생성 및 학습
    - 단층적(Single) CNN: 물체의 종류와 이상치를 한 번에 30개의 상태로 나누어 분류하는 단편적 형태의 CNN 모델
    - CNN(Convolutional Neural Networks) Model Architect
        - image size : 128*128
        - using 5 Convolutional Layers
        - using max pooling layers
        - kernel size : 3,3
        - activation function : relu
        - activation function(output layers) : softmax
4. 예측
<br><br>

## Reference
- 데이터셋 : [MVTec Anomaly Detection Dataset: MVTec Software](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
- 모델 구성 : [다중 클래스 이상치 탐지를 위한 계층 CNN의 효과적인 클래스 분할 방법
](https://github.com/jeehyunee3/cnn_multiclass_outlierdetection/tree/main)

# BirdCLEF2025 inference에 대한 설명입니다.
https://www.kaggle.com/code/molcar/lb-0-804-efficientnet-b0-pytorch-pipeline

## 주요 기능
### 오디오 세그먼트 
- 오디오 파일을 5초 단위의 세그먼트로 분할하여 예측을 수행합니다.

### 모델별 전처리 
- 모델의 아키텍처에 맞는 전처리를 적용합니다.(seresnext는 원래 코드의 전처리 방식을 사용)

### TTA 
- 모델의 예측력을 향상시키기 위해 EfficientNet에 대한 수직/수평 뒤집기 TTA를 적용합니다(seresnext는 원래 코드의 TTA 방식을 적용)
  
### 가중치 앙상블

### 예측 스무딩

## Configuration 설명
### SR
- Sampling Rate
### N_FFT
- FFT의 크기
### HOP_LENGTH
- 한 프레임에서 다음 프레임까지의 샘플 수
### N_MELS
- 멜 밴드의 수
### FMIN
- 최소 주파수
### FMAX
- 최대 주파수
  
## 참고사항
### 데이터는 아래 링크에서 받을 수 있습니다.
https://www.kaggle.com/competitions/birdclef-2025

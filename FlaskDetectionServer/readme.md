# YOLO v11 버전 과수화상병 진단 객체탐지 플라스크 서버

## 구조설명 
1. yolov11ObjectDetection으로 모델훈련
2. 훈련을 끝낸뒤 model을 현재위치의 폴더에 넣고
3. app.py실행시 플라스크 서버 실행 (fruit_disease_predictor에서 예측을 진행)


## 파일설명
1. prediction_results : 객체탐지 결과 이미지를 저장하는 폴더 (app.py 실행시 이미지 자동삭제)

2. uploads : 객체탐지에 사용할 예측이미지를 저장하는 폴더 (app.py 실행시 이미지 자동삭제)

3. yolov11_results : 객체탐지 모델(pt) 가 저장되는 장소(에폭별,last,best등이 저장됨)

4. app.py : 플라스크 서버를 실행하는 파일

5. createDataset.ipynb : 원본 데이터셋에서 -> yolov11 용 데이터셋으로 변환하는 파일

6. fruit_disease_predictor.py : 객체탐지 모델을 불러와서 예측을 실행하는 파일

7. requirments.txt : 파일들을 실행하기위해 필요한 라이브러리들
 

## 메뉴얼
1. 가상환경 설치
conda create -n DL python=3.10

2. 가상환경 활성화
conda activate DL

3. 파이토치 12.8버전설치(GPU버전)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

4. createDataset.ipynb실행
4-1 원본데이터셋 경로설정
4-2 변환저장할 Yolo v11용 데이터셋 경로설정
4-3 createDataset 실행

5. app.py 플라스크 서버 실행
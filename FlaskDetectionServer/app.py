import os
import io
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from fruit_disease_predictor import FruitDiseasePredictor
from datetime import datetime
# import shutil

#==============================================================
#상수설정
#==============================================================
MODEL_PATH = "./Yolov11nBestModel.pt"   # <-여기에 모델경로설정


app = Flask(__name__)

# CORS 설정 추가 - 모든 도메인에서의 요청 허용
CORS(app)

# 또는 특정 도메인만 허용하려면:
# CORS(app, origins=['http://localhost:8087'])

# 설정
UPLOAD_FOLDER = 'uploads'
PREDICTION_RESULTS_FOLDER = 'prediction_results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_RESULTS_FOLDER'] = PREDICTION_RESULTS_FOLDER

# 업로드 및 예측 결과 디렉토리가 존재하는지 확인하고 없으면 생성
# 이 부분은 폴더가 없으면 생성하는 역할만 합니다.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_RESULTS_FOLDER, exist_ok=True)

# --- 추가된 코드 시작 ---
def clean_image_files_in_folder(folder_path, allowed_extensions):
    """
    지정된 폴더 내의 이미지 파일들만 삭제합니다.
    """
    print(f"애플리케이션 시작 시 '{folder_path}' 폴더 내의 이미지 파일들을 정리합니다.")
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath) and filename.rsplit('.', 1)[1].lower() in allowed_extensions:
            try:
                os.remove(filepath)
                print(f"  - 파일 삭제: {filename}")
            except Exception as e:
                print(f"  - 파일 삭제 실패 ({filename}): {e}")
# --- 추가된 코드 끝 ---

# 앱 시작 시 폴더 내 이미지 파일 정리
clean_image_files_in_folder(UPLOAD_FOLDER, ALLOWED_EXTENSIONS)
clean_image_files_in_folder(PREDICTION_RESULTS_FOLDER, ALLOWED_EXTENSIONS)

# 앱 시작 시 예측기(predictor)를 한 번만 초기화합니다.
predictor = None
try:
    predictor = FruitDiseasePredictor(MODEL_PATH)
except Exception as e:
    print(f"FruitDiseasePredictor 초기화 오류: {e}")

def allowed_file(filename):
    """
    허용된 파일 확장자인지 확인합니다.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """
    루트 경로에 대한 응답입니다. API가 실행 중임을 알립니다.
    """
    return "과일 병해충 예측 API가 실행 중입니다! /predict 경로로 POST 요청을 보내세요."

@app.route('/predict', methods=['POST'])
def predict_disease():
    """
    이미지를 받아 병해충을 예측하는 엔드포인트입니다.
    """
    # 예측기 초기화에 실패한 경우 오류 반환
    if predictor is None:
        return jsonify({"error": "모델이 로드되지 않았습니다. 서버 초기화에 실패했습니다."}), 500

    # 단일 파일 업로드를 위해 'file' 확인 (JavaScript에서 'file'로 전송)
    if 'file' not in request.files:
        return jsonify({"error": "요청에 파일 부분이 없습니다. 'file'이라는 이름으로 파일을 전송해주세요."}), 400

    file = request.files['file']  # 단일 파일 가져오기

    # 파일이 선택되지 않은 경우
    if file.filename == '':
        return jsonify({"error": "선택된 파일이 없습니다."}), 400

    # 요청 폼에서 conf_threshold 값을 가져오거나 기본값 0.5 사용
    conf_threshold = float(request.form.get('conf_threshold', 0.5))

    # 파일이 존재하고 허용된 확장자인지 확인
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # 파일 이름 충돌 방지를 위한 타임스탬프 추가
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            file.save(filepath)  # 파일을 지정된 경로에 저장

            # 예측 수행
            result_info = predictor.predict_single_image(
                image_path=filepath,
                conf_threshold=conf_threshold,
                save_dir=app.config['PREDICTION_RESULTS_FOLDER']
            )

            # 생성된 결과 이미지를 읽고 base64로 인코딩
            with open(result_info['output']['result_image_path'], "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            # 응답 데이터 구성
            response_data = {
                "success": True,
                "message": f"'{filename}' 예측 성공",
                "input_image_info": {
                    "filename": result_info['input_image']['filename'],
                    "width": result_info['input_image']['size']['width'],
                    "height": result_info['input_image']['size']['height']
                },
                "prediction_details": {
                    "timestamp": result_info['prediction']['timestamp'],
                    "total_detections": result_info['prediction']['total_detections'],
                    "detections": result_info['prediction']['detections']
                },
                "output_image": {
                    "filename": result_info['output']['result_image_filename'],
                    "base64_encoded_image": encoded_image
                }
            }
            
            return jsonify(response_data), 200

        except FileNotFoundError as e:
            return jsonify({"success": False, "error": str(e)}), 404
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        except Exception as e:
            return jsonify({"success": False, "error": f"예측 실패: {str(e)}"}), 500
    else:
        # 허용되지 않는 파일 형식인 경우
        return jsonify({"success": False, "error": "허용되지 않는 파일 형식입니다."}), 400

@app.route('/prediction_results/<filename>')
def serve_prediction_image(filename):
    """
    예측된 이미지를 직접 제공하는 엔드포인트입니다.
    """
    return send_from_directory(app.config['PREDICTION_RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

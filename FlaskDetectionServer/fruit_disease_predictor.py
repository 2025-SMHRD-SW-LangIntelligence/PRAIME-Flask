# fruit_disease_predictor.py
import os
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

#상수설정
MODEL_PATH = "./Yolov11nBestModel.pt"   # <-여기에 모델경로설정

class FruitDiseasePredictor:
    def __init__(self, model_path):
        """
        과수화상병 진단 예측기 초기화
        
        Args:
            model_path (str): 훈련된 YOLO 모델 경로 (best.pt)
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = {
            0: '정상',
            1: '배검은별무니병',
            2: '배과수화상병',
            3: '사과갈색무늬병',
            4: '사과과수화상병',
            5: '사과부란병',
            6: '사과점무늬낙엽병',
            7: '사과탄저병'
        }
        self.colors = {
            0: (0, 255, 0),    # 녹색: 정상
            1: (0, 0, 255),    # 빨강: 배 질병
            2: (0, 255, 255),  # 노랑: 배 질병
            3: (255, 0, 0),    # 파랑: 사과 질병
            4: (255, 165, 0),  # 주황: 사과 질병
            5: (128, 0, 128),  # 보라: 사과 질병
            6: (255, 192, 203), # 핑크: 사과 질병
            7: (0, 128, 128)   # 청록: 사과 질병
        }
        
        # 한글 폰트 설정
        self.font = self._load_korean_font()
        
        self.load_model()
    
    def _load_korean_font(self):
        """한글 폰트 로드"""
        try:
            # Windows 시스템에서 한글 폰트 찾기
            font_paths = [
                "C:/Windows/Fonts/malgun.ttf",  # 맑은 고딕
                "C:/Windows/Fonts/NanumGothic.ttf",  # 나눔고딕
                "/System/Library/Fonts/AppleGothic.ttf",  # macOS
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Ubuntu
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    return ImageFont.truetype(font_path, 20)
            
            # 기본 폰트 사용
            print("Warning: Korean font not found. Using default font.")
            return ImageFont.load_default()
            
        except Exception as e:
            print(f"Font loading error: {e}")
            return ImageFont.load_default()
    
    def load_model(self):
        """모델 로드"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully from: {self.model_path}")
            print(f"Using device: {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _draw_korean_text(self, image, text, position, font, color=(255, 255, 255), bg_color=None):
        """
        PIL을 사용하여 한글 텍스트를 이미지에 그리기
        
        Args:
            image: OpenCV 이미지 (BGR)
            text: 그릴 텍스트
            position: (x, y) 위치
            font: PIL 폰트
            color: 텍스트 색상 (RGB)
            bg_color: 배경 색상 (RGB)
        
        Returns:
            OpenCV 이미지 (BGR)
        """
        try:
            # OpenCV BGR을 PIL RGB로 변환
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # 텍스트 크기 계산
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x, y = position
            
            # 배경 그리기
            if bg_color:
                padding = 5
                draw.rectangle(
                    [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                    fill=bg_color
                )
            
            # 텍스트 그리기
            draw.text((x, y), text, font=font, fill=color)
            
            # PIL RGB를 OpenCV BGR로 변환
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Error drawing Korean text: {e}")
            return image
    
    def predict_single_image(self, image_path, conf_threshold=0.5, save_dir="predictions"):
        """
        단일 이미지에 대한 예측 수행
        
        Args:
            image_path (str): 예측할 이미지 경로
            conf_threshold (float): 신뢰도 임계값
            save_dir (str): 결과 저장 디렉토리
            
        Returns:
            dict: 예측 결과 정보
        """
        try:
            # 이미지 경로 검증
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # 저장 디렉토리 생성
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
            
            # 이미지 로드
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 예측 실행
            print(f"Running prediction on: {image_path.name}")
            results = self.model.predict(
                source=img_rgb,
                conf=conf_threshold,
                imgsz=640,
                device=self.device,
                verbose=False
            )
            
            # 결과 처리
            detections = []
            result_img = img.copy()
            
            # 예측 시간
            prediction_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if results and len(results) > 0 and hasattr(results[0], 'boxes'):
                for i, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    
                    class_name = self.class_names.get(class_id, f'클래스_{class_id}')
                    color = self.colors.get(class_id, (255, 255, 255))
                    
                    detection = {
                        'detection_id': i + 1,
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': round(confidence, 4),
                        'bbox': {
                            'x1': round(x1, 2),
                            'y1': round(y1, 2),
                            'x2': round(x2, 2),
                            'y2': round(y2, 2),
                            'width': round(x2 - x1, 2),
                            'height': round(y2 - y1, 2)
                        }
                    }
                    detections.append(detection)
                    
                    # 박스 그리기
                    cv2.rectangle(
                        result_img,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2
                    )
                    
                    # 라벨 텍스트 (한글 지원)
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # PIL을 사용하여 한글 텍스트 그리기
                    result_img = self._draw_korean_text(
                        result_img,
                        label,
                        (int(x1), int(y1) - 30),
                        self.font,
                        color=(255, 255, 255),
                        bg_color=tuple(reversed(color))  # BGR to RGB
                    )
            
            # 결과 이미지 저장
            output_filename = f"pred_{image_path.stem}_{prediction_time}.jpg"
            output_path = save_path / output_filename
            cv2.imwrite(str(output_path), result_img)
            
            # 결과 정보 구성
            result_info = {
                'input_image': {
                    'path': str(image_path),
                    'filename': image_path.name,
                    'size': {
                        'width': img.shape[1],
                        'height': img.shape[0],
                        'channels': img.shape[2]
                    }
                },
                'prediction': {
                    'timestamp': prediction_time,
                    'model_path': str(self.model_path),
                    'confidence_threshold': conf_threshold,
                    'total_detections': len(detections),
                    'detections': detections
                },
                'output': {
                    'result_image_path': str(output_path),
                    'result_image_filename': output_filename
                }
            }
            
            print(f"Prediction completed. Found {len(detections)} objects.")
            print(f"Result image saved to: {output_path}")
            
            return result_info
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
    
    def save_results_to_yaml(self, result_info, yaml_path=None):
        """
        예측 결과를 YAML 파일로 저장
        
        Args:
            result_info (dict): 예측 결과 정보
            yaml_path (str, optional): YAML 파일 저장 경로
            
        Returns:
            str: 저장된 YAML 파일 경로
        """
        try:
            if yaml_path is None:
                timestamp = result_info['prediction']['timestamp']
                input_filename = result_info['input_image']['filename']
                yaml_filename = f"prediction_result_{input_filename}_{timestamp}.yaml"
                yaml_path = Path("prediction_results") / yaml_filename
            else:
                yaml_path = Path(yaml_path)
            
            # 디렉토리 생성
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            
            # YAML 파일 저장 (UTF-8 인코딩으로 명시적 저장)
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    result_info, 
                    f, 
                    default_flow_style=False, 
                    allow_unicode=True, 
                    indent=2
                )
            
            print(f"Results saved to YAML: {yaml_path}")
            return str(yaml_path)
            
        except Exception as e:
            print(f"Error saving YAML: {e}")
            raise
    
    def save_results_to_json(self, result_info, json_path=None):
        """
        예측 결과를 JSON 파일로 저장 (추가 기능)
        
        Args:
            result_info (dict): 예측 결과 정보
            json_path (str, optional): JSON 파일 저장 경로
            
        Returns:
            str: 저장된 JSON 파일 경로
        """
        try:
            if json_path is None:
                timestamp = result_info['prediction']['timestamp']
                input_filename = result_info['input_image']['filename']
                json_filename = f"prediction_result_{input_filename}_{timestamp}.json"
                json_path = Path("prediction_results") / json_filename
            else:
                json_path = Path(json_path)
            
            # 디렉토리 생성
            json_path.parent.mkdir(parents=True, exist_ok=True)
            
            # JSON 파일 저장 (UTF-8 인코딩으로 명시적 저장)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(
                    result_info, 
                    f, 
                    ensure_ascii=False,  # 한글 문자 보존
                    indent=2
                )
            
            print(f"Results saved to JSON: {json_path}")
            return str(json_path)
            
        except Exception as e:
            print(f"Error saving JSON: {e}")
            raise
    
    def display_result(self, result_info):
        """예측 결과 시각화"""
        try:
            # matplotlib 한글 폰트 설정
            plt.rcParams['font.family'] = ['Malgun Gothic', 'NanumGothic', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            result_image_path = result_info['output']['result_image_path']
            
            if Path(result_image_path).exists():
                img = cv2.imread(result_image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                plt.figure(figsize=(12, 8))
                plt.imshow(img_rgb)
                plt.title(f"예측 결과 - {result_info['input_image']['filename']}")
                plt.axis('off')
                
                # 탐지 결과 요약 표시
                detections = result_info['prediction']['detections']
                if detections:
                    summary_text = f"총 탐지 수: {len(detections)}\n"
                    for det in detections:
                        summary_text += f"• {det['class_name']}: {det['confidence']:.2f}\n"
                    plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                plt.tight_layout()
                plt.show()
            else:
                print(f"Result image not found: {result_image_path}")
                
        except Exception as e:
            print(f"Error displaying result: {e}")


def main():
    """메인 실행 함수"""
    # 콘솔 출력 UTF-8 설정 (Windows용)
    import sys
    if sys.platform.startswith('win'):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    
    # 모델 경로 설정 
    model_path = MODEL_PATH
    
    # 예측기 초기화
    try:
        predictor = FruitDiseasePredictor(model_path)
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        return
    
    # 예측 실행 예시
    # file_name = input("파일명입력 :")
    # test_image_path = "./test_image/"+file_name+".jpg"  
    test_image_path = "./test_image/pear.jpg"  # 실제 이미지 경로로 수정
    
    try:
        # 예측 실행
        result_info = predictor.predict_single_image(
            image_path=test_image_path,
            conf_threshold=0.5,
            save_dir="prediction_results"
        )
        
        # YAML과 JSON 모두 저장
        yaml_path = predictor.save_results_to_yaml(result_info)
        json_path = predictor.save_results_to_json(result_info)
        
        # 결과 시각화
        predictor.display_result(result_info)
        
        # 결과 출력
        print("\n=== 예측 결과 요약 ===")
        print(f"입력 이미지: {result_info['input_image']['filename']}")
        print(f"총 탐지 수: {result_info['prediction']['total_detections']}")
        print(f"결과 이미지: {result_info['output']['result_image_filename']}")
        print(f"YAML 결과: {yaml_path}")
        print(f"JSON 결과: {json_path}")
        
        if result_info['prediction']['detections']:
            print("\n탐지된 객체:")
            for det in result_info['prediction']['detections']:
                print(f"  - {det['class_name']}: {det['confidence']:.4f}")
        else:
            print("탐지된 객체가 없습니다.")
            
    except Exception as e:
        print(f"예측 실패: {e}")


if __name__ == "__main__":
    main()
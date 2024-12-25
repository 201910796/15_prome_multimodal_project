import argparse
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import warnings

from text_preprocessor import TextPreprocessor  # Changed import
from preprocessing import MultimodalPreprocessor  # Original multimodal preprocessor

# 경고 메시지 필터링
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SingleModalityClassifier:
    """단일 모달리티 감정 분류기"""
    def __init__(self, modality_type: str, embedding_dim: int = 768):
        self.modality_type = modality_type
        self.embedding_dim = embedding_dim
        self.dropout_rate = 0.4
        self.l2_reg = 0.001
        
        self.emotion_labels = {
            0: "anger",
            1: "disgust", 
            2: "fear",
            3: "joy",
            4: "sad",
            5: "surprise"
        }
        
        self._build_model()
    
    def _build_model(self):
        """단일 모달리티용 모델 구축"""
        inputs = layers.Input(shape=(self.embedding_dim,), name=f"{self.modality_type}_input")
        
        # Feature extraction
        x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(self.l2_reg))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        outputs = layers.Dense(len(self.emotion_labels), activation="softmax")(x)
        
        self.model = models.Model(
            inputs=inputs,
            outputs=outputs,
            name=f"single_{self.modality_type}_classifier"
        )
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Single modality model for {self.modality_type} built successfully")

    def predict_emotion(self, embedding: np.ndarray) -> tuple[str, float]:
        """감정 예측 수행"""
        try:
            # 배치 차원 추가
            input_data = np.expand_dims(embedding, axis=0)
            
            # 예측 수행
            predictions = self.model.predict(input_data, verbose=0)
            
            # 결과 처리
            emotion_idx = np.argmax(predictions[0])
            confidence = predictions[0][emotion_idx]
            predicted_emotion = self.emotion_labels[emotion_idx]
            
            return predicted_emotion, confidence, predictions[0]
            
        except Exception as e:
            logger.error(f"Error during emotion prediction: {str(e)}")
            raise

def single_run(modality_type: str, input_data: str):
    """단일 모달리티 평가 실행"""
    try:
        # 전처리기 초기화
        if modality_type == "text":
            preprocessor = TextPreprocessor()
            embedding = preprocessor.get_embedding(input_data)
            logger.info("Text preprocessing completed")
        else:
            preprocessor = MultimodalPreprocessor()
            if modality_type == "image":
                embedding = preprocessor.preprocess_image(input_data)
                logger.info("Image preprocessing completed")
            elif modality_type == "audio":
                embedding = preprocessor.preprocess_audio(input_data)
                logger.info("Audio preprocessing completed")
            elif modality_type == "video":
                embedding = preprocessor.preprocess_video(input_data)
                logger.info("Video preprocessing completed")
        
        # 분류기 초기화
        classifier = SingleModalityClassifier(modality_type)
        
        # 예측 수행
        emotion, confidence, all_predictions = classifier.predict_emotion(embedding)
        
        # 결과 출력
        print("\n" + "="*50)
        print(f"{modality_type.upper()} 모달리티 평가 결과")
        print("-" * 30)
        print(f"입력: {input_data}")
        print(f"\n주요 예측 감정: {emotion} (확률: {confidence:.2%})")
        
        print("\n전체 감정 확률 분포:")
        for emotion_name, prob in zip(classifier.emotion_labels.values(), all_predictions):
            print(f"- {emotion_name:8}: {prob:.2%}")
        print("="*50 + "\n")
        
        return emotion, confidence, all_predictions
        
    except Exception as e:
        logger.error(f"모달리티 평가 중 오류 발생: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='단일 모달리티 감정 분석')
    
    parser.add_argument('--modality', 
                       type=str,
                       required=True,
                       choices=['text', 'image', 'audio', 'video'],
                       help='평가할 모달리티 유형')
    
    parser.add_argument('--input', 
                       type=str,
                       required=True,
                       help='입력 데이터 (텍스트 또는 파일 경로)')
    
    args = parser.parse_args()
    
    # 파일 경로 검증 (텍스트 모달리티 제외)
    if args.modality != 'text' and not Path(args.input).exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {args.input}")
    
    single_run(args.modality, args.input)

if __name__ == "__main__":
    main()

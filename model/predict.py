import torch
import os
import clip
from PIL import Image
import numpy as np
from collections import Counter
from tqdm import tqdm
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

from model import ImageEmotionClassifier, TextEmotionClassifier,VideoModel
from preprocessing import preprocess_image, preprocess_text,VideoPreprocessor
from emotion_ensemble import EmotionEnsemble

#  감정라벨 분노 0 기쁨 1 슬픔 2
# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VideoPredictor:
    """비디오 프레임별 감정 예측 및 최종 감정 결정을 담당하는 클래스"""
    def __init__(self, model, preprocess, device):
        self.model = model
        self.preprocess = preprocess
        self.device = device

        # 기본 텍스트 리스트 정의
        self.emotion_labels = [
            "a joyful feeling", "a happy feeling", "an excited feeling", "a laughing feeling", "a peaceful feeling",
            "a sad feeling", "a crying feeling", "a fear feeling", "a frustrated feeling",
            "an angry feeling", "a frustrated feeling", "a disgusting feeling", "a surprised feeling"
        ]

    def set_emotion_labels(self, emotion_labels):
        """텍스트 리스트를 외부에서 재정의"""
        self.emotion_labels = emotion_labels

    def generate_clip_caption(self, frame_path):
        """각 프레임에 대해 감정 예측"""
        image = self.preprocess(Image.open(frame_path)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)

        # 텍스트 예시 목록에 대해 특징 벡터 추출
        text_inputs = torch.cat([clip.tokenize(text) for text in self.emotion_labels]).to(self.device)

        text_features = self.model.encode_text(text_inputs)

        # 유사도 계산
        similarities = (image_features @ text_features.T).squeeze(0)
        values, indices = similarities.topk(1)

        # 가장 유사한 텍스트 예시를 반환
        return self.emotion_labels[indices.item()]

    def get_frame_wise_emotions(self, frame_dir):
        """비디오에서 각 프레임의 감정 라벨을 추출"""
        frame_emotions = []  # 각 프레임의 감정 라벨을 저장할 리스트
        for frame_file in tqdm(os.listdir(frame_dir)):
            if frame_file.endswith('.jpg'):
                frame_path = os.path.join(frame_dir, frame_file)

                # 각 프레임에 대해 단 하나의 감정 라벨만 예측
                frame_caption = self.generate_clip_caption(frame_path)
                frame_emotions.append(frame_caption)  # 감정 라벨만 추출하여 저장

        return frame_emotions

    def get_final_emotion(self, frame_emotions):
        """최종 감정 결정"""
        # frame_emotions가 비어 있는 경우 처리
        if not frame_emotions:
            raise ValueError("No emotions were predicted for any of the frames.")

        # 다수결로 최종 감정 결정
        final_emotion = Counter(frame_emotions).most_common(1)[0][0]

        # 실제 우리 감정 라벨 매핑
        emotion_mapping = {
            "a joyful feeling": "joy",
            "an excited feeling": "joy",
            "a laughing feeling": "joy",
            "a peaceful feeling": "joy",
            "a sad feeling": "sadness",
            "a crying feeling": "sadness",
            "a fear feeling": "sadness",
            "a frustrated feeling": "sadness",
            "an angry feeling": "anger",
            "a disgusting feeling": "anger",
            "a surprised feeling": "anger",
        }
        # 실제 감정라벨을 숫자로 인코딩
        label_mapping = {"joy":'1','sadness':'2','anger':'0'}
        # 최종 감정을 감정 라벨로 매핑
        emotion_label = emotion_mapping.get(final_emotion, "unknown")
        final_emotion_label = label_mapping.get(emotion_label,"unknown")
        return final_emotion_label


# 이미지 감정 분류기 초기화
def load_image_model(model_path):
    model = ImageEmotionClassifier(num_classes=3).to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    # 키 이름 변경: "conv1.weight" -> "resnet.conv1.weight"
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith("resnet."):  # "resnet." 접두사가 없으면 추가
            name = "resnet." + k
        else:
            name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model


def load_text_model(model_path):
    """텍스트 감정 분류기를 초기화하고 가중치를 로드합니다."""
    # BERT 모델 초기화
    bert_model = BertModel.from_pretrained(
        'skt/kobert-base-v1',
        add_pooling_layer=True,
        output_hidden_states=True
    )
    
    # TextEmotionClassifier 초기화
    model = TextEmotionClassifier(bert_model, num_classes=3).to(device)
    
    # 저장된 가중치 로드
    try:
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=device)
        
        # position_ids 관련 키 필터링
        filtered_state_dict = {k: v for k, v in checkpoint.items() 
                             if not k.endswith('position_ids')}
        
        # 필터링된 가중치 로드
        model.load_state_dict(filtered_state_dict, strict=False)
        print("Model weights loaded successfully with position_ids filtered out")
    except Exception as e:
        print(f"Error loading model weights: {str(e)}")
        raise
    
    model.eval()
    return model

# 텍스트 모델의 출력 후처리
def adjust_label_order(logits):
    # 텍스트 모델 출력: [분노, 슬픔, 기쁨]
    # 이미지 모델 순서: [분노, 기쁨, 슬픔]
    return logits[:, [0, 2, 1]]  # 순서 변경


# 이미지 감정 예측
def predict_image(model, image_path):
    if not os.path.exists(image_path):  # 이미지 파일이 없는 경우
        return None
    try:
        input_tensor = preprocess_image(image_path).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
        return predicted_class
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# 폴더 내 모든 이미지의 감정을 예측하고 최종 감정 결정
def predict_folder_emotion(model, image_folder):
    if not os.path.exists(image_folder):  # 폴더가 없는 경우
        return None, None

    # 감정 라벨 매핑
    emotion_mapping = {0: '0', 1: '1', 2: '2'}  # 앙상블을 위한 문자로 변환

    # 폴더 내 모든 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:  # 이미지 파일이 없는 경우
        return None, None

    # 각 이미지의 감정 예측
    emotion_counter = Counter()
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        predicted_class = predict_image(model, image_path)
        if predicted_class is not None:  # 이미지 처리 성공한 경우만 카운트
            emotion = emotion_mapping.get(predicted_class, "unknown")
            emotion_counter[emotion] += 1

    # 최종 감정 결정 (가장 빈도가 높은 감정)
    if not emotion_counter:  # 감정이 예측되지 않은 경우
        return None, None
    final_emotion = emotion_counter.most_common(1)[0][0]
    return final_emotion, emotion_counter
# 텍스트 감정 예측
def predict_text(model, tokenizer, text):
    if not text.strip():  # 텍스트가 비어 있는 경우
        return None
    try:
        input_ids, segment_ids, valid_length = preprocess_text(text, tokenizer)
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            logits = model(token_ids=input_ids, valid_length=valid_length, segment_ids=segment_ids).cpu().numpy()
            logits = adjust_label_order(logits)  # 라벨 순서 변경
            predicted_class = np.argmax(logits)
        return predicted_class
    except Exception as e:
        print(f"Error processing text: {e}")
        return None

import shutil

def predict_video(video_path, frame_dir):
    if not os.path.exists(video_path):  # 비디오 파일이 없는 경우
        return None, None
    try:
        # 모델 로드
        video_model = VideoModel()
        model = video_model.get_model()
        preprocess = video_model.get_preprocess()
        device = video_model.get_device()

        # 전처리 클래스 초기화
        preprocessor = VideoPreprocessor()

        # 예측 클래스 초기화
        predictor = VideoPredictor(model, preprocess, device)

        # 비디오에서 프레임 추출
        preprocessor.extract_frames_from_video(video_path, frame_dir, frame_rate=2)

        # 프레임별 감정 라벨 추출
        frame_emotions = predictor.get_frame_wise_emotions(frame_dir)

        # 최종 감정 라벨 추출
        final_emotion = predictor.get_final_emotion(frame_emotions)

        # 임시 파일 삭제
        shutil.rmtree(frame_dir)

        return frame_emotions, final_emotion
    except Exception as e:
        print(f"Error processing video: {e}")
        return None, None

def predict_final_emotion(text_emotion: str = None, image_emotion: str = None, video_emotion: str = None):
    """
    텍스트, 오디오, 비디오 감정 분류 결과를 입력받아 최종 감정을 반환합니다.
    :param text_emotion: 텍스트 감정 분류 결과 (예: '기쁨')
    :param image_emotion: 오디오 감정 분류 결과 (예: '슬픔')
    :param video_emotion: 비디오 감정 분류 결과 (예: '분노')
    :return: 최종 감정 라벨
    """
    ensemble_model = EmotionEnsemble()
    modality_emotion_dict = {
        'text': text_emotion,
        'image': image_emotion,
        'video': video_emotion
    }

    return ensemble_model.ensemble_emotion(modality_emotion_dict)


def main():
    # 이미지 모델 경로 설정
    image_model_path = 'final_resnet50_emotion_classifier_crawling_3.pth'
    text_model_path = 'trained_model_full_v2_3emotions_weights_new.pth'

    # 모델 로드
    image_model = load_image_model(image_model_path)
    text_model = load_text_model(text_model_path)
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

    # 이미지 감정 분류 테스트
    print("\n=== 이미지 감정 분류 테스트 ===")
    image_folder = r"./images" # 이 부분을 실제 이미지가 있는 폴더 경로로 수정해야 합니다
    
    # 폴더 내 이미지의 최종 감정 예측
    final_image_emotion, emotion_counter = predict_folder_emotion(image_model, image_folder)

    # 결과 출력
    print("\n=== 이미지 감정 분류 결과 ===")
    if emotion_counter is None:
        print("이미지를 찾을 수 없거나 처리할 수 없습니다.")
        print("이미지 폴더를 확인해주세요:", image_folder)
        final_image_emotion = None
    else:
        print("각 감정의 빈도:")
        for emotion, count in emotion_counter.items():
            print(f"{emotion}: {count}")
        print(f"\n최종 감정: {final_image_emotion}")

    # 텍스트 감정 분류 테스트
    print("\n=== 텍스트 감정 분류 테스트 ===")
    test_text = "다같이 화이팅"
    final_text_emotion = predict_text(text_model, tokenizer, test_text)
    print(f"문장: \"{test_text}\" → 예측 감정: {final_text_emotion}")

    # 비디오 감정 분류 테스트
    print("\n=== 비디오 감정 분류 테스트 ===")
    video_path = 'selfrag_workflow.mp4'
    frame_dir = 'output_frame_for_captioning'
    
    if not os.path.exists(video_path):
        print(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        frame_emotions, final_video_emotion = None, None
    else:
        frame_emotions, final_video_emotion = predict_video(video_path, frame_dir)
        
        if frame_emotions:
            print("Frame-wise emotions:")
            for i, emotion in enumerate(frame_emotions):
                print(f"Frame {i + 1}: {emotion}")
            print(f"\nFinal predicted emotion: {final_video_emotion}")
        else:
            print("비디오를 처리할 수 없습니다.")

    # 최종 감정 결정
    emotion_mapping = {0: '분노', 1: '기쁨', 2: '슬픔'}
    
    try:
        ensemble_emotion = predict_final_emotion(final_text_emotion, final_image_emotion, final_video_emotion)
        final_emotion_label = emotion_mapping.get(ensemble_emotion, "unknown")
        print(f"\nFinal Emotion: {final_emotion_label}")
    except Exception as e:
        print(f"\nError in ensemble prediction: {str(e)}")
        
if __name__ == "__main__":
    main()

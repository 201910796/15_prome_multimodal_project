from PIL import Image
import cv2
import os
import shutil
import torchvision.transforms as transforms
from kobert_tokenizer import KoBERTTokenizer

# 이미지 전처리 함수
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)  # 배치 차원 추가

# 텍스트 전처리 함수
def preprocess_text(text, tokenizer):
    tokenized = tokenizer(
        text,
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )
    valid_length = (tokenized['input_ids'] != tokenizer.pad_token_id).sum(dim=1)
    return tokenized['input_ids'], tokenized['token_type_ids'], valid_length


class VideoPreprocessor:
    """비디오 프레임 추출 및 전처리를 담당하는 클래스"""
    def __init__(self):
        pass

    def extract_frames_from_video(self, video_path, frame_dir, frame_rate=1):
        """비디오에서 프레임 추출"""
        # 기존 폴더 비우기
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)  # 폴더와 그 안의 모든 파일 삭제
        os.makedirs(frame_dir)  # 새로 폴더 생성

        # Use cv2.CAP_FFMPEG backend explicitly
        video_capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if not video_capture.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps == 0:
            print("Error: FPS value is 0, can't extract frames properly.")
            return

        print(f"Extracting frames from {video_path}...")

        frame_count = 0

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break  # End of video

            # 일정 간격으로 프레임 추출 (frame_rate에 따라)
            if frame_count % int(max(fps / frame_rate, 1)) == 0:
                frame_filename = os.path.join(frame_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)  # 프레임 저장
                print(f"Saved frame: {frame_filename}")  # 프레임 저장 상태 출력

            frame_count += 1

        video_capture.release()
        print(f"Extracted {frame_count} frames in total.")
import torch
import torch.nn as nn
import numpy as np
import math
import os
import re
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.nn import functional as F
import cv2
import logging
from pathlib import Path
from typing import Union, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MultimodalPreprocessor:
    def __init__(self, d_model: int = 768):
        """
        통합 멀티모달 전처리기 초기화
        Args:
            d_model (int): 출력 특징 벡터의 차원 (기본값: 768)
        """
        self.d_model = d_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # 위치 인코딩 모듈 초기화
        self.positional_encoder = PositionalEncoding(d_model=self.d_model).to(self.device)
        
    def _check_file(self, file_path: Union[str, Path]) -> Path:
        """파일 존재 여부 확인"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        return path

    def _initialize_text_model(self):
        """텍스트 모델 초기화 (Lazy loading)"""
        if not hasattr(self, 'text_model'):
            logger.info("Initializing KoBERT model...")
            self.text_model_name = 'monologg/kobert'
            self.text_tokenizer = BertTokenizer.from_pretrained(self.text_model_name)
            self.text_model = BertModel.from_pretrained(self.text_model_name)
            self.text_model = self.text_model.to(self.device)
            self.text_model.eval()

    def _initialize_image_model(self):
        """이미지 처리를 위한 VATT 스타일 모델 초기화 (Lazy loading)"""
        if not hasattr(self, 'patch_embed'):
            logger.info("Initializing Image model (VATT style)...")
            
            # 패치 크기와 이미지 크기 설정
            self.patch_size = 16
            self.image_size = 224
            self.num_patches = (self.image_size // self.patch_size) ** 2
            
            # 패치 임베딩을 위한 컨볼루션 레이어
            self.patch_embed = nn.Conv2d(
                in_channels=3,
                out_channels=self.d_model,
                kernel_size=self.patch_size,
                stride=self.patch_size
            ).to(self.device)
            
            # CLS 토큰 초기화 (학습 가능한 파라미터)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model)).to(self.device)
            
            # 위치 임베딩 초기화
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + 1, self.d_model)
            ).to(self.device)
            
            # 레이어 정규화
            self.norm = nn.LayerNorm(self.d_model).to(self.device)
            
            # 이미지 전처리를 위한 변환
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def _initialize_video_model(self):
        """비디오 모델 초기화 (Lazy loading)"""
        if not hasattr(self, 'video_model'):
            logger.info("Initializing Video model...")
            self.video_model = resnet50(pretrained=True)
            self.video_model.fc = nn.Linear(self.video_model.fc.in_features, self.d_model)
            self.video_model = self.video_model.to(self.device)
            self.video_model.eval()
            
            self.video_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def _initialize_audio_model(self):
        """오디오 모델 초기화 (Lazy loading)"""
        if not hasattr(self, 'audio_model'):
            logger.info("Initializing Wav2Vec2 model...")
            self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device)
            self.audio_model.eval()

    def preprocess_text(self, text: str, max_length: int = 128) -> np.ndarray:
        """텍스트 전처리 및 임베딩 추출"""
        try:
            # 모델 초기화 (lazy loading)
            self._initialize_text_model()
            
            # 텍스트 클리닝
            cleaned_text = re.sub(r'[^ㄱ-ㅎ가-힣a-zA-Z0-9\s]', '', text).strip()
            
            # BERT 입력 형식으로 변환
            inputs = self.text_tokenizer.encode_plus(
                cleaned_text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=max_length,
                add_special_tokens=True
            )
            
            # 임베딩 추출
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.text_model(**inputs)
                embeddings = outputs.last_hidden_state
                cls_embedding = embeddings[:, 0, :]  # [CLS] 토큰의 임베딩
                text_features = cls_embedding.cpu().numpy()[0]
            
            logger.info("Text processing completed successfully")
            return text_features
            
        except Exception as e:
            logger.error(f"텍스트 처리 중 오류 발생: {str(e)}")
            raise

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """VATT 스타일의 이미지 전처리 및 임베딩 추출"""
        try:
            # 파일 체크
            self._check_file(image_path)
            
            # 모델 초기화 (lazy loading)
            self._initialize_image_model()
            
            # 이미지 로드 및 전처리
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # 패치 임베딩 추출 (B, C, H, W) -> (B, D, H/P, W/P)
                patches = self.patch_embed(image_tensor)
                
                # 패치 재구성 (B, D, H/P, W/P) -> (B, N, D)
                B, D, H, W = patches.shape
                patches = patches.flatten(2).transpose(1, 2)
                
                # CLS 토큰 추가 (B, 1+N, D)
                cls_token = self.cls_token.expand(B, -1, -1)
                patches = torch.cat([cls_token, patches], dim=1)
                
                # 위치 임베딩 추가
                patches = patches + self.pos_embed
                
                # 레이어 정규화
                patches = self.norm(patches)
                
                # CLS 토큰의 특징만 추출 (B, D)
                image_features = patches[:, 0].cpu().numpy().squeeze()
            
            logger.info("Image processing completed successfully")
            return image_features
            
        except Exception as e:
            logger.error(f"이미지 처리 중 오류 발생: {str(e)}")
            raise

    def preprocess_video(self, video_path: str, fps: int = 1) -> np.ndarray:
        """비디오 전처리 및 임베딩 추출"""
        try:
            # 파일 체크
            self._check_file(video_path)
            
            # 모델 초기화 (lazy loading)
            self._initialize_video_model()
            
            # 비디오에서 프레임 추출
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            interval = max(1, frame_rate // fps)
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % interval == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            # 프레임별 특징 추출
            frame_features = []
            with torch.no_grad():
                for frame in frames:
                    input_tensor = self.video_transform(frame).unsqueeze(0).to(self.device)
                    features = self.video_model(input_tensor)
                    frame_features.append(features.cpu().numpy().squeeze())
            
            # 모든 프레임의 특징을 평균
            video_features = np.mean(frame_features, axis=0)
            
            logger.info("Video processing completed successfully")
            return video_features
            
        except Exception as e:
            logger.error(f"비디오 처리 중 오류 발생: {str(e)}")
            raise

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """WAV 오디오 파일 전처리 및 임베딩 추출"""
        try:
            # 파일 존재 확인 및 WAV 확장자 검증
            path = self._check_file(audio_path)
            if path.suffix.lower() != '.wav':
                raise ValueError(f"WAV 파일만 지원됩니다. 입력된 파일: {path}")
                
            logger.info(f"Processing WAV file: {path}")
            
            # 모델 초기화
            self._initialize_audio_model()
            
            # WAV 파일 로드 - soundfile 사용
            import soundfile as sf
            waveform, sample_rate = sf.read(str(path))
            
            # numpy 배열을 torch tensor로 변환
            waveform = torch.from_numpy(waveform).float()
            
            # 모노로 변환이 필요한 경우
            if len(waveform.shape) > 1:
                if waveform.shape[1] == 2:  # 스테레오
                    waveform = waveform.mean(dim=1)
                else:
                    waveform = waveform[:, 0]  # 첫 번째 채널 선택
            
            # 차원 추가 (채널 차원)
            waveform = waveform.unsqueeze(0)
                
            # 16kHz로 리샘플링
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                
            logger.info(f"Audio loaded successfully - Shape: {waveform.shape}, Sample rate: {sample_rate}Hz")
                
            # Wav2Vec2 입력 전처리
            inputs = self.audio_processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # 특징 추출
            with torch.no_grad():
                input_values = inputs.input_values.to(self.device)
                outputs = self.audio_model(input_values)
                audio_features = outputs.last_hidden_state
                audio_features = self.positional_encoder(audio_features)
                audio_features = audio_features.mean(dim=1).squeeze().cpu().numpy()
                
            logger.info("Audio processing completed successfully")
            return audio_features
            
        except Exception as e:
            logger.error(f"오디오 처리 중 오류 발생: {str(e)}")
            raise

    def preprocess_batch(self, items: List[dict]) -> dict:
        """배치 처리 함수
        Args:
            items: 처리할 아이템 리스트. 각 아이템은 {'type': str, 'path': str} 형식
        Returns:
            dict: 각 모달리티별 처리 결과
        """
        results = {}
        for item in items:
            try:
                item_type = item['type']
                if item_type == 'text':
                    result = self.preprocess_text(item['content'])
                elif item_type == 'video':
                    result = self.preprocess_video(item['path'])
                elif item_type == 'audio':
                    result = self.preprocess_audio(item['path'])
                else:
                    logger.warning(f"Unsupported modality type: {item_type}")
                    continue
                
                if item_type not in results:
                    results[item_type] = []
                results[item_type].append(result)
                
            except Exception as e:
                logger.error(f"배치 처리 중 오류 발생: {str(e)}")
                continue
                
        return results

    def __del__(self):
        """리소스 정리"""
        try:
            # GPU 메모리 정리
            if hasattr(self, 'text_model'):
                del self.text_model
            if hasattr(self, 'video_model'):
                del self.video_model
            if hasattr(self, 'audio_model'):
                del self.audio_model
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"리소스 정리 중 오류 발생: {str(e)}")

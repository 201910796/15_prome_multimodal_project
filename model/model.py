import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel
import clip

# 이미지 감정 분류를 위한 ResNet-50 모델
class ImageEmotionClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ImageEmotionClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# 텍스트 감정 분류를 위한 KoBERT 모델
class TextEmotionClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=3, dr_rate=0.5):
        super(TextEmotionClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
           
        ## 
        self.register_buffer(
            "position_ids",
            torch.arange(512).expand((1, -1))
        )

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    # def forward(self, token_ids, valid_length, segment_ids):
    #     attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
    #     ##
    #     position_ids = self.position_ids[:, :token_ids.size(1)]

    #     outputs = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
    #                         attention_mask=attention_mask.to(token_ids.device), return_dict=True)
    #     pooler = outputs.pooler_output
    #     if self.dr_rate:
    #         pooler = self.dropout(pooler)
    #     return self.classifier(pooler)
    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        outputs = self.bert(
            input_ids=token_ids,
            token_type_ids=segment_ids.long(),
            attention_mask=attention_mask.to(token_ids.device),
            return_dict=True
        )
        
        pooler = outputs.pooler_output
        if self.dr_rate:
            pooler = self.dropout(pooler)
        return self.classifier(pooler)
    
class VideoModel:
    """CLIP 모델을 로드하고 관리하는 클래스"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", self.device)  # CLIP 모델과 전처리 로드

    def get_model(self):
        """모델 반환"""
        return self.model

    def get_preprocess(self):
        """전처리 함수 반환"""
        return self.preprocess

    def get_device(self):
        """디바이스 반환"""
        return self.device

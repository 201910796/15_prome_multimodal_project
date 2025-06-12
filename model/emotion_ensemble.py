from collections import Counter
from typing import Dict, List

class EmotionEnsemble:
    def __init__(self):
        
        # 각 모달리티의 기본 가중치 초기화
        self.weights = {
            'text': 0.5,
            'image': 0.3,
            'video': 0.2
        }
        # 동점 처리 우선순위 (텍스트 > 오디오 > 비디오)
        self.priority = {'text': 0, 'image': 1, 'video': 2}

    def set_weights(self, modalities: List[str]):
        """
        입력된 모달리티에 따라 가중치를 조정합니다.
        """
        if 'text' in modalities and 'image' in modalities and 'video' in modalities:
            self.weights = {'text': 0.5, 'image': 0.3, 'video': 0.2}
        elif 'text' in modalities and ('image' in modalities or 'video' in modalities):
            self.weights = {'text': 0.7, 'image': 0.2, 'video': 0.1}
        elif 'image' in modalities and 'video' in modalities:
            self.weights = {'image': 0.5, 'video': 0.5}
        else:
            for modality in modalities:
                self.weights = {modality: 1.0}

    def ensemble_emotion(self, modality_emotion_dict: Dict[str, str]):
        """
        모달리티와 감정 라벨을 키-값으로 입력받아 가중치를 적용하고, 최종 감정을 결정합니다.
        :param modality_emotion_dict: {'text': '기쁨', 'image': '슬픔', 'video': '분노'} 형태의 딕셔너리
        :return: 최종 감정 라벨
        """
        # 입력된 모달리티 확인
        modalities = [modality for modality, emotion in modality_emotion_dict.items() if emotion is not None]
        self.set_weights(modalities)

        # 감정 점수를 저장할 딕셔너리
        emotion_scores = Counter()
        modality_map = {}

        # 각 모달리티의 감정에 가중치 적용
        for modality, emotion in modality_emotion_dict.items():
            if emotion is not None:
                weight = self.weights.get(modality, 0)
                emotion_scores[emotion] += weight
                if emotion not in modality_map:
                    modality_map[emotion] = []
                modality_map[emotion].append(modality)

        # 가중치 적용된 감정별 점수 및 해당 모달리티 출력
        print("Emotion Scores (weighted):")
        for emotion, score in emotion_scores.items():
            modalities = ', '.join(modality_map[emotion])
            print(f"{emotion}: {score:.2f} (from {modalities})")

        # 최종 감정 결정 (동점일 경우 우선순위 적용)
        max_score = max(emotion_scores.values(), default=0)
        top_emotions = [emotion for emotion, score in emotion_scores.items() if score == max_score]

        # 우선순위 기준 적용
        for modality in self.priority:
            for emotion in top_emotions:
                if any(mod for mod, emo in modality_emotion_dict.items() if emo == emotion and mod == modality):
                    return emotion

        

        return top_emotions[0]



import argparse
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple
import warnings

from preprocessing import MultimodalPreprocessor
from model import MultiModalClassifier

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Multimodal Emotion Analysis')
    parser.add_argument('--model_path', type=str, default='models/vatt_model.h5')
    parser.add_argument('--text', type=str, default="No emotion.")
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--audio', type=str, default=None)
    parser.add_argument('--video', type=str, default=None)
    return parser

def get_default_embeddings(dim: int = 768) -> np.ndarray:
    return np.zeros(dim)

def calculate_modality_weights(args) -> Dict[str, float]:
    weights = {
        'text': 1.0 if args.text and args.text != "No emotion." else 0.0,
        'image': 1.0 if args.image and Path(args.image).exists() else 0.0,
        'audio': 1.0 if args.audio and Path(args.audio).exists() else 0.0,
        'video': 1.0 if args.video and Path(args.video).exists() else 0.0
    }
    
    active_count = sum(1 for w in weights.values() if w > 0)
    if active_count > 0:
        for k in weights:
            weights[k] = weights[k] / active_count if weights[k] > 0 else 0.0
            
    return weights

def process_input(preprocessor: MultimodalPreprocessor, args) -> Tuple[Dict, Dict[str, float]]:
    embeddings = {}
    modality_weights = calculate_modality_weights(args)
    
    # Process each modality
    modalities = {
        'text': (args.text, preprocessor.preprocess_text),
        'image': (args.image, preprocessor.preprocess_image),
        'audio': (args.audio, preprocessor.preprocess_audio),
        'video': (args.video, preprocessor.preprocess_video)
    }
    
    for name, (input_data, process_fn) in modalities.items():
        try:
            if modality_weights[name] > 0:
                # 키 이름에서 '_input' 접미사 제거
                embeddings[name] = process_fn(input_data)
                logger.info(f"{name.capitalize()} processing completed")
            else:
                embeddings[name] = get_default_embeddings()
        except Exception as e:
            logger.error(f"{name.capitalize()} processing failed: {str(e)}")
            embeddings[name] = get_default_embeddings()
            modality_weights[name] = 0.0
    
    return embeddings, modality_weights

def print_results(args, emotion: str, confidence: float, 
                 probabilities: np.ndarray, classifier, modality_weights: Dict[str, float]) -> None:
    print("\n" + "="*50)
    
    # Input information
    print("\nInput Information:")
    inputs = {
        'Text': args.text if args.text != "No emotion." else "(None)",
        'Image': args.image if args.image else "(None)",
        'Audio': args.audio if args.audio else "(None)",
        'Video': args.video if args.video else "(None)"
    }
    for k, v in inputs.items():
        print(f"- {k}: {v}")
    
    # Modality weights information
    print("\nModality Weights:")
    for modality, weight in modality_weights.items():
        weight_percent = weight * 100
        bar_length = int(weight * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"{modality:8}: {weight_percent:6.2f}% {bar}")
    
    # Results
    print("\nEmotion Analysis Results:")
    print(f"Predicted Emotion: {emotion}")
    print(f"Confidence: {confidence:.2%}")
    
    # Emotion distribution
    print("\nEmotion Distribution:")
    for i, prob in enumerate(probabilities):
        emotion_name = classifier.emotion_labels[i]
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"{emotion_name:8}: {prob:6.2%} {bar}")
    
    print("\n" + "="*50)

def main():
    try:
        parser = create_parser()
        args = parser.parse_args()
        
        preprocessor = MultimodalPreprocessor()
        classifier = MultiModalClassifier()
        
        if Path(args.model_path).exists():
            classifier.model.load_weights(args.model_path)
        else:
            raise FileNotFoundError(f"Model not found: {args.model_path}")
        
        embeddings, modality_weights = process_input(preprocessor, args)
        emotion, confidence, probabilities = classifier.predict_emotion(
            embeddings, modality_weights
        )
        
        print_results(args, emotion, confidence, probabilities, classifier, modality_weights)
        
    except Exception as e:
        logger.error(f"Execution error: {str(e)}")
        raise

if __name__ == "__main__":
    main()

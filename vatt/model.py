import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.initializers import HeNormal
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalClassifier:
    def __init__(self, embedding_dim=768, num_classes=6, dropout_rate=0.5):
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l2_reg = 0.001
        
        self.emotion_labels = {
            0: "anger", 1: "disgust", 2: "fear",
            3: "joy", 4: "sad", 5: "surprise"
        }
        
        self._build_model()

    def _build_model(self):
        # Input layers
        text_input = layers.Input(shape=(self.embedding_dim,), name="text_input")
        image_input = layers.Input(shape=(self.embedding_dim,), name="image_input")
        audio_input = layers.Input(shape=(self.embedding_dim,), name="audio_input")
        video_input = layers.Input(shape=(self.embedding_dim,), name="video_input")

        # Dense layers for each modality
        text_x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001))(text_input)
        text_x = layers.BatchNormalization()(text_x)
        text_x = layers.Dropout(0.4)(text_x)

        image_x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001))(image_input)
        image_x = layers.BatchNormalization()(image_x)
        image_x = layers.Dropout(0.4)(image_x)

        audio_x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001))(audio_input)
        audio_x = layers.BatchNormalization()(audio_x)
        audio_x = layers.Dropout(0.4)(audio_x)

        video_x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001))(video_input)
        video_x = layers.BatchNormalization()(video_x)
        video_x = layers.Dropout(0.4)(video_x)

        # Modality fusion
        features = [text_x, image_x, audio_x, video_x]
        weighted_features = [layers.Lambda(lambda x: x * 0.25)(feat) for feat in features]
        merged_features = layers.Concatenate()(weighted_features)
        merged_features = layers.Reshape((4, 512))(merged_features)

        # Cross attention
        attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(merged_features, merged_features)
        attention_output = layers.Flatten()(attention_output)

        # Final classification layers
        x = layers.Dense(256, activation="relu", kernel_initializer=HeNormal(), 
                        kernel_regularizer=regularizers.l2(0.001))(attention_output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(128, activation="relu", kernel_initializer=HeNormal(),
                        kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        outputs = layers.Dense(self.num_classes)(x)

        self.model = Model(
            inputs=[text_input, image_input, audio_input, video_input],
            outputs=outputs,
            name="multimodal_emotion_classifier"
        )

    def _create_feature_processor(self, input_layer, name):
        x = layers.Dense(512, activation='gelu')(input_layer)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(0.3)(x)
        
        # Residual connection
        x = layers.Add()([
            layers.Dense(self.embedding_dim)(x),
            input_layer
        ])
        return x

    def _fuse_modalities(self, features):
        # Convert to sequence for attention
        feature_seq = layers.Lambda(
            lambda x: tf.stack(list(x.values()), axis=1)
        )(features)
        
        # Self-attention for modality interaction
        att_output = MultiHeadAttention(
            num_heads=8,
            key_dim=64,
            dropout=0.1
        )(feature_seq, feature_seq)
        
        # Add & Norm
        att_output = LayerNormalization(epsilon=1e-6)(
            layers.Add()([feature_seq, att_output])
        )
        
        return layers.Flatten()(att_output)

    def _create_classifier_head(self, x):
        for units in [256, 128]:
            x = layers.Dense(
                units,
                activation='gelu',
                kernel_initializer=HeNormal()
            )(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        return layers.Dense(self.num_classes)(x)

    def predict_emotion(self, embeddings: dict, modality_weights: dict) -> tuple:
        try:
            # 모델 입력을 위한 키 이름 변환
            inputs = {
                f"{k}_input": v[np.newaxis, ...]
                for k, v in embeddings.items()
            }
            
            logits = self.model.predict(inputs, verbose=0)[0]
            
            # Adaptive temperature scaling
            active_modalities = sum(1 for w in modality_weights.values() if w > 0)
            temperature = max(1.0, 6 - active_modalities)
            scaled_logits = logits / temperature
            
            probabilities = tf.nn.softmax(scaled_logits).numpy()
            
            confidence = self._calculate_confidence(
                probabilities, modality_weights, active_modalities
            )
            
            emotion_idx = np.argmax(probabilities)
            predicted_emotion = self.emotion_labels[emotion_idx]
            
            return predicted_emotion, confidence, probabilities
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def _calculate_confidence(self, probs, weights, active_modalities):
        # Entropy-based confidence
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = -np.log(1.0 / self.num_classes)
        entropy_conf = 1.0 - (entropy / max_entropy)
        
        # Modality-based confidence
        modality_conf = active_modalities / len(weights)
        
        # Probability-based confidence
        prob_conf = np.max(probs)
        
        # Weighted combination
        confidence = (0.4 * entropy_conf + 
                     0.3 * modality_conf + 
                     0.3 * prob_conf)
        
        return min(confidence, 0.95)  # Cap maximum confidence

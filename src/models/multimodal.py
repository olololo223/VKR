import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from src.utils.model_utils import load_and_rename_model, extract_features_model

class MultimodalEmotionRecognition:
    # В multimodal.py, замените метод __init__:

    def __init__(self, visual_model_path=None, audio_model_path=None, num_classes=7):
        self.num_classes = num_classes
        self.visual_model = None
        self.audio_model = None
        self.visual_feature_model = None
        self.audio_feature_model = None
        
        if visual_model_path:
            try:
                self.visual_model = tf.keras.models.load_model(visual_model_path, compile=False)
                print(f"Визуальная модель загружена. Выходной shape: {self.visual_model.output_shape}")
                
                # Создаем feature extractor из предобученной модели
                if len(self.visual_model.layers) > 1:
                    # Берем выход из предпоследнего слоя для features
                    visual_features = self.visual_model.layers[-2].output
                    self.visual_feature_model = tf.keras.Model(
                        inputs=self.visual_model.input,
                        outputs=visual_features
                    )
                    print(f"Визуальный feature extractor создан. Output: {visual_features.shape}")
                    
            except Exception as e:
                print(f"Ошибка загрузки визуальной модели: {e}")
        
        if audio_model_path:
            try:
                self.audio_model = tf.keras.models.load_model(audio_model_path, compile=False)
                print(f"Аудио модель загружена. Выходной shape: {self.audio_model.output_shape}")
                
                # Создаем feature extractor для аудио
                if len(self.audio_model.layers) > 1:
                    audio_features = self.audio_model.layers[-2].output
                    self.audio_feature_model = tf.keras.Model(
                        inputs=self.audio_model.input,
                        outputs=audio_features
                    )
                    print(f"Аудио feature extractor создан. Output: {audio_features.shape}")
                    
            except Exception as e:
                print(f"Ошибка загрузки аудио модели: {e}")
    
    def create_feature_fusion_model(self, visual_input_shape, audio_input_shape):
        """Создание улучшенной модели с fusion предобученных features"""
        
        # Визуальный вход и features
        visual_input = layers.Input(shape=visual_input_shape, name='visual_input')
        if self.visual_feature_model:
            visual_features = self.visual_feature_model(visual_input)
            print(f"Визуальные features shape: {visual_features.shape}")
        else:
            # Fallback: простая CNN
            x = layers.Conv2D(32, (3, 3), activation='relu', name='visual_conv1')(visual_input)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(64, (3, 3), activation='relu')(x)
            x = layers.GlobalAveragePooling2D()(x)
            visual_features = layers.Dense(128, activation='relu', name='visual_features')(x)

        # Аудио вход и features
        audio_input = layers.Input(shape=audio_input_shape, name='audio_input')
        if self.audio_feature_model:
            audio_features = self.audio_feature_model(audio_input)
            print(f"Аудио features shape: {audio_features.shape}")
        else:
            # Fallback: простая CNN для audio
            if len(audio_input_shape) == 3:  # MFCC features
                x = layers.Conv2D(32, (3, 3), activation='relu', name='audio_conv1')(audio_input)
                x = layers.MaxPooling2D((1, 2))(x)
                x = layers.GlobalAveragePooling2D()(x)
                audio_features = layers.Dense(128, activation='relu', name='audio_features')(x)
            else:
                x = layers.LSTM(64, return_sequences=True)(audio_input)
                x = layers.LSTM(32)(x)
                audio_features = layers.Dense(128, activation='relu', name='audio_features')(x)

        # Нормализация features
        visual_features = layers.BatchNormalization(name='visual_bn')(visual_features)
        audio_features = layers.BatchNormalization(name='audio_bn')(audio_features)
        
        # Обработка features (упрощенная для малого датасета)
        # Визуальная ветвь
        visual_processed = layers.Dense(256, activation='relu', name='visual_dense')(visual_features)
        visual_processed = layers.BatchNormalization()(visual_processed)
        visual_processed = layers.Dropout(0.3)(visual_processed)
        
        # Аудио ветвь
        audio_processed = layers.Dense(256, activation='relu', name='audio_dense')(audio_features)
        audio_processed = layers.BatchNormalization()(audio_processed)
        audio_processed = layers.Dropout(0.3)(audio_processed)
        
        # Объединение через concatenation
        combined = layers.Concatenate(name='feature_concat')([visual_processed, audio_processed])
        
        # Финальные слои (упрощенные)
        x = layers.Dense(512, activation='relu', name='fusion_dense1')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu', name='fusion_dense2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Выход
        output = layers.Dense(self.num_classes, activation='softmax', name='emotion_output')(x)
        
        model = models.Model(
            inputs=[visual_input, audio_input],
            outputs=output,
            name='multimodal_feature_fusion'
        )
        
        return model

    def create_attention_fusion_model(self, visual_input_shape, audio_input_shape):
        """Улучшенная модель с механизмом внимания для fusion"""
        
        # Визуальные features
        visual_input = layers.Input(shape=visual_input_shape, name='visual_input')
        if self.visual_feature_model:
            visual_features = self.visual_feature_model(visual_input)
        else:
            # Fallback
            x = layers.Conv2D(32, (3, 3), activation='relu')(visual_input)
            x = layers.GlobalAveragePooling2D()(x)
            visual_features = layers.Dense(128, activation='relu')(x)
        
        # Аудио features
        audio_input = layers.Input(shape=audio_input_shape, name='audio_input')
        if self.audio_feature_model:
            audio_features = self.audio_feature_model(audio_input)
        else:
            # Fallback
            if len(audio_input_shape) == 3:
                x = layers.Conv2D(32, (3, 3), activation='relu')(audio_input)
                x = layers.GlobalAveragePooling2D()(x)
                audio_features = layers.Dense(128, activation='relu')(x)
            else:
                x = layers.LSTM(64)(audio_input)
                audio_features = layers.Dense(128, activation='relu')(x)
        
        # Нормализация
        visual_features = layers.BatchNormalization()(visual_features)
        audio_features = layers.BatchNormalization()(audio_features)
        
        # Улучшенная обработка перед attention
        visual_dense = layers.Dense(512, activation='relu', name='visual_dense1')(visual_features)
        visual_dense = layers.BatchNormalization()(visual_dense)
        visual_dense = layers.Dropout(0.3)(visual_dense)
        visual_dense = layers.Dense(256, activation='relu', name='visual_dense2')(visual_dense)
        
        audio_dense = layers.Dense(512, activation='relu', name='audio_dense1')(audio_features)
        audio_dense = layers.BatchNormalization()(audio_dense)
        audio_dense = layers.Dropout(0.3)(audio_dense)
        audio_dense = layers.Dense(256, activation='relu', name='audio_dense2')(audio_dense)
        
        # Улучшенный механизм внимания - упрощенная версия
        # Создаем attention weights через shared MLP
        attention_input = layers.Concatenate()([visual_dense, audio_dense])
        attention_weights = layers.Dense(256, activation='relu', name='attention_mlp1')(attention_input)
        attention_weights = layers.Dense(2, activation='softmax', name='attention_weights')(attention_weights)
        
        # Взвешенное объединение с attention
        visual_att = layers.Multiply()([visual_dense, layers.Lambda(lambda x: x[:, 0:1])(attention_weights)])
        audio_att = layers.Multiply()([audio_dense, layers.Lambda(lambda x: x[:, 1:2])(attention_weights)])
        
        # Объединение через concatenation
        combined = layers.Concatenate()([visual_dense, audio_dense, visual_att, audio_att])
        
        # Финальные слои
        x = layers.Dense(512, activation='relu', name='fusion_dense1')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(512, activation='relu', name='fusion_dense2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu', name='fusion_dense3')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(
            inputs=[visual_input, audio_input],
            outputs=output,
            name='multimodal_attention_fusion'
        )
        
        return model
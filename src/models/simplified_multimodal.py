import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class SimplifiedMultimodal:
    def __init__(self, num_classes=7):
        self.num_classes = num_classes

    def create_visual_branch(self, input_shape):
        """Создание визуальной ветви"""
        inputs = layers.Input(shape=input_shape, name='visual_input')

        x = layers.Conv2D(32, (3, 3), activation='relu', name='visual_conv1')(inputs)
        x = layers.BatchNormalization(name='visual_bn1')(x)
        x = layers.MaxPooling2D((2, 2), name='visual_pool1')(x)
        x = layers.Dropout(0.25, name='visual_dropout1')(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', name='visual_conv2')(x)
        x = layers.BatchNormalization(name='visual_bn2')(x)
        x = layers.MaxPooling2D((2, 2), name='visual_pool2')(x)
        x = layers.Dropout(0.25, name='visual_dropout2')(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', name='visual_conv3')(x)
        x = layers.BatchNormalization(name='visual_bn3')(x)
        x = layers.GlobalAveragePooling2D(name='visual_gap')(x)

        x = layers.Dense(256, activation='relu', name='visual_dense1')(x)
        x = layers.BatchNormalization(name='visual_bn4')(x)
        x = layers.Dropout(0.5, name='visual_dropout3')(x)

        outputs = layers.Dense(128, activation='relu', name='visual_output')(x)

        return inputs, outputs

    def create_audio_branch(self, input_shape):
        """Создание аудио ветви"""
        inputs = layers.Input(shape=input_shape, name='audio_input')

        x = layers.Conv2D(32, (3, 3), activation='relu', name='audio_conv1')(inputs)
        x = layers.BatchNormalization(name='audio_bn1')(x)
        x = layers.MaxPooling2D((1, 2), name='audio_pool1')(x)
        x = layers.Dropout(0.25, name='audio_dropout1')(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', name='audio_conv2')(x)
        x = layers.BatchNormalization(name='audio_bn2')(x)
        x = layers.MaxPooling2D((1, 2), name='audio_pool2')(x)
        x = layers.Dropout(0.25, name='audio_dropout2')(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', name='audio_conv3')(x)
        x = layers.BatchNormalization(name='audio_bn3')(x)
        x = layers.GlobalAveragePooling2D(name='audio_gap')(x)

        x = layers.Dense(256, activation='relu', name='audio_dense1')(x)
        x = layers.BatchNormalization(name='audio_bn4')(x)
        x = layers.Dropout(0.5, name='audio_dropout3')(x)

        outputs = layers.Dense(128, activation='relu', name='audio_output')(x)

        return inputs, outputs

    def create_multimodal_model(self, visual_input_shape, audio_input_shape):
        """Создание полной мультимодальной модели
        Args:
            visual_input_shape: форма визуальных данных
            audio_input_shape: форма аудио данных
        """
        visual_input, visual_output = self.create_visual_branch(visual_input_shape)

        audio_input, audio_output = self.create_audio_branch(audio_input_shape)

        combined = layers.Concatenate(name='fusion_concat')([visual_output, audio_output])

        x = layers.Dense(512, activation='relu', name='fusion_dense1')(combined)
        x = layers.BatchNormalization(name='fusion_bn1')(x)
        x = layers.Dropout(0.5, name='fusion_dropout1')(x)

        x = layers.Dense(256, activation='relu', name='fusion_dense2')(x)
        x = layers.BatchNormalization(name='fusion_bn2')(x)
        x = layers.Dropout(0.3, name='fusion_dropout2')(x)

        x = layers.Dense(128, activation='relu', name='fusion_dense3')(x)
        x = layers.BatchNormalization(name='fusion_bn3')(x)
        x = layers.Dropout(0.2, name='fusion_dropout3')(x)

        output = layers.Dense(self.num_classes, activation='softmax', name='emotion_output')(x)

        model = models.Model(
            inputs=[visual_input, audio_input],
            outputs=output,
            name='multimodal_emotion_recognition'
        )

        return model

import tensorflow as tf
from tensorflow.keras import layers, models, applications


class TransferLearningModels:
    @staticmethod
    def create_mobilenetv2_model(input_shape=(48, 48, 3), num_classes=7):
        """MobileNetV2 с transfer learning"""
        base_model = applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )

        # Заморозка базовой модели
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    @staticmethod
    def create_efficientnet_model(input_shape=(48, 48, 3), num_classes=7):
        """EfficientNetB0 с transfer learning"""
        base_model = applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )

        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model
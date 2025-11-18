import tensorflow as tf
from tensorflow.keras import layers, models

class AudioModelFactory:
    def create_improved_model(self, input_shape, num_classes):
        model = models.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((1, 2)),
            layers.Dropout(0.3),

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((1, 2)),
            layers.Dropout(0.3),

            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((1, 2)),
            layers.Dropout(0.4),

            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),

            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(num_classes, activation='softmax')
        ])

        return model

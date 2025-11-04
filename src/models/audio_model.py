import tensorflow as tf
from tensorflow.keras import layers, models

class AudioModelFactory:
    @staticmethod
    def create_cnn_audio_model(input_shape=(13, 200, 1), num_classes=8):
        """CNN модель для обработки MFCC features"""
        model = models.Sequential([
            # Первый сверточный блок
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((1, 2)),  # Изменено с (2, 2) на (1, 2) чтобы сохранить размер по оси MFCC
            layers.Dropout(0.25),

            # Второй сверточный блок
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((1, 2)),  # Изменено с (2, 2) на (1, 2)
            layers.Dropout(0.25),

            # Третий сверточный блок
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((1, 2)),  # Изменено с (2, 2) на (1, 2)
            layers.Dropout(0.25),

            # Четвертый сверточный блок
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),

            # Полносвязные слои
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model

    @staticmethod
    def create_cnn_audio_model_v2(input_shape=(13, 200, 1), num_classes=8):
        """Альтернативная CNN архитектура с адаптивным пуллингом"""
        model = models.Sequential([
            # Первый сверточный блок - только по временной оси
            layers.Conv2D(32, (1, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((1, 2)),
            layers.Dropout(0.25),

            # Второй сверточный блок
            layers.Conv2D(64, (1, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((1, 2)),
            layers.Dropout(0.25),

            # Третий сверточный блок
            layers.Conv2D(128, (1, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((1, 2)),
            layers.Dropout(0.25),

            # Четвертый сверточный блок
            layers.Conv2D(256, (1, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),

            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model

    @staticmethod
    def create_simple_cnn_audio_model(input_shape=(13, 200, 1), num_classes=8):
        """Упрощенная CNN архитектура"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((1, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((1, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model

    @staticmethod
    def create_lstm_audio_model(input_shape=(200, 13), num_classes=8):
        """LSTM модель для обработки MFCC последовательностей"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.LSTM(128, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.LSTM(64),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(num_classes, activation='softmax')
        ])

        return model
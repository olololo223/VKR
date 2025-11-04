import tensorflow as tf
import numpy as np
import json
import os
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from src.models.audio_model import AudioModelFactory
from src.data_preprocessing.ravdess_audio_processor import RavdessAudioProcessor

class AudioModelTrainer:
    def __init__(self, config_path='config/model_config.json'):
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = json.load(file)

        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.audio_config = self.config['audio']

    def setup_callbacks(self, log_dir):
        """Настройка callback'ов для обучения"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=20,  # Увеличили patience
                restore_best_weights=True,
                verbose=1,
                monitor='val_accuracy',
                min_delta=0.01  # Минимальное улучшение
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=10,  # Увеличили patience
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(log_dir, 'best_audio_model.h5'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(log_dir, 'tensorboard'),
                histogram_freq=1
            ),
            tf.keras.callbacks.CSVLogger(
                filename=os.path.join(log_dir, 'training_log.csv')
            )
        ]

        return callbacks

    def get_class_weights(self, labels):
        """Вычисление весов классов для несбалансированных данных"""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        print(f"Class weights: {dict(enumerate(class_weights))}")
        return dict(enumerate(class_weights))

    def analyze_data(self, X_train, y_train, X_val, y_val):
        """Анализ данных перед обучением"""
        print("\n=== АНАЛИЗ ДАННЫХ ===")
        print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
        print(f"Val shapes: X={X_val.shape}, y={y_val.shape}")
        
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"Train distribution: {dict(zip(unique, counts))}")
        
        unique, counts = np.unique(y_val, return_counts=True)
        print(f"Val distribution: {dict(zip(unique, counts))}")
        
        # Проверка нормализации
        print(f"Data range - Min: {X_train.min():.3f}, Max: {X_train.max():.3f}, Mean: {X_train.mean():.3f}, Std: {X_train.std():.3f}")

    def create_improved_model(self, input_shape, num_classes):
        """Улучшенная архитектура модели"""
        model = tf.keras.Sequential([
            # Первый блок - больше фильтров
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((1, 2)),
            tf.keras.layers.Dropout(0.3),

            # Второй блок
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((1, 2)),
            tf.keras.layers.Dropout(0.3),

            # Третий блок
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((1, 2)),
            tf.keras.layers.Dropout(0.4),

            # Четвертый блок
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),

            # Полносвязные слои
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        return model

    def train_model(self, model_type='cnn_improved', data_path='data', processed_data_dir=None):
        """Обучение аудио модели"""
        # Создание директории для логов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/training/audio_{model_type}_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)

        # Подготовка данных
        processor = RavdessAudioProcessor()
        
        if processed_data_dir and os.path.exists(processed_data_dir):
            print("Загрузка ранее обработанных данных...")
            features, labels, metadata = processor.load_processed_data(processed_data_dir)
        else:
            print("Обработка аудио данных...")
            features, labels, metadata = processor.process_ravdess_audio_dataset(
                dataset_path=data_path,
                output_dir='data/processed/ravdess_audio'
            )

        # Подготовка данных для обучения
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.prepare_audio_data(
            features, labels
        )

        # Анализ данных
        self.analyze_data(X_train, y_train, X_val, y_val)

        # Создание модели
        input_shape = (self.audio_config['n_mfcc'], self.audio_config['max_length'], 1)
        num_classes = len(self.config['ravdess']['emotion_map'])
        
        print(f"\nСоздание модели: input_shape={input_shape}, num_classes={num_classes}")

        if model_type == 'cnn_improved':
            model = self.create_improved_model(input_shape, num_classes)
        elif model_type == 'cnn_simple':
            model = AudioModelFactory.create_simple_cnn_audio_model(input_shape, num_classes)
        elif model_type == 'lstm':
            # Для LSTM нужно транспонировать данные
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2])
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
            
            input_shape = (self.audio_config['max_length'], self.audio_config['n_mfcc'])
            model = AudioModelFactory.create_lstm_audio_model(input_shape, num_classes)
        else:
            raise ValueError("Неизвестный тип модели")

        # Компиляция модели с другим оптимизатором
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"Модель создана. Параметров: {model.count_params():,}")
        model.summary()

        # Callbacks
        callbacks = self.setup_callbacks(log_dir)

        # Вычисление весов классов
        class_weights = self.get_class_weights(y_train)

        print("\n=== НАЧАЛО ОБУЧЕНИЯ ===")
        
        # Обучение с увеличенным количеством эпох
        history = model.fit(
            X_train, y_train,
            batch_size=32,  # Фиксированный batch size
            epochs=100,     # Увеличили количество эпох
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
            shuffle=True
        )

        # Сохранение модели
        model.save(os.path.join(log_dir, 'final_audio_model.h5'))

        # Сохранение конфигурации
        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

        # Оценка на тестовых данных
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n=== РЕЗУЛЬТАТЫ ===")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        return model, history, test_accuracy


if __name__ == "__main__":
    trainer = AudioModelTrainer()
    model, history, accuracy = trainer.train_model(model_type='cnn_improved')
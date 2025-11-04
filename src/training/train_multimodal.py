import tensorflow as tf
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class MultimodalTrainer:
    def __init__(self, config_path='config/model_config.json'):
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = json.load(file)

        self.multimodal_config = self.config['multimodal']
        
        # Переменные для отслеживания лучшей точности
        self.best_accuracy = 0.0

    def setup_callbacks(self, log_dir):
        """Настройка callback'ов для обучения с сохранением по максимальной точности"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=50,  # Умеренное терпение
                restore_best_weights=True,
                verbose=1,
                monitor='val_accuracy',  # ← Следим за точностью
                mode='max',              # ← Сохраняем когда MAX
                min_delta=0.0005  # Увеличен min_delta для избежания ложных остановок
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,  # Умеренное уменьшение learning rate
                patience=15,  # Больше терпения перед уменьшением
                min_lr=1e-5,  # Минимальный learning rate выше, чтобы модель могла обучаться
                verbose=1,
                monitor='val_accuracy',  # ← Следим за точностью
                mode='max'
            ),
            # ГЛАВНЫЙ CALLBACK: сохраняем по val_accuracy
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(log_dir, 'best_multimodal_model.h5'),  # ← ТАКОЕ ЖЕ НАЗВАНИЕ
                save_best_only=True,
                monitor='val_accuracy',  # ← Следим за точностью
                mode='max',              # ← Сохраняем когда MAX
                verbose=1
            ),
            # Backup callback по val_loss
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(log_dir, 'best_val_loss_model.h5'),  # ← ТАКОЕ ЖЕ НАЗВАНИЕ
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(log_dir, 'tensorboard'),
                histogram_freq=1,
                update_freq='epoch'
            ),
            tf.keras.callbacks.CSVLogger(
                filename=os.path.join(log_dir, 'training_log.csv')  # ← ТАКОЕ ЖЕ НАЗВАНИЕ
            ),
        ]

        return callbacks

    def train_multimodal_model(self, visual_model_path, audio_model_path,
                               visual_data, audio_data, 
                               test_size=0.2, epochs=500, batch_size=32,
                               fusion_method='feature_fusion', use_augmentation=True):
        """Обучение мультимодальной модели с сохранением по максимальной точностью"""
        
        # Согласование меток
        visual_aligned, audio_aligned, common_emotions = self.prepare_multimodal_data(
            visual_data, audio_data
        )
        
        # Создание парных данных
        X_vis, X_aud, y_labels = self.create_paired_dataset(visual_aligned, audio_aligned)
        
        print(f"Создано {len(X_vis)} пар для обучения")
        print(f"Распределение меток: {np.bincount(y_labels)}")
        
        # Разделение на train и test
        indices = np.arange(len(X_vis))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=y_labels
        )
        
        # Дальнейшее разделение train на train и val
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.2, random_state=42, stratify=y_labels[train_idx]
        )
        
        # Подготовка данных для обучения
        X_vis_train = X_vis[train_idx]
        X_aud_train = X_aud[train_idx]
        y_train = y_labels[train_idx]
        
        X_vis_val = X_vis[val_idx]
        X_aud_val = X_aud[val_idx]
        y_val = y_labels[val_idx]
        
        X_vis_test = X_vis[test_idx]
        X_aud_test = X_aud[test_idx]
        y_test = y_labels[test_idx]
        
        print(f"Train: {X_vis_train.shape[0]}, Val: {X_vis_val.shape[0]}, Test: {X_vis_test.shape[0]}")
        
        # Вычисление весов классов
        class_weights = self.get_class_weights(y_train)

        # Создание мультимодальной модели
        try:
            from src.models.multimodal import MultimodalEmotionRecognition
            
            multimodal_system = MultimodalEmotionRecognition(
                visual_model_path, audio_model_path, num_classes=len(common_emotions)
            )

            if fusion_method == 'feature_fusion':
                multimodal_model = multimodal_system.create_feature_fusion_model(
                    visual_input_shape=X_vis_train.shape[1:],
                    audio_input_shape=X_aud_train.shape[1:]
                )
                print("Используется feature fusion модель")
            elif fusion_method == 'attention_fusion':
                multimodal_model = multimodal_system.create_attention_fusion_model(
                    visual_input_shape=X_vis_train.shape[1:],
                    audio_input_shape=X_aud_train.shape[1:]
                )
                print("Используется attention fusion модель")
            else:
                multimodal_model = multimodal_system.create_feature_fusion_model(
                    visual_input_shape=X_vis_train.shape[1:],
                    audio_input_shape=X_aud_train.shape[1:]
                )
                print("Используется feature fusion модель (по умолчанию)")
                
        except Exception as e:
            print(f"Ошибка при создании модели с предобученными весами: {e}")
            print("Используем упрощенную модель...")
            
            from src.models.simplified_multimodal import SimplifiedMultimodal
            multimodal_system = SimplifiedMultimodal(num_classes=len(common_emotions))
            multimodal_model = multimodal_system.create_multimodal_model(
                visual_input_shape=X_vis_train.shape[1:],
                audio_input_shape=X_aud_train.shape[1:]
            )
            print("Используется упрощенная модель")

        # Компиляция модели с улучшенными параметрами
        if 'feature_fusion' in fusion_method:
            lr = 0.0002  # Немного увеличен learning rate для лучшей сходимости
        else:
            lr = 0.0005  # Уменьшенный learning rate для attention модели
        
        # Используем Adam с обычным learning rate (ReduceLROnPlateau будет его регулировать)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Используем обычную loss функцию (label smoothing может мешать на малых данных)
        # Можно вернуть label smoothing позже, если нужно
        multimodal_model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Мультимодальная модель создана:")
        multimodal_model.summary()

        # Настройка callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/training/multimodal_{fusion_method}_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)

        callbacks = self.setup_callbacks(log_dir)

        print("Начало обучения мультимодальной модели...")
        print("✅ Модель будет сохраняться при улучшении VALIDATION ACCURACY")

        # Обучение модели
        if use_augmentation:
            print("Использование улучшенной аугментации...")
            
            # Улучшенная аугментация: более разнообразные техники
            X_vis_train_aug = []
            X_aud_train_aug = []
            y_train_aug = []
            
            for i in range(len(X_vis_train)):
                # Оригинальный sample
                X_vis_train_aug.append(X_vis_train[i])
                X_aud_train_aug.append(X_aud_train[i])
                y_train_aug.append(y_train[i])
                
                # Аугментированные samples
                aug_prob = np.random.random()
                
                if aug_prob > 0.3:  # 70% вероятность аугментации
                    # Горизонтальное отражение
                    if len(X_vis_train[i].shape) == 2:  # Grayscale
                        X_vis_aug = np.flip(X_vis_train[i], axis=1)
                    else:  # Color
                        X_vis_aug = np.flip(X_vis_train[i], axis=1)
                    X_vis_train_aug.append(X_vis_aug)
                    X_aud_train_aug.append(X_aud_train[i])
                    y_train_aug.append(y_train[i])
                
                if aug_prob > 0.6:  # 40% вероятность дополнительной аугментации
                    # Небольшой шум
                    if len(X_vis_train[i].shape) == 2:
                        X_vis_aug = X_vis_train[i] + np.random.normal(0, 0.02, X_vis_train[i].shape)
                        X_vis_aug = np.clip(X_vis_aug, 0, 1)
                    else:
                        X_vis_aug = X_vis_train[i] + np.random.normal(0, 0.02, X_vis_train[i].shape)
                        X_vis_aug = np.clip(X_vis_aug, 0, 1)
                    X_vis_train_aug.append(X_vis_aug)
                    X_aud_train_aug.append(X_aud_train[i])
                    y_train_aug.append(y_train[i])
                
                if aug_prob > 0.8:  # 20% вероятность дополнительной аугментации аудио
                    # Небольшой шум для аудио
                    X_aud_aug = X_aud_train[i] + np.random.normal(0, 0.01, X_aud_train[i].shape)
                    X_aud_aug = np.clip(X_aud_aug, 0, 1)
                    X_vis_train_aug.append(X_vis_train[i])
                    X_aud_train_aug.append(X_aud_aug)
                    y_train_aug.append(y_train[i])
            
            X_vis_train_aug = np.array(X_vis_train_aug)
            X_aud_train_aug = np.array(X_aud_train_aug)
            y_train_aug = np.array(y_train_aug)
            
            print(f"Расширенный набор данных: {len(X_vis_train_aug)} samples")
            
            history = multimodal_model.fit(
                [X_vis_train_aug, X_aud_train_aug], y_train_aug,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=([X_vis_val, X_aud_val], y_val),
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1,
                shuffle=True
            )
        else:
            print("Использование обычного обучения с class_weight...")
            history = multimodal_model.fit(
                [X_vis_train, X_aud_train], y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=([X_vis_val, X_aud_val], y_val),
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1,
                shuffle=True
            )

        # Сохранение финальной модели
        multimodal_model.save(os.path.join(log_dir, 'final_multimodal_model.h5'))

        # Сохранение конфигурации обучения
        training_config = {
            'visual_model_path': visual_model_path,
            'audio_model_path': audio_model_path,
            'common_emotions': common_emotions,
            'fusion_method': fusion_method,
            'training_params': {
                'batch_size': batch_size,
                'epochs': epochs,
                'test_size': test_size,
                'use_augmentation': use_augmentation,
                'learning_rate': lr
            },
            'data_info': {
                'train_samples': len(X_vis_train),
                'val_samples': len(X_vis_val),
                'test_samples': len(X_vis_test),
                'class_distribution': np.bincount(y_train).tolist(),
                'class_weights': class_weights
            }
        }
        
        with open(os.path.join(log_dir, 'training_config.json'), 'w') as f:
            json.dump(training_config, f, indent=2)

        # Оценка на тестовых данных
        print("\nОценка на тестовых данных...")
        test_loss, test_accuracy = multimodal_model.evaluate(
            [X_vis_test, X_aud_test], y_test, verbose=1
        )
        
        # Предсказания для дополнительной аналитики
        y_pred = multimodal_model.predict([X_vis_test, X_aud_test], verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n" + "="*50)
        print("ТЕСТОВЫЕ РЕЗУЛЬТАТЫ")
        print("="*50)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=common_emotions))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_classes))
        
        # Сохранение результатов
        results = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'classification_report': classification_report(y_test, y_pred_classes, target_names=common_emotions, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred_classes).tolist()
        }
        
        with open(os.path.join(log_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        return multimodal_model, history, test_accuracy

    # Остальные методы остаются без изменений
    def align_emotions(self):
        """Согласование эмоций между FER2013 и RAVDESS"""
        fer_to_common = {
            0: 0, 1: 4, 2: 2, 3: 1, 4: 3, 5: 5, 6: 6
        }
        ravdess_to_common = {
            0: 6, 1: 6, 2: 1, 3: 3, 4: 0, 5: 2, 6: 4, 7: 5
        }
        common_emotions = ['angry', 'happy', 'fear', 'sad', 'disgust', 'surprise', 'neutral']
        return fer_to_common, ravdess_to_common, common_emotions

    def prepare_multimodal_data(self, visual_data, audio_data):
        """Подготовка мультимодальных данных"""
        X_visual, y_visual = visual_data
        X_audio, y_audio = audio_data
        
        fer_to_common, ravdess_to_common, common_emotions = self.align_emotions()
        
        y_visual_aligned = np.array([fer_to_common[label] for label in y_visual])
        y_audio_aligned = np.array([ravdess_to_common[label] for label in y_audio])
        
        print(f"Общие эмоции: {common_emotions}")
        print(f"Visual labels distribution: {np.bincount(y_visual_aligned)}")
        print(f"Audio labels distribution: {np.bincount(y_audio_aligned)}")
        
        return (X_visual, y_visual_aligned), (X_audio, y_audio_aligned), common_emotions

    def create_balanced_paired_dataset(self, visual_data, audio_data, samples_per_class=200):
        """Создание сбалансированного парного датасета"""
        X_visual, y_visual = visual_data
        X_audio, y_audio = audio_data
        
        visual_indices = []
        audio_indices = []
        
        print("Создание сбалансированного датасета...")
        
        for label in np.unique(y_visual):
            vis_idx = np.where(y_visual == label)[0]
            aud_idx = np.where(y_audio == label)[0]
            
            n_pairs = min(len(vis_idx), len(aud_idx), samples_per_class)
            
            if n_pairs > 0:
                vis_selected = np.random.choice(vis_idx, n_pairs, replace=False)
                aud_selected = np.random.choice(aud_idx, n_pairs, replace=False)
                
                visual_indices.extend(vis_selected)
                audio_indices.extend(aud_selected)
                
                print(f"Класс {label}: {n_pairs} пар")
        
        indices = np.random.permutation(len(visual_indices))
        visual_indices = np.array(visual_indices)[indices]
        audio_indices = np.array(audio_indices)[indices]
        
        X_visual_paired = X_visual[visual_indices]
        X_audio_paired = X_audio[audio_indices]
        y_paired = y_visual[visual_indices]
        
        print(f"Создано {len(X_visual_paired)} сбалансированных пар")
        
        return (X_visual_paired, X_audio_paired, y_paired)

    def create_paired_dataset(self, visual_data, audio_data, paired_indices=None):
        """Создание парных данных для мультимодального обучения"""
        X_visual, y_visual = visual_data
        X_audio, y_audio = audio_data
        
        if paired_indices is None:
            return self.create_balanced_paired_dataset(visual_data, audio_data)
        else:
            visual_indices, audio_indices = paired_indices
        
        X_visual_paired = X_visual[visual_indices]
        X_audio_paired = X_audio[audio_indices]
        y_paired = y_visual[visual_indices]
        
        return (X_visual_paired, X_audio_paired, y_paired)

    def get_class_weights(self, labels):
        """Вычисление весов классов для несбалансированных данных"""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        weights_dict = {int(k): float(v) for k, v in enumerate(class_weights)}
        print(f"Веса классов: {weights_dict}")
        return weights_dict


if __name__ == "__main__":
    trainer = MultimodalTrainer()
    print("Multimodal trainer initialized")
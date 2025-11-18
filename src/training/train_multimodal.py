import tensorflow as tf
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.models.multimodal import MultimodalEmotionRecognition



class MultimodalTrainer:
    def __init__(self, config_path='config/model_config.json'):
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = json.load(file)

        self.multimodal_config = self.config['multimodal']
        self.best_accuracy = 0.0

    def setup_callbacks(self, log_dir):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=50,
                restore_best_weights=True,
                verbose=1,
                monitor='val_accuracy',
                mode='max',
                min_delta=0.0005
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=15,
                min_lr=1e-5,
                verbose=1,
                monitor='val_accuracy',
                mode='max'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(log_dir, 'best_multimodal_model.h5'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                filename=os.path.join(log_dir, 'training_log.csv')
            ),
        ]

        return callbacks

    def train_multimodal_model(self, visual_model_path, audio_model_path,
                               visual_data, audio_data,
                               test_size=0.2, epochs=500, batch_size=32):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/training/multimodal_{fusion_method}_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)

        visual_aligned, audio_aligned, common_emotions = self.prepare_multimodal_data(
            visual_data, audio_data
        )

        X_vis, X_aud, y_labels = self.create_paired_dataset(visual_aligned, audio_aligned)

        indices = np.arange(len(X_vis))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=y_labels
        )

        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.2, random_state=42, stratify=y_labels[train_idx]
        )

        X_vis_train = X_vis[train_idx]
        X_aud_train = X_aud[train_idx]
        y_train = y_labels[train_idx]

        X_vis_val = X_vis[val_idx]
        X_aud_val = X_aud[val_idx]
        y_val = y_labels[val_idx]

        X_vis_test = X_vis[test_idx]
        X_aud_test = X_aud[test_idx]
        y_test = y_labels[test_idx]

        class_weights = self.get_class_weights(y_train)

        try:
            multimodal_system = MultimodalEmotionRecognition(
                visual_model_path, audio_model_path, num_classes=len(common_emotions)
            )

            multimodal_model = multimodal_system.create_feature_fusion_model(
                visual_input_shape=X_vis_train.shape[1:],
                audio_input_shape=X_aud_train.shape[1:]
            )
        except Exception as e:
            print(f"Ошибка: {e}")
        
        multimodal_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = self.setup_callbacks(log_dir)

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

        multimodal_model.save(os.path.join(log_dir, 'final_multimodal_model.h5'))

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


        test_loss, test_accuracy = multimodal_model.evaluate(
            [X_vis_test, X_aud_test], y_test, verbose=1
        )

        y_pred = multimodal_model.predict([X_vis_test, X_aud_test], verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        print(f"Test Accuracy: {test_accuracy:.4f}")


        return multimodal_model, history, test_accuracy

    def align_emotions(self):
        fer_to_common = {
            0: 0, 1: 4, 2: 2, 3: 1, 4: 3, 5: 5, 6: 6
        }
        ravdess_to_common = {
            0: 6, 1: 6, 2: 1, 3: 3, 4: 0, 5: 2, 6: 4, 7: 5
        }
        common_emotions = ['angry', 'happy', 'fear', 'sad', 'disgust', 'surprise', 'neutral']
        return fer_to_common, ravdess_to_common, common_emotions

    def prepare_multimodal_data(self, visual_data, audio_data):
        X_visual, y_visual = visual_data
        X_audio, y_audio = audio_data

        fer_to_common, ravdess_to_common, common_emotions = self.align_emotions()

        y_visual_aligned = np.array([fer_to_common[label] for label in y_visual])
        y_audio_aligned = np.array([ravdess_to_common[label] for label in y_audio])


        return (X_visual, y_visual_aligned), (X_audio, y_audio_aligned), common_emotions

    def create_balanced_paired_dataset(self, visual_data, audio_data, samples_per_class=200):
        X_visual, y_visual = visual_data
        X_audio, y_audio = audio_data

        visual_indices = []
        audio_indices = []

        for label in np.unique(y_visual):
            vis_idx = np.where(y_visual == label)[0]
            aud_idx = np.where(y_audio == label)[0]

            n_pairs = min(len(vis_idx), len(aud_idx), samples_per_class)

            if n_pairs > 0:
                vis_selected = np.random.choice(vis_idx, n_pairs, replace=False)
                aud_selected = np.random.choice(aud_idx, n_pairs, replace=False)

                visual_indices.extend(vis_selected)
                audio_indices.extend(aud_selected)

        indices = np.random.permutation(len(visual_indices))
        visual_indices = np.array(visual_indices)[indices]
        audio_indices = np.array(audio_indices)[indices]

        X_visual_paired = X_visual[visual_indices]
        X_audio_paired = X_audio[audio_indices]
        y_paired = y_visual[visual_indices]

        return (X_visual_paired, X_audio_paired, y_paired)

    def create_paired_dataset(self, visual_data, audio_data, paired_indices=None):
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
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        weights_dict = {int(k): float(v) for k, v in enumerate(class_weights)}
        return weights_dict


if __name__ == "__main__":
    trainer = MultimodalTrainer()

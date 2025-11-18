import tensorflow as tf
import numpy as np
import json
import os
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from src.models.audio_model import AudioModelFactory
from src.data_preprocessing.ravdess_audio_processor import RavdessAudioProcessor
from tensorflow.keras import layers, models, regularizers, callbacks, optimizers

class AudioModelTrainer:
    def __init__(self, config_path='config/model_config.json'):
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = json.load(file)

        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.audio_config = self.config['audio']

    def setup_callbacks(self, log_dir):
        callbacks = [
            callbacks.EarlyStopping(
                patience=20,
                restore_best_weights=True,
                verbose=1,
                monitor='val_accuracy',
                min_delta=0.01
            ),
            callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(log_dir, 'best_audio_model.h5'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            callbacks.CSVLogger(
                filename=os.path.join(log_dir, 'training_log.csv')
            )
        ]

        return callbacks

    def get_class_weights(self, labels):
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        return dict(enumerate(class_weights))


    def train_model(self, data_path='data', processed_data_dir=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/training/audio_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)

        processor = RavdessAudioProcessor()

        if processed_data_dir and os.path.exists(processed_data_dir):
            features, labels, metadata = processor.load_processed_data(processed_data_dir)
        else:
            features, labels, metadata = processor.process_ravdess_audio_dataset(
                dataset_path=data_path,
                output_dir='data/processed/ravdess_audio'
            )

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.prepare_audio_data(
            features, labels
        )


        input_shape = (self.audio_config['n_mfcc'], self.audio_config['max_length'], 1)
        num_classes = len(self.config['ravdess']['emotion_map'])
        
        model = AudioModelFactory.create_improved_model(input_shape, num_classes)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = self.setup_callbacks(log_dir)
        class_weights = self.get_class_weights(y_train)

        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
            shuffle=True
        )

        model.save(os.path.join(log_dir, 'final_audio_model.h5'))

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")

        return model, history, test_accuracy


if __name__ == "__main__":
    trainer = AudioModelTrainer()
    model, history, accuracy = trainer.train_model()

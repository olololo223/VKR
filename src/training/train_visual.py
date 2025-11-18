import tensorflow as tf
import numpy as np
import json
import os
from datetime import datetime
from src.data_preprocessing.fer_processor import FERProcessor
from src.models.cnn_models import CNNModelFactory

class VisualModelTrainer:
    def __init__(self, config_path='config/model_config.json'):
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = json.load(file)

        self.model_config = self.config['model']
        self.training_config = self.config['training']

    def setup_callbacks(self, log_dir):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=self.training_config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=self.training_config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(log_dir, 'best_model.h5'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                filename=os.path.join(log_dir, 'training_log.csv')
            )
        ]

        return callbacks

    def train_model(self, data_path='data/train'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/training/visual_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)

        processor = FERProcessor()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.prepare_data(data_path)

        model = CNNModelFactory.create_cnn()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = self.setup_callbacks(log_dir)
        class_weights = processor.get_class_weights(y_train)

        history = model.fit(
            X_train, y_train,
            batch_size=self.training_config['batch_size'],
            epochs=self.training_config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        model.save(os.path.join(log_dir, 'final_model.h5'))

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")

        return model, history, test_accuracy


if __name__ == "__main__":
    trainer = VisualModelTrainer()
    model, history, accuracy = trainer.train_model()

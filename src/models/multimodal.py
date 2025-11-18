import tensorflow as tf
from tensorflow.keras import layers, models

class MultimodalEmotionRecognition:

    def __init__(self, visual_model_path=None, audio_model_path=None, num_classes=7):
        self.num_classes = num_classes
        self.visual_model = None
        self.audio_model = None
        self.visual_feature_model = None
        self.audio_feature_model = None

        if visual_model_path:
            try:
                self.visual_model = tf.keras.models.load_model(visual_model_path, compile=False)
                if len(self.visual_model.layers) > 1:
                    visual_features = self.visual_model.layers[-2].output
                    self.visual_feature_model = tf.keras.Model(
                        inputs=self.visual_model.input,
                        outputs=visual_features
                    )
            except Exception as e:
                print(f"Ошибка загрузки визуальной модели: {e}")

        if audio_model_path:
            try:
                self.audio_model = tf.keras.models.load_model(audio_model_path, compile=False)
                if len(self.audio_model.layers) > 1:
                    audio_features = self.audio_model.layers[-2].output
                    self.audio_feature_model = tf.keras.Model(
                        inputs=self.audio_model.input,
                        outputs=audio_features
                    )
            except Exception as e:
                print(f"Ошибка загрузки аудио модели: {e}")

    def create_feature_fusion_model(self, visual_input_shape, audio_input_shape):
        visual_input = layers.Input(shape=visual_input_shape, name='visual_input')
        visual_features = self.visual_feature_model(visual_input)

        audio_input = layers.Input(shape=audio_input_shape, name='audio_input')
        audio_features = self.audio_feature_model(audio_input)

        visual_features = layers.BatchNormalization(name='visual_bn')(visual_features)
        audio_features = layers.BatchNormalization(name='audio_bn')(audio_features)

        visual_processed = layers.Dense(256, activation='relu', name='visual_dense')(visual_features)
        visual_processed = layers.BatchNormalization()(visual_processed)
        visual_processed = layers.Dropout(0.3)(visual_processed)

        audio_processed = layers.Dense(256, activation='relu', name='audio_dense')(audio_features)
        audio_processed = layers.BatchNormalization()(audio_processed)
        audio_processed = layers.Dropout(0.3)(audio_processed)

        combined = layers.Concatenate(name='feature_concat')([visual_processed, audio_processed])

        x = layers.Dense(512, activation='relu', name='fusion_dense1')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(256, activation='relu', name='fusion_dense2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        output = layers.Dense(self.num_classes, activation='softmax', name='emotion_output')(x)

        model = models.Model(
            inputs=[visual_input, audio_input],
            outputs=output,
            name='multimodal_feature_fusion'
        )

        return model

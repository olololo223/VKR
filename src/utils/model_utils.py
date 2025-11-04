import tensorflow as tf
import os

def load_and_rename_model(model_path, new_name):
    """Загрузка модели с переименованием всех слоев"""
    model = tf.keras.models.load_model(model_path)
    
    # Переименовываем модель
    model._name = new_name
    
    # Переименовываем все слои
    for i, layer in enumerate(model.layers):
        layer._name = f"{new_name}_{layer.name}_{i}"
    
    return model

def extract_features_model(model, layer_name=None):
    """Создание модели для извлечения features"""
    if layer_name:
        # Используем определенный слой для features
        feature_layer = model.get_layer(layer_name)
        feature_model = tf.keras.Model(
            inputs=model.input,
            outputs=feature_layer.output
        )
    else:
        # Используем выход предпоследнего слоя (before softmax)
        feature_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.layers[-2].output
        )
    
    return feature_model
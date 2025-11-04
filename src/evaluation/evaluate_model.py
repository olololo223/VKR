import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse


class ModelEvaluator:
    def __init__(self, emotion_labels):
        self.emotion_labels = emotion_labels

    def evaluate_model(self, model, X_test, y_test, verbose=True):
        """Полная оценка модели"""
        
        # Предсказания
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        if verbose:
            print("="*50)
            print("Оценка модели")
            print("="*50)
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print("="*50)
            
            # По эмоциям
            print("\nМетрики по эмоциям:")
            precision_per_class = precision_score(y_test, y_pred, average=None)
            recall_per_class = recall_score(y_test, y_pred, average=None)
            f1_per_class = f1_score(y_test, y_pred, average=None)
            
            for i, emotion in enumerate(self.emotion_labels):
                print(f"{emotion:12s} - P: {precision_per_class[i]:.3f}, "
                      f"R: {recall_per_class[i]:.3f}, F1: {f1_per_class[i]:.3f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_probs
        }

    def evaluate_multimodal_model(self, model, X_vis_test, X_aud_test, y_test, verbose=True):
        """Оценка мультимодальной модели"""
        
        # Предсказания
        y_pred_probs = model.predict([X_vis_test, X_aud_test], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        if verbose:
            print("="*50)
            print("Оценка мультимодальной модели")
            print("="*50)
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print("="*50)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_probs
        }

    def get_sample_predictions(self, model, X, y_true, num_samples=10):
        """Получить примеры предсказаний"""
        indices = np.random.choice(len(X), num_samples, replace=False)
        
        results = []
        for idx in indices:
            if len(X.shape) == 5:  # Multimodal
                X_vis = np.expand_dims(X[0][idx], 0)
                X_aud = np.expand_dims(X[1][idx], 0)
                prediction = model.predict([X_vis, X_aud], verbose=0)
            else:
                X_sample = np.expand_dims(X[idx], 0)
                prediction = model.predict(X_sample, verbose=0)
            
            pred_label = np.argmax(prediction)
            confidence = np.max(prediction)
            
            results.append({
                'index': idx,
                'true_label': self.emotion_labels[y_true[idx]],
                'predicted_label': self.emotion_labels[pred_label],
                'confidence': confidence,
                'correct': y_true[idx] == pred_label
            })
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Оценка модели')
    parser.add_argument('--model_path', type=str, required=True, help='Путь к модели')
    parser.add_argument('--data_path', type=str, required=True, help='Путь к тестовым данным')
    parser.add_argument('--mode', type=str, choices=['visual', 'audio', 'multimodal'],
                       required=True, help='Тип модели')
    
    args = parser.parse_args()
    
    # Загрузка данных
    data = np.load(args.data_path)
    
    # Загрузка модели
    model = tf.keras.models.load_model(args.model_path)
    
    # Оценка
    evaluator = ModelEvaluator(emotion_labels=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    
    if args.mode == 'multimodal':
        X_vis, X_aud, y = data['X_vis'], data['X_aud'], data['y']
        evaluator.evaluate_multimodal_model(model, X_vis, X_aud, y)
    else:
        X, y = data['X'], data['y']
        evaluator.evaluate_model(model, X, y)


if __name__ == "__main__":
    main()

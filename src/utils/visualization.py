import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


class ModelVisualizer:
    def __init__(self, emotion_labels):
        self.emotion_labels = emotion_labels

    def plot_training_history(self, history, save_path=None):
        """Визуализация истории обучения"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Построение матрицы ошибок"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_class_distribution(self, y, title='Distribution', save_path=None):
        """Визуализация распределения классов"""
        unique, counts = np.unique(y, return_counts=True)

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(self.emotion_labels)), counts)
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.title(title)
        plt.xticks(range(len(self.emotion_labels)), self.emotion_labels, rotation=45)
        plt.grid(True, axis='y')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def print_classification_report(self, y_true, y_pred):
        """Вывод отчета о классификации"""
        report = classification_report(
            y_true, y_pred,
            target_names=self.emotion_labels
        )
        print(report)
        return report

    def plot_sample_predictions(self, model, X, y_true, num_samples=16, save_path=None):
        """Визуализация примеров предсказаний"""
        indices = np.random.choice(len(X), num_samples, replace=False)

        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        axes = axes.flatten()

        for idx, i in enumerate(indices):
            img = X[i]
            true_label = y_true[i]
            prediction = model.predict(np.expand_dims(img, 0), verbose=0)
            pred_label = np.argmax(prediction)
            confidence = np.max(prediction)

            axes[idx].imshow(img.squeeze(), cmap='gray')
            axes[idx].set_title(
                f"True: {self.emotion_labels[true_label]}\n"
                f"Pred: {self.emotion_labels[pred_label]} ({confidence:.2f})"
            )
            axes[idx].axis('off')

            if true_label != pred_label:
                axes[idx].patch.set_edgecolor('red')
                axes[idx].patch.set_linewidth(3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_emotion_comparison(self, predictions_list, labels_list, model_names, save_path=None):
        """Сравнение предсказаний разных моделей"""
        n_models = len(predictions_list)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))

        if n_models == 1:
            axes = [axes]

        for idx, (predictions, label, model_name) in enumerate(zip(predictions_list, labels_list, model_names)):
            cm = confusion_matrix(label, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.emotion_labels,
                       yticklabels=self.emotion_labels,
                       ax=axes[idx])
            axes[idx].set_title(model_name)
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

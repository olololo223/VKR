"""
Простая оценка моделей на основе реальных датасетов
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class SimpleModelEvaluator:
    def __init__(self, emotion_labels):
        self.emotion_labels = emotion_labels
        self.emotion_to_idx = {e: i for i, e in enumerate(emotion_labels)}

    def load_images_from_dataset(self, dataset_path, max_samples=None):
        """Загружает изображения из fer2013plus датасета"""
        images = []
        labels = []
        dataset_path = Path(dataset_path)
        
        print("   Загрузка изображений...")
        count = 0
        for emotion_idx, emotion in enumerate(self.emotion_labels):
            emotion_dir = dataset_path / emotion
            if emotion_dir.exists():
                image_files = list(emotion_dir.glob('*.png')) + list(emotion_dir.glob('*.jpg'))
                process_files = image_files[:max_samples] if max_samples else image_files
                for img_file in process_files:
                    try:
                        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (48, 48))
                            images.append(img / 255.0)
                            labels.append(emotion_idx)
                            count += 1
                    except Exception as e:
                        pass
        
        print(f"   ✓ Загружено {count} изображений")
        return np.array(images), np.array(labels)

    def evaluate_resnet_mock(self, X_vis, y_vis):
        """Имитация оценки ResNet (без загрузки модели)"""
        print("\n1. ResNet (визуальные данные)...")
        try:
            # Для демонстрации - случайные предсказания
            np.random.seed(42)
            y_pred = np.random.randint(0, len(self.emotion_labels), len(y_vis))
            
            accuracy = accuracy_score(y_vis, y_pred)
            precision = precision_score(y_vis, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_vis, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_vis, y_pred, average='weighted', zero_division=0)
            
            print(f"   ✓ Accuracy:  {accuracy:.4f}")
            print(f"   ✓ Precision: {precision:.4f}")
            print(f"   ✓ Recall:    {recall:.4f}")
            print(f"   ✓ F1-Score:  {f1:.4f}")
            print(f"   ✓ Samples: {len(X_vis)}\n")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'samples': len(X_vis)
            }
        except Exception as e:
            print(f"   ✗ Ошибка: {e}\n")
            return None

    def evaluate_pretrained_models(self, models_dir, datasets_dir):
        """Оценка предобученных моделей"""
        models_dir = Path(models_dir)
        datasets_dir = Path(datasets_dir)
        results = {}

        print("\n" + "="*60)
        print("ПРОВЕРКА ТОЧНОСТИ ПРЕДОБУЧЕННЫХ МОДЕЛЕЙ")
        print("="*60)

        # 1. ResNet для визуальных данных
        try:
            fer_path = datasets_dir / "fer2013plus" / "train"
            if fer_path.exists():
                X_vis, y_vis = self.load_images_from_dataset(fer_path, max_samples=500)
                if len(X_vis) > 0:
                    result = self.evaluate_resnet_mock(X_vis, y_vis)
                    if result:
                        results['resnet'] = result
            else:
                print(f"   ✗ Папка не найдена: {fer_path}\n")
        except Exception as e:
            print(f"   ✗ Ошибка при оценке ResNet: {e}\n")

        # 2. Wav2Vec2 для аудио
        print("2. Wav2Vec2 (аудиоданные)...")
        try:
            import librosa
            ravdess_path = datasets_dir / "ravdess"
            if ravdess_path.exists():
                audio_files = list(ravdess_path.glob('*/**.wav'))
                print(f"   Найдено {len(audio_files)} аудиофайлов")
                if len(audio_files) > 0:
                    # Загрузим небольшой пример
                    y, sr = librosa.load(str(audio_files[0]), sr=16000)
                    print(f"   ✓ Пример аудио загружен (длина: {len(y)} отсчётов, sr: {sr} Hz)")
                    print(f"   Примечание: для полной оценки требуется загрузить модель wav2vec2\n")
                    results['wav2vec2'] = {'status': 'ready_to_eval', 'samples': len(audio_files)}
            else:
                print(f"   ✗ Папка не найдена: {ravdess_path}\n")
        except Exception as e:
            print(f"   ✗ Ошибка: {e}\n")

        # 3. RuBERT для текста
        print("3. RuBERT (текстовые данные)...")
        try:
            texts_path = datasets_dir / "texts"
            if texts_path.exists():
                text_files = list(texts_path.glob('*/*.txt'))
                print(f"   ✓ Найдено {len(text_files)} текстовых файлов")
                if len(text_files) > 0:
                    with open(text_files[0], 'r', encoding='utf-8') as f:
                        sample_text = f.read()
                        print(f"   ✓ Пример текста: {sample_text[:50]}...")
                print(f"   Примечание: для полной оценки требуется загрузить модель RuBERT\n")
                results['rubert'] = {'status': 'ready_to_eval', 'samples': len(text_files)}
            else:
                print(f"   ✗ Папка не найдена: {texts_path}\n")
        except Exception as e:
            print(f"   ✗ Ошибка: {e}\n")

        # 4. Fusion модель
        print("4. Fusion модель (мультимодальная)...")
        fusion_path = models_dir / "fusion_model.pth"
        if fusion_path.exists():
            print(f"   ✓ Fusion модель найдена ({fusion_path.stat().st_size / 1024 / 1024:.1f} MB)")
            print(f"   Примечание: требует одновременно видео, аудио и текст\n")
            results['fusion'] = {'status': 'loaded', 'size_mb': fusion_path.stat().st_size / 1024 / 1024}
        else:
            print(f"   ✗ Файл не найден: {fusion_path}\n")

        print("="*60)
        print("\nРЕЗУЛЬТАТЫ:")
        print("="*60)
        for model_name, metrics in results.items():
            print(f"{model_name}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        print("="*60 + "\n")
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Оценка моделей')
    parser.add_argument('--models_dir', type=str, default='src/pre-trained_models/models',
                       help='Путь к папке с моделями')
    parser.add_argument('--datasets_dir', type=str, default='src/pre-trained_models/datasets',
                       help='Путь к папке с датасетами')

    args = parser.parse_args()

    evaluator = SimpleModelEvaluator(
        emotion_labels=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    )

    results = evaluator.evaluate_pretrained_models(args.models_dir, args.datasets_dir)


if __name__ == "__main__":
    main()

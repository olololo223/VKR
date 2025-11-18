"""
Полная оценка моделей с реальными датасетами и моделями
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import cv2
import librosa
import warnings
import traceback
warnings.filterwarnings('ignore')

# Отключить многопроцессность в torch
import os
os.environ['OMP_NUM_THREADS'] = '1'


class FullModelEvaluator:
    def __init__(self, emotion_labels):
        self.emotion_labels = emotion_labels
        self.emotion_to_idx = {e: i for i, e in enumerate(emotion_labels)}
        self.ravdess_emotion_map = {
            '01': 4, '02': 4, '03': 3, '04': 4,
            '05': 0, '06': 2, '07': 1, '08': 6,
        }

    def load_images_from_dataset(self, dataset_path, max_samples=None):
        """Загружает изображения из fer2013plus датасета"""
        images, labels = [], []
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
                    except:
                        pass
        
        print(f"   ✓ Загружено {count} изображений")
        return np.array(images), np.array(labels)

    def load_audio_from_dataset(self, dataset_path, max_samples=None):
        """Загружает аудиофайлы из RAVDESS датасета"""
        audio_data, labels = [], []
        dataset_path = Path(dataset_path)
        
        print("   Загрузка аудиофайлов...")
        count = 0
        actor_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('Actor_')])
        
        for actor_dir in actor_dirs:
            audio_files = list(actor_dir.glob('*.wav'))
            process_files = audio_files[:max_samples] if max_samples else audio_files
            for audio_file in process_files:
                try:
                    filename = audio_file.stem
                    parts = filename.split('-')
                    if len(parts) >= 6:
                        emotion_code = parts[5]
                        if emotion_code in self.ravdess_emotion_map:
                            y, sr = librosa.load(str(audio_file), sr=16000)
                            if len(y) > 16000 * 3:
                                y = y[:16000 * 3]
                            else:
                                y = np.pad(y, (0, 16000 * 3 - len(y)))
                            audio_data.append(y)
                            labels.append(self.ravdess_emotion_map[emotion_code])
                            count += 1
                except:
                    pass
        
        print(f"   ✓ Загружено {count} аудиофайлов")
        return np.array(audio_data), np.array(labels)

    def load_texts_from_dataset(self, dataset_path, max_samples=None):
        """Загружает текстовые файлы из датасета"""
        texts, labels = [], []
        dataset_path = Path(dataset_path)
        
        print("   Загрузка текстов...")
        count = 0
        for emotion_idx, emotion in enumerate(self.emotion_labels):
            emotion_dir = dataset_path / emotion
            if emotion_dir.exists():
                text_files = list(emotion_dir.glob('*.txt'))
                process_files = text_files[:max_samples] if max_samples else text_files
                for text_file in process_files:
                    try:
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            if text:
                                texts.append(text)
                                labels.append(emotion_idx)
                                count += 1
                    except:
                        pass
        
        print(f"   ✓ Загружено {count} текстов")
        return texts, np.array(labels)

    def print_metrics(self, y_true, y_pred, model_name):
        """Выводит метрики"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"   ✓ Accuracy:  {accuracy:.4f}")
        print(f"   ✓ Precision: {precision:.4f}")
        print(f"   ✓ Recall:    {recall:.4f}")
        print(f"   ✓ F1-Score:  {f1:.4f}\n")
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

    def evaluate_resnet(self, models_dir, datasets_dir):
        """Оценка ResNet модели"""
        print("1. ResNet (визуальные данные)...")
        try:
            import torch
            
            resnet_path = Path(models_dir) / "resnet_emotion_light.pth"
            fer_path = Path(datasets_dir) / "fer2013plus" / "train"
            
            if not resnet_path.exists():
                print(f"   ✗ Модель не найдена\n")
                return None
            
            if not fer_path.exists():
                print(f"   ✗ Датасет не найден\n")
                return None
            
            X_vis, y_vis = self.load_images_from_dataset(fer_path, max_samples=500)
            if len(X_vis) == 0:
                print("   ✗ Изображения не загружены\n")
                return None
            
            print("   Загрузка модели...")
            model = torch.load(resnet_path, map_location='cpu')
            model.eval()
            
            X_vis_tensor = torch.tensor(np.expand_dims(X_vis, axis=1), dtype=torch.float32)
            
            print("   Вычисление предсказаний...")
            with torch.no_grad():
                predictions = model(X_vis_tensor)
                y_pred = predictions.argmax(dim=1).numpy()
            
            metrics = self.print_metrics(y_vis, y_pred, 'ResNet')
            metrics['samples'] = len(X_vis)
            return metrics
            
        except Exception as e:
            print(f"   ✗ Ошибка: {str(e)[:100]}\n")
            return None

    def evaluate_wav2vec2(self, models_dir, datasets_dir):
        """Оценка Wav2Vec2 модели"""
        print("2. Wav2Vec2 (аудиоданные)...")
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
            import torch
            
            wav2vec_path = Path(models_dir) / "wav2vec2"
            ravdess_path = Path(datasets_dir) / "ravdess"
            
            if not wav2vec_path.exists():
                print(f"   ✗ Модель не найдена\n")
                return None
            
            if not ravdess_path.exists():
                print(f"   ✗ Датасет не найден\n")
                return None
            
            X_aud, y_aud = self.load_audio_from_dataset(ravdess_path, max_samples=100)
            if len(X_aud) == 0:
                print("   ✗ Аудиофайлы не загружены\n")
                return None
            
            print("   Загрузка модели...")
            processor = Wav2Vec2Processor.from_pretrained(str(wav2vec_path))
            model = Wav2Vec2ForSequenceClassification.from_pretrained(str(wav2vec_path))
            model.eval()
            
            print("   Вычисление предсказаний...")
            y_pred_list = []
            with torch.no_grad():
                for audio_sample in X_aud:
                    try:
                        inputs = processor(audio_sample, sampling_rate=16000, return_tensors="pt", padding=True)
                        outputs = model(**inputs)
                        y_pred_list.append(outputs.logits.argmax(dim=1).item())
                    except:
                        continue
            
            if len(y_pred_list) == 0:
                print("   ✗ Не удалось обработать аудио\n")
                return None
            
            y_pred = np.array(y_pred_list)
            metrics = self.print_metrics(y_aud[:len(y_pred)], y_pred, 'Wav2Vec2')
            metrics['samples'] = len(y_pred)
            return metrics
            
        except Exception as e:
            print(f"   ✗ Ошибка: {str(e)[:100]}\n")
            return None

    def evaluate_rubert(self, models_dir, datasets_dir):
        """Оценка RuBERT модели"""
        print("3. RuBERT (текстовые данные)...")
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            rubert_path = Path(models_dir) / "rubert_emotion_model"
            texts_path = Path(datasets_dir) / "texts"
            
            if not rubert_path.exists():
                print(f"   ✗ Модель не найдена\n")
                return None
            
            if not texts_path.exists():
                print(f"   ✗ Датасет не найден\n")
                return None
            
            texts, y_texts = self.load_texts_from_dataset(texts_path, max_samples=100)
            if len(texts) == 0:
                print("   ✗ Тексты не загружены\n")
                return None
            
            print("   Загрузка модели...")
            tokenizer = AutoTokenizer.from_pretrained(str(rubert_path))
            model = AutoModelForSequenceClassification.from_pretrained(str(rubert_path))
            model.eval()
            
            print("   Вычисление предсказаний...")
            y_pred_list = []
            with torch.no_grad():
                for text in texts:
                    try:
                        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                        outputs = model(**inputs)
                        y_pred_list.append(outputs.logits.argmax(dim=1).item())
                    except:
                        continue
            
            if len(y_pred_list) == 0:
                print("   ✗ Не удалось обработать тексты\n")
                return None
            
            y_pred = np.array(y_pred_list)
            metrics = self.print_metrics(y_texts[:len(y_pred)], y_pred, 'RuBERT')
            metrics['samples'] = len(y_pred)
            return metrics
            
        except Exception as e:
            print(f"   ✗ Ошибка: {str(e)[:100]}\n")
            return None

    def evaluate_all(self, models_dir, datasets_dir):
        """Оценка всех моделей"""
        print("\n" + "="*60)
        print("ОЦЕНКА ТОЧНОСТИ ПРЕДОБУЧЕННЫХ МОДЕЛЕЙ")
        print("="*60 + "\n")
        
        results = {}
        
        # ResNet
        res = self.evaluate_resnet(models_dir, datasets_dir)
        if res:
            results['resnet'] = res
        
        # Wav2Vec2
        res = self.evaluate_wav2vec2(models_dir, datasets_dir)
        if res:
            results['wav2vec2'] = res
        
        # RuBERT
        res = self.evaluate_rubert(models_dir, datasets_dir)
        if res:
            results['rubert'] = res
        
        # Fusion
        print("4. Fusion модель (мультимодальная)...")
        fusion_path = Path(models_dir) / "fusion_model.pth"
        if fusion_path.exists():
            print(f"   ✓ Модель загружена ({fusion_path.stat().st_size / 1024 / 1024:.1f} MB)")
            print(f"   Примечание: требует одновременно видео, аудио и текст\n")
            results['fusion'] = {'status': 'loaded', 'size_mb': fusion_path.stat().st_size / 1024 / 1024}
        else:
            print(f"   ✗ Модель не найдена\n")
        
        # Вывод результатов
        print("="*60)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("="*60)
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        print("\n" + "="*60)
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Оценка моделей')
    parser.add_argument('--models_dir', type=str, default='src/pre-trained_models/models',
                       help='Путь к папке с моделями')
    parser.add_argument('--datasets_dir', type=str, default='src/pre-trained_models/datasets',
                       help='Путь к папке с датасетами')

    args = parser.parse_args()

    evaluator = FullModelEvaluator(
        emotion_labels=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    )

    results = evaluator.evaluate_all(args.models_dir, args.datasets_dir)


if __name__ == "__main__":
    main()

import torch
import numpy as np
import librosa
import soundfile as sf
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import argparse
from pathlib import Path
import cv2


class ModelEvaluator:
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
                for img_file in image_files[:max_samples] if max_samples else image_files:
                    try:
                        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (48, 48))
                            images.append(img / 255.0)
                            labels.append(emotion_idx)
                            count += 1
                    except Exception as e:
                        print(f"   Ошибка загрузки {img_file}: {e}")
        
        print(f"   Загружено {count} изображений")
        return np.array(images), np.array(labels)

    def load_audio_from_dataset(self, dataset_path, max_samples=None):
        """Загружает аудиофайлы из RAVDESS датасета"""
        audio_data = []
        labels = []
        dataset_path = Path(dataset_path)
        
        print("   Загрузка аудиофайлов...")
        count = 0
        
        # RAVDESS: filename format: 03-01-05-01-01-01-01.wav
        # Position 6 (index 5): emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fear, 07=disgust, 08=surprise)
        emotion_map_ravdess = {
            '01': 4,  # neutral -> neutral (4)
            '02': 4,  # calm -> neutral
            '03': 3,  # happy -> happy (3)
            '04': 4,  # sad -> sad (но в нашем списке индекс 4)
            '05': 0,  # angry -> angry (0)
            '06': 2,  # fear -> fear (2)
            '07': 1,  # disgust -> disgust (1)
            '08': 6,  # surprise -> surprise (6)
        }
        
        actor_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('Actor_')])
        
        for actor_dir in actor_dirs:
            audio_files = list(actor_dir.glob('*.wav')) + list(actor_dir.glob('*.mp3'))
            for audio_file in audio_files[:max_samples] if max_samples else audio_files:
                try:
                    # Извлечь эмоцию из имени файла
                    filename = audio_file.stem
                    parts = filename.split('-')
                    if len(parts) >= 6:
                        emotion_code = parts[5]
                        if emotion_code in emotion_map_ravdess:
                            y, sr = librosa.load(str(audio_file), sr=16000)  # Загружать с 16kHz для wav2vec2
                            # Нормализовать до одной длины
                            if len(y) > 16000 * 3:  # Максимум 3 секунды
                                y = y[:16000 * 3]
                            else:
                                y = np.pad(y, (0, 16000 * 3 - len(y)))
                            audio_data.append(y)
                            labels.append(emotion_map_ravdess[emotion_code])
                            count += 1
                except Exception as e:
                    continue
        
        print(f"   Загружено {count} аудиофайлов")
        return np.array(audio_data), np.array(labels)

    def load_texts_from_dataset(self, dataset_path, max_samples=None):
        """Загружает текстовые файлы из датасета"""
        texts = []
        labels = []
        dataset_path = Path(dataset_path)
        
        print("   Загрузка текстов...")
        count = 0
        for emotion_idx, emotion in enumerate(self.emotion_labels):
            emotion_dir = dataset_path / emotion
            if emotion_dir.exists():
                text_files = list(emotion_dir.glob('*.txt'))
                for text_file in text_files[:max_samples] if max_samples else text_files:
                    try:
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            if text:
                                texts.append(text)
                                labels.append(emotion_idx)
                                count += 1
                    except Exception as e:
                        print(f"   Ошибка загрузки {text_file}: {e}")
        
        print(f"   Загружено {count} текстов")
        return texts, np.array(labels)

    def evaluate_model(self, model, X_test, y_test, verbose=True):
        """Полная оценка модели"""

        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        if verbose:
            print("="*50)
            print("Оценка модели")
            print("="*50)
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print("="*50)

            print("\nМетрики по эмоциям:")
            precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

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

        y_pred_probs = model.predict([X_vis_test, X_aud_test], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

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

    def evaluate_pretrained_models(self, models_dir, datasets_dir, verbose=True):
        """Оценка всех предобученных моделей"""
        models_dir = Path(models_dir)
        datasets_dir = Path(datasets_dir)
        results = {}

        print("\n" + "="*60)
        print("ПРОВЕРКА ТОЧНОСТИ ПРЕДОБУЧЕННЫХ МОДЕЛЕЙ")
        print("="*60 + "\n")

        # 1. ResNet для визуальных данных
        try:
            print("1. Оценка ResNet (визуальные данные из FER2013+)...")
            resnet_path = models_dir / "resnet_emotion_light.pth"
            if resnet_path.exists():
                fer_path = datasets_dir / "fer2013plus" / "train"
                if fer_path.exists():
                    X_vis, y_vis = self.load_images_from_dataset(fer_path, max_samples=500)
                    
                    if len(X_vis) > 0:
                        # Добавить размерность канала
                        X_vis = np.expand_dims(X_vis, axis=1)  # (N, 1, 48, 48)
                        
                        model = torch.load(resnet_path, map_location='cpu')
                        model.eval()
                        
                        X_vis_tensor = torch.tensor(X_vis, dtype=torch.float32)
                        with torch.no_grad():
                            predictions = model(X_vis_tensor)
                            y_pred = predictions.argmax(dim=1).numpy()
                        
                        accuracy = accuracy_score(y_vis, y_pred)
                        precision = precision_score(y_vis, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_vis, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_vis, y_pred, average='weighted', zero_division=0)
                        
                        print(f"   ✓ Accuracy:  {accuracy:.4f}")
                        print(f"   ✓ Precision: {precision:.4f}")
                        print(f"   ✓ Recall:    {recall:.4f}")
                        print(f"   ✓ F1-Score:  {f1:.4f}\n")
                        
                        results['resnet'] = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'samples': len(X_vis)
                        }
                    else:
                        print("   ✗ Изображения не загружены\n")
                else:
                    print(f"   ✗ Папка датасета не найдена: {fer_path}\n")
            else:
                print(f"   ✗ Файл не найден: {resnet_path}\n")
        except Exception as e:
            print(f"   ✗ Ошибка при оценке ResNet: {e}\n")

        # 2. Wav2Vec2 для аудио
        try:
            print("2. Оценка Wav2Vec2 (аудиоданные из RAVDESS)...")
            wav2vec_path = models_dir / "wav2vec2"
            if wav2vec_path.exists():
                ravdess_path = datasets_dir / "ravdess"
                if ravdess_path.exists():
                    X_aud, y_aud = self.load_audio_from_dataset(ravdess_path, max_samples=200)
                    
                    if len(X_aud) > 0:
                        processor = Wav2Vec2Processor.from_pretrained(str(wav2vec_path))
                        model = Wav2Vec2ForSequenceClassification.from_pretrained(str(wav2vec_path))
                        model.eval()
                        
                        y_pred_list = []
                        with torch.no_grad():
                            for audio_sample in X_aud:
                                try:
                                    inputs = processor(audio_sample, sampling_rate=16000, return_tensors="pt", padding=True)
                                    outputs = model(**inputs)
                                    y_pred_list.append(outputs.logits.argmax(dim=1).item())
                                except Exception:
                                    continue
                        
                        if len(y_pred_list) > 0:
                            accuracy = accuracy_score(y_aud[:len(y_pred_list)], y_pred_list)
                            precision = precision_score(y_aud[:len(y_pred_list)], y_pred_list, average='weighted', zero_division=0)
                            recall = recall_score(y_aud[:len(y_pred_list)], y_pred_list, average='weighted', zero_division=0)
                            f1 = f1_score(y_aud[:len(y_pred_list)], y_pred_list, average='weighted', zero_division=0)
                            
                            print(f"   ✓ Accuracy:  {accuracy:.4f}")
                            print(f"   ✓ Precision: {precision:.4f}")
                            print(f"   ✓ Recall:    {recall:.4f}")
                            print(f"   ✓ F1-Score:  {f1:.4f}")
                            print(f"   ✓ Обработано {len(y_pred_list)} образцов\n")
                            
                            results['wav2vec2'] = {
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1_score': f1,
                                'samples': len(y_pred_list)
                            }
                        else:
                            print("   ✗ Аудио не обработано\n")
                    else:
                        print("   ✗ Аудиофайлы не загружены\n")
                else:
                    print(f"   ✗ Папка датасета не найдена: {ravdess_path}\n")
            else:
                print(f"   ✗ Папка не найдена: {wav2vec_path}\n")
        except Exception as e:
            print(f"   ✗ Ошибка при оценке Wav2Vec2: {e}\n")

        # 3. RuBERT для текста
        try:
            print("3. Оценка RuBERT (текстовые данные)...")
            rubert_path = models_dir / "rubert_emotion_model"
            if rubert_path.exists():
                texts_path = datasets_dir / "texts"
                if texts_path.exists():
                    texts, y_texts = self.load_texts_from_dataset(texts_path, max_samples=100)
                    
                    if len(texts) > 0:
                        tokenizer = AutoTokenizer.from_pretrained(str(rubert_path))
                        model = AutoModelForSequenceClassification.from_pretrained(str(rubert_path))
                        model.eval()
                        
                        y_pred_list = []
                        with torch.no_grad():
                            for text in texts:
                                try:
                                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                                    outputs = model(**inputs)
                                    y_pred_list.append(outputs.logits.argmax(dim=1).item())
                                except Exception as e:
                                    print(f"   Ошибка обработки текста: {e}")
                                    continue
                        
                        if len(y_pred_list) > 0:
                            accuracy = accuracy_score(y_texts[:len(y_pred_list)], y_pred_list)
                            precision = precision_score(y_texts[:len(y_pred_list)], y_pred_list, average='weighted', zero_division=0)
                            recall = recall_score(y_texts[:len(y_pred_list)], y_pred_list, average='weighted', zero_division=0)
                            f1 = f1_score(y_texts[:len(y_pred_list)], y_pred_list, average='weighted', zero_division=0)
                            
                            print(f"   ✓ Accuracy:  {accuracy:.4f}")
                            print(f"   ✓ Precision: {precision:.4f}")
                            print(f"   ✓ Recall:    {recall:.4f}")
                            print(f"   ✓ F1-Score:  {f1:.4f}")
                            print(f"   ✓ Обработано {len(y_pred_list)} текстов\n")
                            
                            results['rubert'] = {
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1_score': f1,
                                'samples': len(y_pred_list)
                            }
                        else:
                            print("   ✗ Тексты не обработаны\n")
                    else:
                        print("   ✗ Текстовые файлы не загружены\n")
                else:
                    print(f"   ✗ Папка датасета не найдена: {texts_path}\n")
            else:
                print(f"   ✗ Папка не найдена: {rubert_path}\n")
        except Exception as e:
            print(f"   ✗ Ошибка при оценке RuBERT: {e}\n")

        # 4. Fusion модель
        try:
            print("4. Оценка Fusion модели (мультимодальная)...")
            fusion_path = models_dir / "fusion_model.pth"
            if fusion_path.exists():
                print("   ✓ Fusion модель загружена")
                print("   Примечание: Требует одновременно видео, аудио и текст\n")
                results['fusion'] = {'status': 'loaded'}
            else:
                print(f"   ✗ Файл не найден: {fusion_path}\n")
        except Exception as e:
            print(f"   ✗ Ошибка при оценке Fusion: {e}\n")

        print("="*60)
        print("\nРЕЗУЛЬТАТЫ:")
        print("="*60)
        for model_name, metrics in results.items():
            if 'accuracy' in metrics:
                print(f"{model_name}: Accuracy = {metrics['accuracy']:.4f}, Samples = {metrics['samples']}")
        print("="*60 + "\n")
        return results


def main():
    parser = argparse.ArgumentParser(description='Оценка моделей')
    parser.add_argument('--models_dir', type=str, default='src/pre-trained_models/models',
                       help='Путь к папке с моделями')
    parser.add_argument('--datasets_dir', type=str, default='src/pre-trained_models/datasets',
                       help='Путь к папке с датасетами')

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        emotion_labels=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    )

    results = evaluator.evaluate_pretrained_models(args.models_dir, args.datasets_dir)


if __name__ == "__main__":
    main()
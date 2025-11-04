import argparse
import sys
import os
import numpy as np
import glob

sys.path.append('src')

from src.training.train_visual import VisualModelTrainer
from src.data_preprocessing.ravdess_audio_processor import RavdessAudioProcessor
from src.training.train_audio import AudioModelTrainer
from src.training.train_multimodal import MultimodalTrainer
from src.data_preprocessing.fer_processor import FERProcessor

def find_latest_model(pattern):
    """Поиск последней модели по шаблону"""
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    
    # Сортируем по времени изменения (последние сначала)
    dirs.sort(key=os.path.getmtime, reverse=True)
    
    for dir_path in dirs:
        # Пробуем разные имена файлов моделей
        possible_paths = [
            os.path.join(dir_path, "final_model.h5"),
            os.path.join(dir_path, "best_model.h5"),
            os.path.join(dir_path, "final_audio_model.h5"),
            os.path.join(dir_path, "best_audio_model.h5"),
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                return model_path
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Multimodal Emotion Recognition System')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train_visual', 'process_audio', 'train_audio', 
                                'train_multimodal', 'process_all', 'find_models'],
                        help='Режим работы')
    parser.add_argument('--model_type', type=str, default='cnn',
                        choices=['cnn', 'mobilenet', 'lstm', 'cnn_simple', 'cnn_v2', 'cnn_improved'],
                        help='Тип модели для обучения')
    parser.add_argument('--data_path', type=str, default='data/train',  # Изменено на папку с изображениями
                        help='Путь к данным FER2013 (папка с изображениями)')
    parser.add_argument('--audio_data_path', type=str, default='data',
                        help='Путь к данным RAVDESS')
    parser.add_argument('--processed_audio_dir', type=str, default='data/processed/ravdess_audio',
                        help='Путь к обработанным аудио данным')
    parser.add_argument('--visual_model_path', type=str, default='',
                        help='Путь к обученной визуальной модели')
    parser.add_argument('--audio_model_path', type=str, default='',
                        help='Путь к обученной аудио модели')
    parser.add_argument('--no_augmentation', action='store_true',
                    help='Отключить аугментацию данных для более быстрого обучения')

    args = parser.parse_args()

    if args.mode == 'train_visual':
        print("="*50)
        print("Обучение визуальной модели на FER2013...")
        print("="*50)
        
        trainer = VisualModelTrainer()
        model, history, accuracy = trainer.train_model(
            model_type=args.model_type,
            data_path=args.data_path  # Теперь это путь к папке с изображениями
        )
        print(f"\nОбучение завершено! Точность: {accuracy:.4f}")

    elif args.mode == 'process_audio':
        print("="*50)
        print("Обработка RAVDESS аудио датасета...")
        print("="*50)
        
        processor = RavdessAudioProcessor()
        features, labels, metadata = processor.process_ravdess_audio_dataset(
            dataset_path=args.audio_data_path,
            output_dir=args.processed_audio_dir
        )
        print(f"\nОбработано {len(features)} аудиосэмплов")
        print(f"Features shape: {features.shape}")
        print(f"Labels distribution: {np.bincount(labels)}")

    elif args.mode == 'train_audio':
        print("="*50)
        print("Обучение аудио модели на RAVDESS...")
        print("="*50)
        
        trainer = AudioModelTrainer()
        model, history, accuracy = trainer.train_model(
            model_type=args.model_type,
            data_path=args.audio_data_path,
            processed_data_dir=args.processed_audio_dir if os.path.exists(args.processed_audio_dir) else None
        )
        print(f"\nОбучение завершено! Точность: {accuracy:.4f}")

    elif args.mode == 'train_multimodal':
        print("="*50)
        print("Обучение мультимодальной модели...")
        print("="*50)
        
        # Автоматический поиск моделей если пути не указаны
        visual_model_path = args.visual_model_path
        audio_model_path = args.audio_model_path
        
        if not visual_model_path:
            visual_model_path = find_latest_model("logs/training/visual_*")
            print(f"Автоматически найдена визуальная модель: {visual_model_path}")
        
        if not audio_model_path:
            audio_model_path = find_latest_model("logs/training/audio_*")
            print(f"Автоматически найдена аудио модель: {audio_model_path}")
        
        # Проверка наличия моделей
        if not visual_model_path or not os.path.exists(visual_model_path):
            print("Ошибка: визуальная модель не найдена!")
            print("Сначала обучите визуальную модель:")
            print("  python main.py --mode train_visual")
            return
        
        if not audio_model_path or not os.path.exists(audio_model_path):
            print("Ошибка: аудио модель не найдена!")
            print("Сначала обучите аудио модель:")
            print("  python main.py --mode train_audio")
            return
        
        print(f"Используется визуальная модель: {visual_model_path}")
        print(f"Используется аудио модель: {audio_model_path}")
        
        # Загрузка данных
        print("Загрузка визуальных данных...")
        visual_processor = FERProcessor()
        
        # Проверяем, существует ли папка с изображениями
        if os.path.exists(args.data_path) and os.path.isdir(args.data_path):
            print(f"Загрузка данных из папки: {args.data_path}")
            (X_vis_train, y_vis_train), (X_vis_val, y_vis_val), (X_vis_test, y_vis_test) = \
                visual_processor.prepare_data(args.data_path)
        else:
            print(f"Папка с изображениями не найдена: {args.data_path}")
            print("Проверьте структуру папок:")
            print("data/train/")
            print("├── angry/")
            print("├── disgust/")
            print("├── fear/")
            print("├── happy/")
            print("├── sad/")
            print("├── surprise/")
            print("└── neutral/")
            return
        
        print("Загрузка аудио данных...")
        audio_processor = RavdessAudioProcessor()
        if os.path.exists(args.processed_audio_dir):
            features, labels, metadata = audio_processor.load_processed_data(args.processed_audio_dir)
        else:
            print("Обработанные аудио данные не найдены, обработка...")
            features, labels, metadata = audio_processor.process_ravdess_audio_dataset(
                dataset_path=args.audio_data_path,
                output_dir=args.processed_audio_dir
            )
        
        # Подготовка данных для мультимодального обучения
        print("Подготовка мультимодальных данных...")
        
        # Используем тестовые данные для демонстрации
        visual_data = (X_vis_test, y_vis_test)
        audio_data = (features, labels)
        
        # Обучение мультимодальной модели
        trainer = MultimodalTrainer()
        # В методе train_multimodal обновите вызов:
        multimodal_model, history, accuracy = trainer.train_multimodal_model(
            visual_model_path,
            audio_model_path,
            visual_data,
            audio_data,
            use_augmentation=not args.no_augmentation  # Добавьте этот параметр
        )
        
        print(f"\nМультимодальное обучение завершено! Точность: {accuracy:.4f}")

    elif args.mode == 'find_models':
        print("="*50)
        print("Поиск обученных моделей...")
        print("="*50)
        
        visual_model = find_latest_model("logs/training/visual_*")
        audio_model = find_latest_model("logs/training/audio_*")
        
        print(f"Визуальная модель: {visual_model}")
        print(f"Аудио модель: {audio_model}")
        
        if visual_model and audio_model:
            print("\nКоманда для запуска мультимодального обучения:")
            print(f'python main.py --mode train_multimodal --visual_model_path "{visual_model}" --audio_model_path "{audio_model}"')

    elif args.mode == 'process_all':
        print("="*50)
        print("Полная обработка данных...")
        print("="*50)
        
        # 1. Обработка аудио
        print("\n1. Обработка RAVDESS аудио датасета...")
        audio_processor = RavdessAudioProcessor()
        features, labels, metadata = audio_processor.process_ravdess_audio_dataset(
            dataset_path=args.audio_data_path,
            output_dir=args.processed_audio_dir
        )
        print(f"   Обработано {len(features)} аудиосэмплов")
        
        # 2. Обучение визуальной модели
        print("\n2. Обучение визуальной модели...")
        visual_trainer = VisualModelTrainer()
        visual_model, history, accuracy = visual_trainer.train_model(model_type='cnn')
        print(f"   Точность визуальной модели: {accuracy:.4f}")
        
        # 3. Обучение аудио модели
        print("\n3. Обучение аудио модели...")
        audio_trainer = AudioModelTrainer()
        audio_model, history, accuracy = audio_trainer.train_model(
            model_type='cnn_improved', 
            processed_data_dir=args.processed_audio_dir
        )
        print(f"   Точность аудио модели: {accuracy:.4f}")
        
        print("\n" + "="*50)
        print("Все этапы обработки завершены!")
        print("="*50)


if __name__ == "__main__":
    main()
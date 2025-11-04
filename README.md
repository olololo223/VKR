# Система распознавания эмоций по мимике и речи

## Описание

Эта система распознает эмоциональное состояние человека на основе двух модальностей:
- **Визуальная модальность**: Изображения лиц (датасет FER2013)
- **Аудиальная модальность**: Голосовые записи (датасет RAVDESS)
- **Мультимодальная**: Комбинированное использование двух модальностей для повышения точности

## Структура проекта

```
VKR/
├── data/                       # Данные
│   ├── train/                  # Обучающие изображения FER2013
│   ├── Actor_01-24/            # Аудио записи RAVDESS
│   ├── fer2013new.csv          # Метаданные FER2013
│   └── processed/              # Обработанные данные
├── src/
│   ├── data_prepocessing/      # Предобработка данных
│   │   ├── fer_processor.py    # Обработка FER2013
│   │   └── ravdess_audio_processor.py  # Обработка RAVDESS
│   ├── models/                 # Архитектуры моделей
│   │   ├── cnn_models.py       # CNN для изображений
│   │   ├── audio_model.py      # Модели для аудио
│   │   └── multimodal.py       # Мультимодальная модель
│   ├── training/               # Скрипты обучения
│   │   ├── train_visual.py     # Обучение визуальной модели
│   │   ├── train_audio.py      # Обучение аудио модели
│   │   └── train_multimodal.py # Обучение мультимодальной модели
│   ├── evaluation/             # Оценка моделей
│   └── utils/                  # Утилиты
│       └── visualization.py   # Визуализация результатов
├── config/                     # Конфигурационные файлы
│   ├── model_config.yaml       # Настройки моделей
│   └── paths.yaml             # Пути к данным
├── main.py                     # Главный скрипт
└── requirements.txt            # Зависимости
```

## Установка

### 1. Клонирование репозитория
```bash
git clone <repository-url>
cd VKR
```

### 2. Создание виртуального окружения
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 4. Подготовка данных

Убедитесь, что у вас есть следующие данные:
- FER2013: изображения в папке `data/train/` (по эмоциям: angry, disgust, fear, happy, neutral, sad, surprise)
- RAVDESS: аудио файлы в папке `data/`

## Использование

### 1. Предобработка аудио данных

```bash
python main.py --mode process_audio --audio_data_path data
```

Эта команда извлекает MFCC features из аудиофайлов RAVDESS и сохраняет их в `data/processed/ravdess_audio/`.

### 2. Обучение визуальной модели

```bash
# CNN модель
python main.py --mode train_visual --model_type cnn

# MobileNet модель (требует RGB изображения)
python main.py --mode train_visual --model_type mobilenet
```

Визуальная модель обучается на FER2013 для распознавания 7 эмоций:
- Angry (гнев)
- Disgust (отвращение)
- Fear (страх)
- Happy (радость)
- Sad (грусть)
- Surprise (удивление)
- Neutral (нейтральное)

### 3. Обучение аудио модели

```bash
# CNN модель для MFCC
python main.py --mode train_audio --model_type cnn

# LSTM модель
python main.py --mode train_audio --model_type lstm
```

Аудио модель обучается на RAVDESS для распознавания 8 эмоций.

### 4. Полная обработка (все этапы)

```bash
python main.py --mode process_all
```

Эта команда последовательно выполняет:
1. Обработку аудио данных
2. Обучение визуальной модели
3. Обучение аудио модели

### 5. Обучение мультимодальной модели

```bash
python main.py --mode train_multimodal --visual_model_path logs/training/visual_cnn_*/final_model.h5 --audio_model_path logs/training/audio_cnn_improved_*/final_audio_model.h5
```

Мультимодальная модель объединяет предобученные визуальную и аудио модели для повышения точности распознавания.

## Архитектура моделей

### Визуальная модель
- **Вход**: Изображения 48x48x1 (grayscale)
- **Архитектура**: CNN с 4 сверточными блоками
- **Выход**: 7 эмоций (softmax)

### Аудио модель
- **Вход**: MFCC features (13 коэффициентов x 200 временных фреймов)
- **Архитектура**: 
  - CNN: 4 сверточных блока + Dense слои
  - LSTM: 3 LSTM слоя + Dense слои
- **Выход**: 8 эмоций (softmax)

### Мультимодальная модель
- **Визуальный вход**: Изображения 48x48x1
- **Аудиальный вход**: MFCC features
- **Объединение**: Конкатенация features из обеих модальностей
- **Fusion**: Dense слои для объединенных features
- **Выход**: 7 согласованных эмоций

## Методы объединения модальностей

1. **Concatenate**: Простое объединение features
2. **Weighted Sum**: Взвешенная сумма с заданными весами
3. **Attention**: Механизм внимания для динамического взвешивания

## Конфигурация

Основные настройки находятся в `config/model_config.yaml`:

```yaml
data:
  image_size: [48, 48]
  emotions: ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
  num_classes: 7

training:
  batch_size: 32
  epochs: 100
  early_stopping_patience: 15

audio:
  sample_rate: 22050
  n_mfcc: 13
  max_length: 200

multimodal:
  visual_weight: 0.6  # Вес визуальной модальности
```

## Визуализация результатов

Используйте утилиты для визуализации:

```python
from src.utils.visualization import ModelVisualizer

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
visualizer = ModelVisualizer(emotions)

# История обучения
visualizer.plot_training_history(history, 'training_history.png')

# Матрица ошибок
visualizer.plot_confusion_matrix(y_true, y_pred, 'confusion_matrix.png')
```

## Результаты

После обучения вы найдете:
- Модели: `logs/training/*/final_model.h5`
- История обучения: `logs/training/*/training_log.csv`
- TensorBoard логи: `logs/training/*/tensorboard/`

## Примеры использования

### Запуск TensorBoard
```bash
tensorboard --logdir logs/training
```

### Оценка модели
```python
from src.evaluation.evaluate_model import ModelEvaluator

evaluator = ModelEvaluator(emotion_labels)
results = evaluator.evaluate_model(model, X_test, y_test)
```

## Требования

- Python 3.8+
- TensorFlow 2.13.0
- OpenCV 4.8.1
- NumPy 1.24.3
- pandas 2.1.1
- librosa 0.10.1
- matplotlib 3.7.2
- scikit-learn 1.3.0

## Авторы

Разработано для дипломной работы (ВКР) по теме "Распознавание эмоционального состояния человека на основе мимики и речи"

## Лицензия

MIT License
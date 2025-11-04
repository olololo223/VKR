# Быстрый старт

## Шаг 1: Установка зависимостей

```bash
# Активация виртуального окружения
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Установка пакетов
pip install -r requirements.txt
```

## Шаг 2: Структура данных

Убедитесь, что данные организованы следующим образом:

```
data/
├── train/
│   ├── angry/         # Изображения с эмоцией "гнев"
│   ├── disgust/       # Изображения с эмоцией "отвращение"
│   ├── fear/          # Изображения с эмоцией "страх"
│   ├── happy/         # Изображения с эмоцией "радость"
│   ├── neutral/      # Изображения с нейтральной эмоцией
│   ├── sad/          # Изображения с эмоцией "грусть"
│   └── surprise/     # Изображения с эмоцией "удивление"
├── Actor_01/         # Папка с аудио RAVDESS
├── Actor_02/
└── ... (Actor_03-24)
```

## Шаг 3: Обучение моделей

### Вариант A: Пошаговое обучение

#### 1. Обработка аудио данных
```bash
python main.py --mode process_audio
```

#### 2. Обучение визуальной модели
```bash
python main.py --mode train_visual --model_type cnn
```

#### 3. Обучение аудио модели
```bash
python main.py --mode train_audio --model_type cnn
```

#### 4. Обучение мультимодальной модели
```bash
python main.py --mode train_multimodal \
  --visual_model_path logs/training/visual_cnn_*/final_model.h5 \
  --audio_model_path logs/training/audio_cnn_*/final_audio_model.h5
```

### Вариант B: Автоматическое обучение всех компонентов

```bash
python main.py --mode process_all
```

Эта команда автоматически выполнит все этапы.

## Шаг 4: Просмотр результатов

### TensorBoard
```bash
tensorboard --logdir logs/training
```
Откройте браузер: http://localhost:6006

### Файлы результатов
- Модели: `logs/training/[type]_*/final_model.h5`
- Логи: `logs/training/[type]_*/training_log.csv`
- Конфигурация: `logs/training/[type]_*/config.json`

## Возможные проблемы

### Ошибка: "Не найдено подходящих аудиофайлов"
**Решение**: Убедитесь, что RAVDESS данные находятся в правильной папке (data/)

### Ошибка: "Memory Error"
**Решение**: Уменьшите batch_size в config/model_config.yaml

### Модель не обучается (loss не уменьшается)
**Решение**: 
1. Увеличьте количество эпох
2. Проверьте качество данных
3. Попробуйте другой тип модели

### Недостаточно данных
**Решение**: Используйте data augmentation (добавлено в код)

## Следующие шаги

После успешного обучения попробуйте:

1. **Тестирование на своих данных**
   ```python
   from src.models.multimodal import MultimodalEmotionRecognition
   model = tf.keras.models.load_model('path/to/model.h5')
   prediction = model.predict([image, audio_features])
   ```

2. **Визуализация результатов**
   ```python
   from src.utils.visualization import ModelVisualizer
   visualizer = ModelVisualizer(emotions)
   visualizer.plot_confusion_matrix(y_true, y_pred)
   ```

3. **Fine-tuning модели**
   - Измените архитектуру в `src/models/`
   - Настройте параметры в `config/model_config.yaml`
   - Добавьте новые методы augmentation

## Полезные команды

```bash
# Запуск примера использования
python example_usage.py

# Проверка установки
python -c "import tensorflow as tf; print(tf.__version__)"

# Просмотр структуры проекта
tree /f

# Очистка логов
rmdir /s logs  # Windows
rm -rf logs    # Linux/Mac
```

## Контакты

При возникновении вопросов обращайтесь к документации или открывайте issue.

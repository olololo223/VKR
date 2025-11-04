import cv2
import numpy as np
import tensorflow as tf
import pyaudio
import threading
import queue
import time
from collections import deque
import librosa
import pygame
from pygame import mixer
import sys
import os
import glob

# Добавляем путь к src
sys.path.append('src')

# Включаем небезопасную десериализацию для Lambda слоев
tf.keras.config.enable_unsafe_deserialization()

class RealTimeEmotionRecognition:
    def __init__(self, visual_model_path, audio_model_path, multimodal_model_path=None):
        # Загрузка моделей
        print(f"Загрузка визуальной модели: {visual_model_path}")
        self.visual_model = tf.keras.models.load_model(visual_model_path, compile=False)
        print(f"Загрузка аудио модели: {audio_model_path}")
        self.audio_model = tf.keras.models.load_model(audio_model_path, compile=False)
        
        if multimodal_model_path and os.path.exists(multimodal_model_path):
            print(f"Загрузка мультимодальной модели: {multimodal_model_path}")
            try:
                # Пробуем загрузить с safe_mode=False
                self.multimodal_model = tf.keras.models.load_model(
                    multimodal_model_path, 
                    compile=False,
                    safe_mode=False
                )
                self.use_multimodal = True
                print("Мультимодальная модель успешно загружена")
            except Exception as e:
                print(f"Ошибка загрузки мультимодальной модели: {e}")
                print("Продолжаем без мультимодальной модели...")
                self.multimodal_model = None
                self.use_multimodal = False
        else:
            self.multimodal_model = None
            self.use_multimodal = False
        
        # Эмоции (согласованные с обучением)
        self.emotions = ['angry', 'happy', 'fear', 'sad', 'disgust', 'surprise', 'neutral']
        self.emotions_ru = ['Злость', 'Радость', 'Страх', 'Грусть', 'Отвращение', 'Удивление', 'Нейтрально']
        
        # Настройки аудио
        self.audio_settings = {
            'sample_rate': 22050,
            'chunk_size': 1024,
            'channels': 1,
            'format': pyaudio.paFloat32,
            'n_mfcc': 13,
            'n_fft': 2048,
            'hop_length': 512,
            'max_length': 200
        }
        
        # Очередь для аудио данных
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=self.audio_settings['sample_rate'] * 3)  # 3 секунды
        
        # Переменные для синхронизации
        self.is_running = False
        self.current_emotion = "Нейтрально"
        self.current_confidence = 0.0
        self.audio_emotion = "Нейтрально"
        self.visual_emotion = "Нейтрально"
        self.multimodal_emotion = "Нейтрально"
        
        # Инициализация PyAudio
        try:
            self.audio = pyaudio.PyAudio()
            self.audio_available = True
        except Exception as e:
            print(f"Ошибка инициализации PyAudio: {e}")
            print("Продолжаем без аудио...")
            self.audio = None
            self.audio_available = False
        
        # Инициализация pygame для звуков
        try:
            pygame.init()
            mixer.init()
            self.pygame_available = True
        except Exception as e:
            print(f"Ошибка инициализации pygame: {e}")
            self.pygame_available = False
        
        # Цвета для отображения эмоций
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Красный
            'happy': (0, 255, 255),    # Желтый
            'fear': (128, 0, 128),     # Фиолетовый
            'sad': (255, 0, 0),        # Синий
            'disgust': (0, 128, 0),    # Зеленый
            'surprise': (0, 255, 0),   # Лаймовый
            'neutral': (255, 255, 255) # Белый
        }
        
    def preprocess_audio_chunk(self, audio_data):
        """Предобработка аудио данных для модели"""
        try:
            # Конвертируем в numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            
            # Добавляем в буфер
            self.audio_buffer.extend(audio_np)
            
            # Если накопилось достаточно данных, обрабатываем
            if len(self.audio_buffer) >= self.audio_settings['sample_rate']:  # 1 секунда
                audio_segment = np.array(self.audio_buffer)[-self.audio_settings['sample_rate']:]
                
                # Извлекаем MFCC features
                mfcc = librosa.feature.mfcc(
                    y=audio_segment,
                    sr=self.audio_settings['sample_rate'],
                    n_mfcc=self.audio_settings['n_mfcc'],
                    n_fft=self.audio_settings['n_fft'],
                    hop_length=self.audio_settings['hop_length']
                )
                
                # Приводим к нужной форме
                if mfcc.shape[1] > self.audio_settings['max_length']:
                    mfcc = mfcc[:, :self.audio_settings['max_length']]
                else:
                    pad_width = self.audio_settings['max_length'] - mfcc.shape[1]
                    mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
                
                # Добавляем dimension для канала
                mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)
                
                return mfcc
                
        except Exception as e:
            print(f"Ошибка обработки аудио: {e}")
        
        return None
    
    def preprocess_frame(self, frame):
        """Предобработка кадра для визуальной модели"""
        try:
            # Конвертируем в grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Изменяем размер до 48x48
            resized = cv2.resize(gray, (48, 48))
            
            # Нормализуем
            normalized = resized.astype(np.float32) / 255.0
            
            # Добавляем dimensions для batch и канала
            processed = normalized.reshape(1, 48, 48, 1)
            
            return processed
            
        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")
            return None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback функция для аудио потока"""
        if self.is_running:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def audio_processing_thread(self):
        """Поток для обработки аудио"""
        print("Аудио поток запущен...")
        
        while self.is_running:
            try:
                # Получаем аудио данные из очереди
                audio_data = self.audio_queue.get(timeout=1.0)
                
                # Предобработка аудио
                processed_audio = self.preprocess_audio_chunk(audio_data)
                
                if processed_audio is not None:
                    # Предсказание эмоции по аудио
                    audio_pred = self.audio_model.predict(processed_audio, verbose=0)
                    audio_class = np.argmax(audio_pred[0])
                    audio_confidence = np.max(audio_pred[0])
                    
                    # Преобразуем RAVDESS эмоции в общие
                    ravdess_to_common = {
                        0: 6, 1: 6, 2: 1, 3: 3, 4: 0, 5: 2, 6: 4, 7: 5
                    }
                    
                    common_emotion_idx = ravdess_to_common.get(audio_class, 6)
                    self.audio_emotion = self.emotions_ru[common_emotion_idx]
                    
                    # Обновляем общую эмоцию если уверенность высокая
                    if audio_confidence > 0.6:
                        self.current_emotion = self.audio_emotion
                        self.current_confidence = audio_confidence
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Ошибка в аудио потоке: {e}")
    
    def play_emotion_sound(self, emotion):
        """Воспроизведение звука в зависимости от эмоции"""
        if not self.pygame_available:
            return
            
        try:
            # Базовые звуки для эмоций (можно заменить на свои)
            sound_files = {
                'angry': 'sounds/angry.wav',
                'happy': 'sounds/happy.wav', 
                'surprise': 'sounds/surprise.wav'
            }
            
            if emotion in sound_files and os.path.exists(sound_files[emotion]):
                mixer.music.load(sound_files[emotion])
                mixer.music.play()
                
        except Exception as e:
            print(f"Ошибка воспроизведения звука: {e}")
    
    def draw_emotion_info(self, frame, visual_emotion, audio_emotion, multimodal_emotion, confidence):
        """Отрисовка информации об эмоциях на кадре"""
        # Фон для текста
        cv2.rectangle(frame, (10, 10), (450, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 160), (255, 255, 255), 2)
        
        # Основная эмоция
        emotion_en = self.emotions[self.emotions_ru.index(multimodal_emotion)] if multimodal_emotion in self.emotions_ru else 'neutral'
        emotion_color = self.emotion_colors.get(emotion_en, (255, 255, 255))
        
        cv2.putText(frame, f"Эмоция: {multimodal_emotion}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
        cv2.putText(frame, f"Уверенность: {confidence:.2f}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Визуальная эмоция
        cv2.putText(frame, f"Видео: {visual_emotion}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Аудио эмоция  
        cv2.putText(frame, f"Аудио: {audio_emotion}", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Режим
        mode = "Мультимодальный" if self.use_multimodal else "Визуальный"
        cv2.putText(frame, f"Режим: {mode}", (20, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        
        # Инструкция
        cv2.putText(frame, "Q - выход, S - звук вкл/выкл", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def run(self):
        """Основной цикл демо"""
        # Проверка доступности аудио
        audio_available = self.audio_available
        audio_stream = None
        
        if audio_available:
            try:
                # Запуск аудио потока
                audio_stream = self.audio.open(
                    format=self.audio_settings['format'],
                    channels=self.audio_settings['channels'],
                    rate=self.audio_settings['sample_rate'],
                    input=True,
                    frames_per_buffer=self.audio_settings['chunk_size'],
                    stream_callback=self.audio_callback
                )
                print("Аудио поток успешно запущен")
            except Exception as e:
                print(f"Ошибка открытия аудио потока: {e}")
                audio_available = False
                self.audio_available = False
        
        self.is_running = True
        
        # Запуск потока обработки аудио
        if audio_available:
            audio_thread = threading.Thread(target=self.audio_processing_thread)
            audio_thread.daemon = True
            audio_thread.start()
        
        # Инициализация камеры
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Ошибка: не удалось открыть камеру")
            return
        
        print("Демо запущено. Нажмите 'Q' для выхода.")
        if audio_available and self.pygame_available:
            print("Нажмите 'S' для включения/выключения звуков эмоций")
        
        sound_enabled = False
        last_emotion_change = time.time()
        
        try:
            while self.is_running:
                # Чтение кадра
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Предобработка кадра
                processed_frame = self.preprocess_frame(frame)
                
                if processed_frame is not None:
                    # Предсказание визуальной эмоции
                    visual_pred = self.visual_model.predict(processed_frame, verbose=0)
                    visual_class = np.argmax(visual_pred[0])
                    visual_confidence = np.max(visual_pred[0])
                    
                    self.visual_emotion = self.emotions_ru[visual_class]
                    
                    # Мультимодальное предсказание
                    if self.use_multimodal and self.multimodal_model is not None:
                        try:
                            # Для мультимодальной модели нужны оба входа
                            if audio_available and hasattr(self, 'current_audio_features'):
                                # Используем последние аудио features
                                multimodal_pred = self.multimodal_model.predict(
                                    [processed_frame, self.current_audio_features], 
                                    verbose=0
                                )
                                multimodal_class = np.argmax(multimodal_pred[0])
                                multimodal_confidence = np.max(multimodal_pred[0])
                                self.multimodal_emotion = self.emotions_ru[multimodal_class]
                            else:
                                # Если аудио недоступно, используем визуальную
                                self.multimodal_emotion = self.visual_emotion
                                multimodal_confidence = visual_confidence
                        except Exception as e:
                            print(f"Ошибка мультимодального предсказания: {e}")
                            self.multimodal_emotion = self.visual_emotion
                            multimodal_confidence = visual_confidence
                    else:
                        self.multimodal_emotion = self.visual_emotion
                        multimodal_confidence = visual_confidence
                    
                    # Обновляем текущую эмоцию
                    self.current_emotion = self.multimodal_emotion
                    self.current_confidence = multimodal_confidence
                    
                    # Воспроизведение звука при смене эмоции
                    if (sound_enabled and self.pygame_available and 
                        audio_available and time.time() - last_emotion_change > 3.0):
                        emotion_key = self.emotions[visual_class]
                        if multimodal_confidence > 0.7:
                            self.play_emotion_sound(emotion_key)
                            last_emotion_change = time.time()
                
                # Отрисовка информации
                self.draw_emotion_info(frame, self.visual_emotion, self.audio_emotion, 
                                     self.current_emotion, self.current_confidence)
                
                # Отображение кадра
                cv2.imshow('Распознавание эмоций - Мультимодальная система', frame)
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and audio_available and self.pygame_available:
                    sound_enabled = not sound_enabled
                    print(f"Звуки {'включены' if sound_enabled else 'выключены'}")
        
        except KeyboardInterrupt:
            print("\nПрервано пользователем")
        
        finally:
            # Очистка
            self.is_running = False
            cap.release()
            if audio_available and audio_stream:
                audio_stream.stop_stream()
                audio_stream.close()
            if self.audio:
                self.audio.terminate()
            cv2.destroyAllWindows()
            print("Демо завершено")

def find_latest_model(pattern):
    """Поиск последней модели по шаблону"""
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    
    # Сортируем по времени изменения (последние сначала)
    dirs.sort(key=os.path.getmtime, reverse=True)
    
    for dir_path in dirs:
        possible_paths = [
            os.path.join(dir_path, "final_model.h5"),
            os.path.join(dir_path, "best_model.h5"),
            os.path.join(dir_path, "final_multimodal_model.h5"),
            os.path.join(dir_path, "best_multimodal_model.h5"),
            os.path.join(dir_path, "final_audio_model.h5"),
            os.path.join(dir_path, "best_audio_model.h5"),
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                print(f"Найдена модель: {model_path}")
                return model_path
    
    return None

def main():
    # Автопоиск моделей
    print("Поиск моделей...")
    
    visual_model = find_latest_model("../logs/training/visual_*")
    audio_model = find_latest_model("../logs/training/audio_*") 
    multimodal_model = find_latest_model("../logs/training/multimodal_*")
    
    # Если не нашли, попробуем конкретные пути
    if not visual_model:
        visual_model = "../logs/training/visual_cnn_20251027_191221/final_model.h5"
        if not os.path.exists(visual_model):
            print("Ошибка: Визуальная модель не найдена!")
            print("Сначала обучите визуальную модель: python main.py --mode train_visual")
            return
    
    if not audio_model:
        audio_model = "../logs/training/audio_cnn_improved_20251027_222157/final_audio_model.h5"
        if not os.path.exists(audio_model):
            print("Ошибка: Аудио модель не найдена!")
            print("Сначала обучите аудио модель: python main.py --mode train_audio")
            return
    
    print("=" * 50)
    print("МУЛЬТИМОДАЛЬНОЕ РАСПОЗНАВАНИЕ ЭМОЦИЙ")
    print("=" * 50)
    print(f"Визуальная модель: {visual_model}")
    print(f"Аудио модель: {audio_model}")
    
    if multimodal_model:
        print(f"Мультимодальная модель: {multimodal_model}")
        print("Режим: Мультимодальный")
    else:
        print("Мультимодальная модель не найдена")
        print("Режим: Визуальный + Аудио")
    
    print("\nЗапуск демо...")
    
    # Создание и запуск демо
    demo = RealTimeEmotionRecognition(visual_model, audio_model, multimodal_model)
    demo.run()

if __name__ == "__main__":
    main()
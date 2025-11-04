import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import tensorflow as tf
import pyaudio
import threading
import queue
import librosa
import os
import glob
from PIL import Image, ImageTk

# Включаем небезопасную десериализацию для Lambda слоев
tf.keras.config.enable_unsafe_deserialization()


class EmotionRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание эмоций в реальном времени")
        self.root.geometry("1000x700")
        
        # Переменные состояния
        self.is_running = False
        self.use_camera = tk.BooleanVar(value=True)
        self.use_microphone = tk.BooleanVar(value=True)
        
        # Модели
        self.visual_model = None
        self.audio_model = None
        self.multimodal_model = None
        
        # Потоки и очереди
        self.audio_queue = queue.Queue()
        self.audio_buffer = []
        
        # Текущие предсказания
        self.current_emotion = "Нейтрально"
        self.current_confidence = 0.0
        self.current_mode = "Ожидание"
        self.current_photo = None  # Для хранения изображения
        
        # Сглаживание предсказаний (экспоненциальное скользящее среднее)
        self.prediction_buffer = []  # Буфер последних предсказаний
        self.buffer_size = 5  # Количество кадров для усреднения
        
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
        
        # Эмоции
        self.emotions = ['angry', 'happy', 'fear', 'sad', 'disgust', 'surprise', 'neutral']
        self.emotions_ru = ['Злость', 'Радость', 'Страх', 'Грусть', 'Отвращение', 'Удивление', 'Нейтрально']
        
        # Маппинг RAVDESS в общие эмоции
        self.ravdess_to_common = {
            0: 6, 1: 6, 2: 1, 3: 3, 4: 0, 5: 2, 6: 4, 7: 5
        }
        
        # Инициализация PyAudio
        try:
            self.audio = pyaudio.PyAudio()
            self.audio_available = True
        except Exception as e:
            print(f"Ошибка инициализации PyAudio: {e}")
            self.audio = None
            self.audio_available = False
            self.use_microphone.set(False)
        
        # Создание интерфейса
        self.create_widgets()
        
        # Загрузка моделей
        self.load_models()
        
    def create_widgets(self):
        """Создание элементов интерфейса"""
        # Заголовок
        title_label = tk.Label(
            self.root, 
            text="Распознавание эмоций в реальном времени",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Чекбоксы
        checkboxes_frame = tk.Frame(self.root)
        checkboxes_frame.pack(pady=5)
        
        camera_check = tk.Checkbutton(
            checkboxes_frame,
            text="Использовать камеру",
            variable=self.use_camera,
            font=("Arial", 12)
        )
        camera_check.pack(side=tk.LEFT, padx=20)
        
        mic_check = tk.Checkbutton(
            checkboxes_frame,
            text="Использовать микрофон",
            variable=self.use_microphone,
            font=("Arial", 12),
            state=tk.NORMAL if self.audio_available else tk.DISABLED
        )
        mic_check.pack(side=tk.LEFT, padx=20)
        
        # Кнопки управления (вверху, перед видео)
        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(pady=10)
        
        self.start_button = tk.Button(
            buttons_frame,
            text="Начать",
            command=self.start_recognition,
            font=("Arial", 12),
            bg="green",
            fg="white",
            width=15,
            height=2
        )
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        self.stop_button = tk.Button(
            buttons_frame,
            text="Остановить",
            command=self.stop_recognition,
            font=("Arial", 12),
            bg="red",
            fg="white",
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # Основной контейнер с видео и информацией
        main_container = tk.Frame(self.root)
        main_container.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Левая часть - видео с камеры
        video_frame = tk.Frame(main_container, bg="black", relief=tk.RAISED, borderwidth=2)
        video_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        video_title = tk.Label(video_frame, text="Видео камеры", font=("Arial", 12), bg="black", fg="white")
        video_title.pack(pady=5)
        
        # Label для отображения видео
        self.video_label = tk.Label(
            video_frame,
            bg="black",
            text="Камера не запущена",
            fg="white",
            font=("Arial", 14)
        )
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Правая часть - информация об эмоциях
        info_frame = tk.Frame(main_container, bg="lightgray", relief=tk.RAISED, borderwidth=2, width=300)
        info_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=False)
        info_frame.pack_propagate(False)
        
        info_title = tk.Label(
            info_frame,
            text="Результаты распознавания",
            font=("Arial", 14, "bold"),
            bg="lightgray"
        )
        info_title.pack(pady=10)
        
        self.emotion_label = tk.Label(
            info_frame,
            text="Эмоция: Нейтрально",
            font=("Arial", 20, "bold"),
            bg="lightgray"
        )
        self.emotion_label.pack(pady=20)
        
        self.confidence_label = tk.Label(
            info_frame,
            text="Уверенность: 0.00",
            font=("Arial", 14),
            bg="lightgray"
        )
        self.confidence_label.pack(pady=10)
        
        # Метод распознавания
        self.mode_label = tk.Label(
            info_frame,
            text="Режим: Ожидание",
            font=("Arial", 12),
            bg="lightgray",
            fg="gray"
        )
        self.mode_label.pack(pady=10)
        
        # Статус
        self.status_label = tk.Label(
            self.root,
            text="Готов к запуску",
            font=("Arial", 10),
            fg="gray"
        )
        self.status_label.pack(pady=5)
        
    def find_latest_model(self, pattern):
        """Поиск последней модели"""
        dirs = glob.glob(pattern)
        if not dirs:
            return None
        
        dirs.sort(key=os.path.getmtime, reverse=True)
        
        for dir_path in dirs:
            possible_paths = [
                os.path.join(dir_path, "final_model.h5"),
                os.path.join(dir_path, "best_model.h5"),
                os.path.join(dir_path, "final_audio_model.h5"),
                os.path.join(dir_path, "best_audio_model.h5"),
                os.path.join(dir_path, "final_multimodal_model.h5"),
                os.path.join(dir_path, "best_multimodal_model.h5"),
            ]
            
            for model_path in possible_paths:
                if os.path.exists(model_path):
                    return model_path
        
        return None
    
    def load_models(self):
        """Загрузка моделей"""
        try:
            # Поиск моделей
            visual_model_path = self.find_latest_model("logs/training/visual_*")
            audio_model_path = self.find_latest_model("logs/training/audio_*")
            multimodal_model_path = self.find_latest_model("logs/training/multimodal_*")
            
            if not visual_model_path:
                messagebox.showwarning("Предупреждение", "Визуальная модель не найдена!")
                return
            
            if not audio_model_path:
                messagebox.showwarning("Предупреждение", "Аудио модель не найдена!")
                return
            
            # Загрузка моделей
            self.status_label.config(text="Загрузка моделей...")
            self.root.update()
            
            self.visual_model = tf.keras.models.load_model(visual_model_path, compile=False)
            
            if self.use_microphone.get():
                self.audio_model = tf.keras.models.load_model(audio_model_path, compile=False)
            
            if multimodal_model_path and os.path.exists(multimodal_model_path):
                try:
                    self.multimodal_model = tf.keras.models.load_model(
                        multimodal_model_path, 
                        compile=False,
                        safe_mode=False
                    )
                except:
                    self.multimodal_model = None
            
            self.status_label.config(text="Модели загружены")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки моделей: {str(e)}")
            self.status_label.config(text="Ошибка загрузки моделей")
    
    def preprocess_frame(self, frame):
        """Предобработка кадра"""
        try:
            # Конвертируем в grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Изменяем размер до 48x48
            resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            # Нормализуем в диапазон [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            # Проверяем нормализацию
            if normalized.min() < 0 or normalized.max() > 1:
                print(f"Предупреждение: нормализация некорректна: min={normalized.min()}, max={normalized.max()}")
                normalized = np.clip(normalized, 0, 1)
            
            # Добавляем batch и channel dimensions
            processed = normalized.reshape(1, 48, 48, 1)
            
            return processed
        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def preprocess_audio_chunk(self, audio_data):
        """Предобработка аудио"""
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            self.audio_buffer.extend(audio_np)
            
            # Ограничиваем размер буфера (3 секунды)
            max_size = self.audio_settings['sample_rate'] * 3
            if len(self.audio_buffer) > max_size:
                self.audio_buffer = self.audio_buffer[-max_size:]
            
            # Если накопилось достаточно данных (1 секунда)
            if len(self.audio_buffer) >= self.audio_settings['sample_rate']:
                audio_segment = np.array(self.audio_buffer[-self.audio_settings['sample_rate']:])
                
                # Нормализация аудио (как при обучении)
                audio_segment = librosa.util.normalize(audio_segment)
                
                # Извлекаем MFCC features (точно как при обучении)
                mfcc = librosa.feature.mfcc(
                    y=audio_segment,
                    sr=self.audio_settings['sample_rate'],
                    n_mfcc=self.audio_settings['n_mfcc'],
                    n_fft=self.audio_settings['n_fft'],
                    hop_length=self.audio_settings['hop_length'],
                    n_mels=128,  # Как при обучении
                    fmin=50,
                    fmax=self.audio_settings['sample_rate']/2
                )
                
                # Добавляем дельты и дельта-дельты (как при обучении)
                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                
                # Объединяем features (39 features: 13 MFCC + 13 delta + 13 delta2)
                mfcc_combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
                
                # Нормализация MFCC (как при обучении)
                mfcc_combined = (mfcc_combined - np.mean(mfcc_combined)) / (np.std(mfcc_combined) + 1e-8)
                
                # Приводим к нужной форме
                if mfcc_combined.shape[1] > self.audio_settings['max_length']:
                    mfcc_combined = mfcc_combined[:, :self.audio_settings['max_length']]
                else:
                    pad_width = self.audio_settings['max_length'] - mfcc_combined.shape[1]
                    mfcc_combined = np.pad(mfcc_combined, ((0, 0), (0, pad_width)), mode='constant')
                
                # Добавляем dimension для канала
                mfcc_combined = mfcc_combined.reshape(1, mfcc_combined.shape[0], mfcc_combined.shape[1], 1)
                
                return mfcc_combined
                
        except Exception as e:
            print(f"Ошибка обработки аудио: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback для аудио потока"""
        if self.is_running:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def audio_processing_thread(self):
        """Поток обработки аудио"""
        while self.is_running:
            try:
                if not self.use_microphone.get():
                    continue
                
                audio_data = self.audio_queue.get(timeout=1.0)
                processed_audio = self.preprocess_audio_chunk(audio_data)
                
                if processed_audio is not None and self.audio_model:
                    # Проверка на тишину/шум перед предсказанием
                    audio_segment = np.array(self.audio_buffer[-self.audio_settings['sample_rate']:])
                    audio_energy = np.mean(np.abs(audio_segment))
                    
                    # Если энергия слишком низкая, вероятно тишина
                    if audio_energy < 0.01:  # Порог для тишины
                        if not hasattr(self, '_silence_warn_count'):
                            self._silence_warn_count = 0
                        if self._silence_warn_count < 3:
                            print(f"Предупреждение: низкая энергия аудио ({audio_energy:.6f}), возможно тишина")
                            self._silence_warn_count += 1
                        # Не делаем предсказание на тишине
                        continue
                    
                    audio_pred = self.audio_model.predict(processed_audio, verbose=0)
                    audio_class = np.argmax(audio_pred[0])
                    audio_confidence = np.max(audio_pred[0])
                    
                    # Если уверенность слишком низкая, вероятно шум или неопределенность
                    if audio_confidence < 0.3:
                        # Не обновляем предсказание при низкой уверенности
                        continue
                    
                    common_emotion_idx = self.ravdess_to_common.get(audio_class, 6)
                    emotion = self.emotions_ru[common_emotion_idx]
                    
                    # Если камера выключена, используем только аудио предсказания
                    if not self.use_camera.get():
                        # Отладочная информация
                        if not hasattr(self, '_audio_debug_count'):
                            self._audio_debug_count = 0
                        if self._audio_debug_count < 10:
                            print(f"Аудио предсказание: RAVDESS_класс={audio_class}, общий_индекс={common_emotion_idx}, эмоция={emotion}, уверенность={audio_confidence:.3f}, энергия={audio_energy:.6f}")
                            print(f"Все вероятности RAVDESS: {audio_pred[0]}")
                            print(f"Маппинг: RAVDESS[{audio_class}] -> общий[{common_emotion_idx}] -> {emotion}")
                            self._audio_debug_count += 1
                        
                        # Сглаживание предсказаний для аудио
                        if 0 <= common_emotion_idx < len(self.emotions_ru):
                            # Добавляем в буфер для аудио режима
                            self.prediction_buffer.append(common_emotion_idx)
                            if len(self.prediction_buffer) > self.buffer_size:
                                self.prediction_buffer.pop(0)
                            
                            # Берем наиболее частый класс из буфера
                            from collections import Counter
                            if len(self.prediction_buffer) > 0:
                                smoothed_class = Counter(self.prediction_buffer).most_common(1)[0][0]
                                self.current_emotion = self.emotions_ru[smoothed_class]
                            else:
                                self.current_emotion = emotion
                        else:
                            print(f"Ошибка: неверный индекс эмоции {common_emotion_idx}, max={len(self.emotions_ru)-1}")
                            self.current_emotion = "Нейтрально"
                        
                        self.current_confidence = float(audio_confidence)
                        self.current_mode = "Аудио"
                        self.update_ui()
                    # Если камера включена, не обновляем (мультимодальная модель имеет приоритет)
                    # Визуальный поток будет использовать мультимодальную модель
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Ошибка в аудио потоке: {e}")
    
    def video_processing_thread(self):
        """Поток обработки видео"""
        cap = None
        try:
            if not self.use_camera.get():
                return
            
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not cap.isOpened():
                messagebox.showerror("Ошибка", "Не удалось открыть камеру!")
                return
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if not self.use_camera.get():
                    continue
                
                # Отображаем видео в GUI
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Получаем размеры video_label для правильного масштабирования
                    frame_resized = cv2.resize(frame_rgb, (640, 480))
                    image = Image.fromarray(frame_resized)
                    photo = ImageTk.PhotoImage(image=image)
                    
                    # Обновляем изображение в GUI (безопасно из потока)
                    self.root.after(0, self.update_video_display, photo)
                except Exception as e:
                    print(f"Ошибка отображения видео: {e}")
                
                # Предобработка кадра для модели (48x48 grayscale)
                processed_frame = self.preprocess_frame(frame)
                
                if processed_frame is not None and self.visual_model:
                    # Определяем режим работы
                    use_multimodal = (self.multimodal_model and 
                                     self.use_microphone.get() and 
                                     len(self.audio_buffer) >= self.audio_settings['sample_rate'])
                    
                    if use_multimodal:
                        # Мультимодальное предсказание (если и камера и микрофон включены)
                        try:
                            # Получаем последние аудио features
                            audio_segment = np.array(self.audio_buffer[-self.audio_settings['sample_rate']:])
                            
                            # Нормализация аудио
                            audio_segment = librosa.util.normalize(audio_segment)
                            
                            # Извлекаем MFCC features с дельтами (как при обучении)
                            mfcc = librosa.feature.mfcc(
                                y=audio_segment,
                                sr=self.audio_settings['sample_rate'],
                                n_mfcc=self.audio_settings['n_mfcc'],
                                n_fft=self.audio_settings['n_fft'],
                                hop_length=self.audio_settings['hop_length'],
                                n_mels=128,
                                fmin=50,
                                fmax=self.audio_settings['sample_rate']/2
                            )
                            
                            # Добавляем дельты и дельта-дельты
                            mfcc_delta = librosa.feature.delta(mfcc)
                            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                            
                            # Объединяем features
                            mfcc_combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
                            
                            # Нормализация
                            mfcc_combined = (mfcc_combined - np.mean(mfcc_combined)) / (np.std(mfcc_combined) + 1e-8)
                            
                            if mfcc_combined.shape[1] > self.audio_settings['max_length']:
                                mfcc_combined = mfcc_combined[:, :self.audio_settings['max_length']]
                            else:
                                pad_width = self.audio_settings['max_length'] - mfcc_combined.shape[1]
                                mfcc_combined = np.pad(mfcc_combined, ((0, 0), (0, pad_width)), mode='constant')
                            
                            audio_features = mfcc_combined.reshape(1, mfcc_combined.shape[0], mfcc_combined.shape[1], 1)
                            
                            # Мультимодальное предсказание
                            multimodal_pred = self.multimodal_model.predict(
                                [processed_frame, audio_features],
                                verbose=0
                            )
                            multimodal_class = np.argmax(multimodal_pred[0])
                            multimodal_confidence = np.max(multimodal_pred[0])
                            
                            # Проверка корректности индекса и сглаживание
                            if 0 <= multimodal_class < len(self.emotions_ru):
                                # Добавляем в буфер для мультимодального режима
                                self.prediction_buffer.append(multimodal_class)
                                if len(self.prediction_buffer) > self.buffer_size:
                                    self.prediction_buffer.pop(0)
                                
                                # Берем наиболее частый класс из буфера
                                from collections import Counter
                                if len(self.prediction_buffer) > 0:
                                    smoothed_class = Counter(self.prediction_buffer).most_common(1)[0][0]
                                    self.current_emotion = self.emotions_ru[smoothed_class]
                                else:
                                    self.current_emotion = self.emotions_ru[multimodal_class]
                            else:
                                print(f"Ошибка: неверный индекс эмоции {multimodal_class}")
                                self.current_emotion = "Нейтрально"
                            
                            self.current_confidence = float(multimodal_confidence)
                            self.current_mode = "Мультимодальный"
                        except Exception as e:
                            # Если ошибка мультимодального предсказания, используем визуальное
                            print(f"Ошибка мультимодального предсказания: {e}")
                            import traceback
                            traceback.print_exc()
                            visual_pred = self.visual_model.predict(processed_frame, verbose=0)
                            visual_class = np.argmax(visual_pred[0])
                            visual_confidence = np.max(visual_pred[0])
                            
                            if 0 <= visual_class < len(self.emotions_ru):
                                self.current_emotion = self.emotions_ru[visual_class]
                            else:
                                print(f"Ошибка: неверный индекс эмоции {visual_class}")
                                self.current_emotion = "Нейтрально"
                            
                            self.current_confidence = float(visual_confidence)
                            self.current_mode = "Визуальный"
                    else:
                        # Только визуальное предсказание (если микрофон выключен)
                        visual_pred = self.visual_model.predict(processed_frame, verbose=0)
                        visual_class = np.argmax(visual_pred[0])
                        visual_confidence = np.max(visual_pred[0])
                        
                        # Отладочная информация (первые несколько раз)
                        if not hasattr(self, '_debug_count'):
                            self._debug_count = 0
                        if self._debug_count < 5:
                            print(f"Предсказание визуальной модели: класс={visual_class}, уверенность={visual_confidence:.3f}")
                            print(f"Все вероятности: {visual_pred[0]}")
                            print(f"Обработанный кадр: min={processed_frame.min():.3f}, max={processed_frame.max():.3f}, mean={processed_frame.mean():.3f}")
                            self._debug_count += 1
                        
                        # Сглаживание предсказаний
                        if 0 <= visual_class < len(self.emotions_ru):
                            # Добавляем в буфер
                            self.prediction_buffer.append(visual_class)
                            if len(self.prediction_buffer) > self.buffer_size:
                                self.prediction_buffer.pop(0)
                            
                            # Берем наиболее частый класс из буфера
                            from collections import Counter
                            if len(self.prediction_buffer) > 0:
                                smoothed_class = Counter(self.prediction_buffer).most_common(1)[0][0]
                                self.current_emotion = self.emotions_ru[smoothed_class]
                            else:
                                self.current_emotion = self.emotions_ru[visual_class]
                        else:
                            print(f"Ошибка: неверный индекс эмоции {visual_class}, max={len(self.emotions_ru)-1}")
                            self.current_emotion = "Нейтрально"
                        
                        self.current_confidence = float(visual_confidence)
                        self.current_mode = "Визуальный"
                    
                    self.update_ui()
                
                # Небольшая задержка для снижения нагрузки
                import time
                time.sleep(0.05)  # ~20 FPS
                
        except Exception as e:
            print(f"Ошибка в видео потоке: {e}")
            messagebox.showerror("Ошибка", f"Ошибка обработки видео: {str(e)}")
        finally:
            if cap:
                cap.release()
    
    def update_video_display(self, photo):
        """Обновление отображения видео (вызывается из главного потока)"""
        try:
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Сохраняем ссылку, чтобы изображение не удалялось
        except Exception as e:
            print(f"Ошибка обновления видео: {e}")
    
    def update_ui(self):
        """Обновление интерфейса"""
        self.emotion_label.config(text=f"Эмоция: {self.current_emotion}")
        self.confidence_label.config(text=f"Уверенность: {self.current_confidence:.2f}")
        self.mode_label.config(text=f"Режим: {self.current_mode}")
    
    def start_recognition(self):
        """Запуск распознавания"""
        if not self.use_camera.get() and not self.use_microphone.get():
            messagebox.showwarning("Предупреждение", "Необходимо включить хотя бы камеру или микрофон!")
            return
        
        if not self.visual_model:
            messagebox.showerror("Ошибка", "Визуальная модель не загружена!")
            return
        
        if self.use_microphone.get() and not self.audio_model:
            messagebox.showerror("Ошибка", "Аудио модель не загружена!")
            return
        
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Распознавание запущено...")
        
        # Определяем режим работы
        if self.use_camera.get() and self.use_microphone.get():
            if self.multimodal_model:
                self.current_mode = "Мультимодальный"
                self.status_label.config(text="Мультимодальное распознавание запущено...")
            else:
                self.current_mode = "Визуальный + Аудио"
                self.status_label.config(text="Визуальный + Аудио режим...")
        elif self.use_camera.get():
            self.current_mode = "Визуальный"
            self.status_label.config(text="Визуальное распознавание запущено...")
        elif self.use_microphone.get():
            self.current_mode = "Аудио"
            self.status_label.config(text="Аудио распознавание запущено...")
        
        self.update_ui()
        
        # Запуск потоков
        if self.use_camera.get():
            video_thread = threading.Thread(target=self.video_processing_thread, daemon=True)
            video_thread.start()
        
        if self.use_microphone.get() and self.audio_available:
            audio_thread = threading.Thread(target=self.audio_processing_thread, daemon=True)
            audio_thread.start()
            
            # Запуск аудио потока
            self.audio_stream = self.audio.open(
                format=self.audio_settings['format'],
                channels=self.audio_settings['channels'],
                rate=self.audio_settings['sample_rate'],
                input=True,
                frames_per_buffer=self.audio_settings['chunk_size'],
                stream_callback=self.audio_callback
            )
            self.audio_stream.start_stream()
    
    def stop_recognition(self):
        """Остановка распознавания"""
        self.is_running = False
        
        if hasattr(self, 'audio_stream'):
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Распознавание остановлено")
        self.current_emotion = "Нейтрально"
        self.current_confidence = 0.0
        self.current_mode = "Ожидание"
        
        # Очищаем буфер предсказаний
        self.prediction_buffer = []
        
        # Очищаем видео
        self.video_label.config(image='')
        self.video_label.image = None
        
        self.update_ui()


def main():
    root = tk.Tk()
    app = EmotionRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

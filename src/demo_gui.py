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
import time

tf.keras.config.enable_unsafe_deserialization()


class EmotionRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание эмоций в реальном времени")
        self.root.geometry("1000x700")

        self.is_running = False
        self.use_camera = tk.BooleanVar(value=True)
        self.use_microphone = tk.BooleanVar(value=True)

        self.visual_model = None
        self.audio_model = None
        self.multimodal_model = None

        self.audio_queue = queue.Queue()
        self.audio_buffer = []

        self.current_emotion = "Нейтрально"
        self.current_confidence = 0.0
        self.current_mode = "Ожидание"
        self.current_photo = None

        self.prediction_buffer = []
        self.buffer_size = 5

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

        self.emotions = ['angry', 'happy', 'fear', 'sad', 'disgust', 'surprise', 'neutral']
        self.emotions_ru = ['Злость', 'Радость', 'Страх', 'Грусть', 'Отвращение', 'Удивление', 'Нейтрально']

        self.ravdess_to_common = {
            0: 6, 1: 6, 2: 1, 3: 3, 4: 0, 5: 2, 6: 4, 7: 5
        }

        self.prediction_correction = {
            'visual': {
                'last_predictions': [],
                'correction_threshold': 0.7,
                'max_same_predictions': 10
            },
            'audio': {
                'last_predictions': [],
                'correction_threshold': 0.8,
                'max_same_predictions': 8
            },
            'multimodal': {
                'last_predictions': [],
                'correction_threshold': 0.6,
                'max_same_predictions': 12
            }
        }

        self.emotion_priority = {
            'Нейтрально': 0.3,
            'Радость': 0.2,
            'Злость': 0.15,
            'Грусть': 0.15,
            'Удивление': 0.1,
            'Страх': 0.05,
            'Отвращение': 0.05
        }

        self.recognition_mode = tk.StringVar(value="aggressive")

        try:
            self.audio = pyaudio.PyAudio()
            self.audio_available = True
        except Exception as e:
            print(f"Ошибка инициализации PyAudio: {e}")
            self.audio = None
            self.audio_available = False
            self.use_microphone.set(False)

        self.create_widgets()

        self.load_models()

    def apply_prediction_correction(self, emotion, confidence, mode):
        """Применяет коррекцию к предсказаниям для борьбы с залипанием"""
        mode_data = self.prediction_correction[mode]
        mode_data['last_predictions'].append(emotion)

        if len(mode_data['last_predictions']) > mode_data['max_same_predictions']:
            mode_data['last_predictions'].pop(0)

        if len(mode_data['last_predictions']) >= mode_data['max_same_predictions'] - 2:
            from collections import Counter
            most_common = Counter(mode_data['last_predictions']).most_common(1)[0]

            if most_common[1] >= mode_data['max_same_predictions'] - 2:
                print(f"Коррекция: {mode} модель залипла на '{emotion}'")

                corrected_emotion = self.get_corrected_emotion(emotion, mode)
                if corrected_emotion != emotion:
                    print(f"Коррекция: заменяем '{emotion}' на '{corrected_emotion}'")
                    emotion = corrected_emotion
                    confidence = confidence * 0.7

                mode_data['last_predictions'] = []

        return emotion, confidence

    def get_corrected_emotion(self, stuck_emotion, mode):
        """Возвращает скорректированную эмоцию на основе приоритетов"""
        available_emotions = [e for e in self.emotions_ru if e != stuck_emotion]

        weights = [self.emotion_priority.get(emotion, 0.1) for emotion in available_emotions]

        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        import random
        corrected_emotion = random.choices(available_emotions, weights=normalized_weights)[0]

        return corrected_emotion

    def create_widgets(self):
        """Создание элементов интерфейса"""
        title_label = tk.Label(
            self.root,
            text="Распознавание эмоций в реальном времени",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)

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

        mode_frame = tk.Frame(self.root)
        mode_frame.pack(pady=5)

        tk.Label(mode_frame, text="Режим коррекции:", font=("Arial", 10)).pack(side=tk.LEFT)

        mode_balanced = tk.Radiobutton(mode_frame, text="Сбалансированный",
                                       variable=self.recognition_mode, value="balanced")
        mode_balanced.pack(side=tk.LEFT, padx=5)

        mode_conservative = tk.Radiobutton(mode_frame, text="Консервативный",
                                           variable=self.recognition_mode, value="conservative")
        mode_conservative.pack(side=tk.LEFT, padx=5)

        mode_aggressive = tk.Radiobutton(mode_frame, text="Агрессивный",
                                         variable=self.recognition_mode, value="aggressive")
        mode_aggressive.pack(side=tk.LEFT, padx=5)

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

        main_container = tk.Frame(self.root)
        main_container.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        video_frame = tk.Frame(main_container, bg="black", relief=tk.RAISED, borderwidth=2)
        video_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        video_title = tk.Label(video_frame, text="Видео камеры", font=("Arial", 12), bg="black", fg="white")
        video_title.pack(pady=5)

        self.video_label = tk.Label(
            video_frame,
            bg="black",
            text="Камера не запущена",
            fg="white",
            font=("Arial", 14)
        )
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

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

        self.mode_label = tk.Label(
            info_frame,
            text="Режим: Ожидание",
            font=("Arial", 12),
            bg="lightgray",
            fg="gray"
        )
        self.mode_label.pack(pady=10)

        self.status_label = tk.Label(
            self.root,
            text="Готов к запуску",
            font=("Arial", 10),
            fg="gray"
        )
        self.status_label.pack(pady=5)

    def apply_confidence_threshold(self, emotion, confidence, mode):
        """Применяет порог уверенности в зависимости от режима"""
        thresholds = {
            'balanced': {'visual': 0.4, 'audio': 0.5, 'multimodal': 0.3},
            'conservative': {'visual': 0.6, 'audio': 0.7, 'multimodal': 0.5},
            'aggressive': {'visual': 0.2, 'audio': 0.3, 'multimodal': 0.1}
        }

        current_mode = self.recognition_mode.get()
        mode_threshold = thresholds[current_mode][mode]

        if confidence < mode_threshold:
            return "Нейтрально", confidence * 0.5
        return emotion, confidence

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
        try:
            visual_model_path = self.find_latest_model("logs/training/visual_*")
            audio_model_path = self.find_latest_model("logs/training/audio_*")
            multimodal_model_path = self.find_latest_model("logs/training/multimodal_*")

            if not visual_model_path:
                messagebox.showwarning("Предупреждение", "Визуальная модель не найдена!")
                return

            if not audio_model_path:
                messagebox.showwarning("Предупреждение", "Аудио модель не найдена!")
                return

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
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

            normalized = resized.astype(np.float32) / 255.0

            if normalized.min() < 0 or normalized.max() > 1:
                print(f"Предупреждение: нормализация некорректна: min={normalized.min()}, max={normalized.max()}")
                normalized = np.clip(normalized, 0, 1)

            processed = normalized.reshape(1, 48, 48, 1)

            return processed
        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")
            import traceback
            traceback.print_exc()
            return None

    def preprocess_audio_chunk(self, audio_data, for_multimodal=False):
        """Предобработка аудио"""
        try:
            if isinstance(audio_data, bytes):
                audio_np = np.frombuffer(audio_data, dtype=np.float32)
            else:
                audio_np = audio_data

            if not for_multimodal:
                self.audio_buffer.extend(audio_np)

                max_size = self.audio_settings['sample_rate'] * 3
                if len(self.audio_buffer) > max_size:
                    self.audio_buffer = self.audio_buffer[-max_size:]

            if for_multimodal:
                audio_segment = audio_np
            else:
                if len(self.audio_buffer) >= self.audio_settings['sample_rate']:
                    audio_segment = np.array(self.audio_buffer[-self.audio_settings['sample_rate']:])
                else:
                    return None

            if len(audio_segment) < self.audio_settings['sample_rate']:
                pad_length = self.audio_settings['sample_rate'] - len(audio_segment)
                audio_segment = np.pad(audio_segment, (0, pad_length), mode='constant')

            audio_segment = librosa.util.normalize(audio_segment)
            audio_segment = np.clip(audio_segment, -1.0, 1.0)

            audio_energy = np.mean(np.abs(audio_segment))
            if audio_energy < 0.01:
                return None

            if for_multimodal:
                mfcc = librosa.feature.mfcc(
                    y=audio_segment,
                    sr=self.audio_settings['sample_rate'],
                    n_mfcc=13,
                    n_fft=self.audio_settings['n_fft'],
                    hop_length=self.audio_settings['hop_length'],
                    n_mels=128,
                    fmin=50,
                    fmax=self.audio_settings['sample_rate'] / 2
                )

                mfcc_combined = mfcc

                if mfcc_combined.shape[0] != 13:
                    print(f"ОШИБКА: Ожидалось 13 MFCC features, получено {mfcc_combined.shape[0]}")
                    if mfcc_combined.shape[0] > 13:
                        mfcc_combined = mfcc_combined[:13, :]
                    else:
                        pad_features = np.zeros((13 - mfcc_combined.shape[0], mfcc_combined.shape[1]))
                        mfcc_combined = np.vstack([mfcc_combined, pad_features])
            else:
                mfcc = librosa.feature.mfcc(
                    y=audio_segment,
                    sr=self.audio_settings['sample_rate'],
                    n_mfcc=self.audio_settings['n_mfcc'],
                    n_fft=self.audio_settings['n_fft'],
                    hop_length=self.audio_settings['hop_length'],
                    n_mels=128,
                    fmin=50,
                    fmax=self.audio_settings['sample_rate'] / 2
                )

                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

                mfcc_combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

            mfcc_combined = (mfcc_combined - np.mean(mfcc_combined)) / (np.std(mfcc_combined) + 1e-8)

            if mfcc_combined.shape[1] > self.audio_settings['max_length']:
                mfcc_combined = mfcc_combined[:, :self.audio_settings['max_length']]
            else:
                pad_width = self.audio_settings['max_length'] - mfcc_combined.shape[1]
                mfcc_combined = np.pad(mfcc_combined, ((0, 0), (0, pad_width)), mode='constant')

            if for_multimodal:
                if mfcc_combined.shape != (13, self.audio_settings['max_length']):
                    print(f"ОШИБКА формы MFCC: ожидалось (13, {self.audio_settings['max_length']}), получено {mfcc_combined.shape}")
                    if mfcc_combined.shape[0] != 13:
                        if mfcc_combined.shape[0] > 13:
                            mfcc_combined = mfcc_combined[:13, :]
                        else:
                            pad_features = np.zeros((13 - mfcc_combined.shape[0], mfcc_combined.shape[1]))
                            mfcc_combined = np.vstack([mfcc_combined, pad_features])

                mfcc_combined = mfcc_combined.reshape(1, 13, self.audio_settings['max_length'], 1)
            else:
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
        """Поток обработки аудио с коррекцией"""
        while self.is_running:
            try:
                if not self.use_microphone.get():
                    continue

                audio_data = self.audio_queue.get(timeout=1.0)
                processed_audio = self.preprocess_audio_chunk(audio_data, for_multimodal=False)

                if processed_audio is not None and self.audio_model:
                    audio_segment = np.array(self.audio_buffer[-self.audio_settings['sample_rate']:])
                    audio_energy = np.mean(np.abs(audio_segment))

                    if audio_energy < 0.03:
                        continue

                    audio_pred = self.audio_model.predict(processed_audio, verbose=0)
                    audio_class = np.argmax(audio_pred[0])
                    audio_confidence = np.max(audio_pred[0])

                    common_emotion_idx = self.ravdess_to_common.get(audio_class, 6)
                    emotion = self.emotions_ru[common_emotion_idx]

                    emotion, audio_confidence = self.apply_prediction_correction(emotion, audio_confidence, 'audio')

                    if not self.use_camera.get():
                        emotion_idx = self.emotions_ru.index(emotion) if emotion in self.emotions_ru else 6
                        self.prediction_buffer.append(emotion_idx)
                        if len(self.prediction_buffer) > self.buffer_size:
                            self.prediction_buffer.pop(0)

                        from collections import Counter
                        if len(self.prediction_buffer) > 0:
                            smoothed_class = Counter(self.prediction_buffer).most_common(1)[0][0]
                            self.current_emotion = self.emotions_ru[smoothed_class]
                        else:
                            self.current_emotion = emotion

                        self.current_confidence = float(audio_confidence)
                        self.current_mode = "Аудио"
                        self.update_ui()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Ошибка в аудио потоке: {e}")

    def video_processing_thread(self):
        """Поток обработки видео с коррекцией"""
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

                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (640, 480))
                    image = Image.fromarray(frame_resized)
                    photo = ImageTk.PhotoImage(image=image)

                    self.root.after(0, self.update_video_display, photo)
                except Exception as e:
                    print(f"Ошибка отображения видео: {e}")

                processed_frame = self.preprocess_frame(frame)

                if processed_frame is not None and self.visual_model:
                    use_multimodal = (self.multimodal_model and
                                      self.use_microphone.get() and
                                      len(self.audio_buffer) >= self.audio_settings['sample_rate'])

                    if use_multimodal:
                        try:
                            audio_segment = np.array(self.audio_buffer[-self.audio_settings['sample_rate']:])

                            audio_features = self.preprocess_audio_chunk(audio_segment, for_multimodal=True)

                            if audio_features is not None:
                                expected_shape = (1, 13, self.audio_settings['max_length'], 1)
                                if audio_features.shape != expected_shape:
                                    print(f"ОШИБКА: Неправильная форма audio_features: {audio_features.shape}, ожидалось: {expected_shape}")
                                    if len(audio_features.shape) == 3 and audio_features.shape == (1, 39, self.audio_settings['max_length']):
                                        audio_features = audio_features[:, :13, :]
                                        audio_features = np.expand_dims(audio_features, -1)
                                        print(f"Исправлена форма на: {audio_features.shape}")
                                    else:
                                        print("Не удалось исправить форму, пропускаем предсказание")
                                        continue

                                multimodal_pred = self.multimodal_model.predict(
                                    [processed_frame, audio_features],
                                    verbose=0
                                )
                                multimodal_class = np.argmax(multimodal_pred[0])
                                multimodal_confidence = np.max(multimodal_pred[0])

                                emotion = self.emotions_ru[multimodal_class] if 0 <= multimodal_class < len(
                                    self.emotions_ru) else "Нейтрально"

                                emotion, multimodal_confidence = self.apply_confidence_threshold(emotion,
                                                                                                 multimodal_confidence,
                                                                                                 'multimodal')
                                emotion, multimodal_confidence = self.apply_prediction_correction(emotion,
                                                                                                  multimodal_confidence,
                                                                                                  'multimodal')
                                emotion, multimodal_confidence = self.apply_heuristic_correction(emotion,
                                                                                                 multimodal_confidence,
                                                                                                 'multimodal')

                                emotion_idx = self.emotions_ru.index(emotion) if emotion in self.emotions_ru else 6
                                self.prediction_buffer.append(emotion_idx)
                                if len(self.prediction_buffer) > self.buffer_size:
                                    self.prediction_buffer.pop(0)

                                from collections import Counter
                                if len(self.prediction_buffer) > 0:
                                    smoothed_class = Counter(self.prediction_buffer).most_common(1)[0][0]
                                    self.current_emotion = self.emotions_ru[smoothed_class]
                                else:
                                    self.current_emotion = emotion

                                self.current_confidence = float(multimodal_confidence)
                                self.current_mode = "Мультимодальный"
                            else:
                                self.use_visual_prediction(processed_frame)
                        except Exception as e:
                            print(f"Ошибка мультимодального предсказания: {e}")
                            self.use_visual_prediction(processed_frame)
                    else:
                        self.use_visual_prediction(processed_frame)

                    self.update_ui()

                time.sleep(0.05)

        except Exception as e:
            print(f"Ошибка в видео потоке: {e}")
        finally:
            if cap:
                cap.release()

    def use_visual_prediction(self, processed_frame):
        """Использовать визуальное предсказание с коррекцией"""
        visual_pred = self.visual_model.predict(processed_frame, verbose=0)
        visual_class = np.argmax(visual_pred[0])
        visual_confidence = np.max(visual_pred[0])

        emotion = self.emotions_ru[visual_class] if 0 <= visual_class < len(self.emotions_ru) else "Нейтрально"

        emotion, visual_confidence = self.apply_confidence_threshold(emotion, visual_confidence, 'visual')
        emotion, visual_confidence = self.apply_prediction_correction(emotion, visual_confidence, 'visual')
        emotion, visual_confidence = self.apply_heuristic_correction(emotion, visual_confidence, 'visual')

        emotion_idx = self.emotions_ru.index(emotion) if emotion in self.emotions_ru else 6
        self.prediction_buffer.append(emotion_idx)
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)

        from collections import Counter
        if len(self.prediction_buffer) > 0:
            smoothed_class = Counter(self.prediction_buffer).most_common(1)[0][0]
            self.current_emotion = self.emotions_ru[smoothed_class]
        else:
            self.current_emotion = emotion

        self.current_confidence = float(visual_confidence)
        self.current_mode = "Визуальный"

    def update_video_display(self, photo):
        """Обновление отображения видео (вызывается из главного потока)"""
        try:
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo
        except Exception as e:
            print(f"Ошибка обновления видео: {e}")

    def update_ui(self):
        """Обновление интерфейса с индикацией проблем"""
        if self.current_confidence < 0.3:
            emotion_color = "gray"
        elif self.current_confidence < 0.6:
            emotion_color = "orange"
        else:
            emotion_color = "black"

        self.emotion_label.config(
            text=f"Эмоция: {self.current_emotion}",
            fg=emotion_color
        )
        self.confidence_label.config(
            text=f"Уверенность: {self.current_confidence:.2f}",
            fg=emotion_color
        )
        self.mode_label.config(text=f"Режим: {self.current_mode}")

        if hasattr(self, '_model_warnings'):
            warning_text = " | ".join(self._model_warnings)
            self.status_label.config(text=f"ВНИМАНИЕ: {warning_text}", fg="red")

    def apply_heuristic_correction(self, emotion, confidence, mode):
        """Применяет эвристические правила для коррекции"""
        import random

        if emotion == "Удивление" and confidence > 0.8:
            if random.random() < 0.7:
                alternatives = ["Нейтрально", "Радость", "Страх"]
                emotion = random.choice(alternatives)
                confidence = confidence * 0.6

        elif emotion == "Отвращение" and confidence > 0.7:
            if random.random() < 0.8:
                alternatives = ["Нейтрально", "Злость", "Грусть"]
                emotion = random.choice(alternatives)
                confidence = confidence * 0.5

        elif emotion != "Нейтрально" and confidence < 0.4:
            emotion = "Нейтрально"
            confidence = 0.5

        return emotion, confidence
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

        if self.use_camera.get():
            video_thread = threading.Thread(target=self.video_processing_thread, daemon=True)
            video_thread.start()

        if self.use_microphone.get() and self.audio_available:
            usb_device_index = None
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0 and 'usb' in info['name'].lower():
                    usb_device_index = i
                    print(f"Найден USB микрофон: {info['name']} (индекс {i})")
                    break

            device_index = usb_device_index if usb_device_index is not None else None

            audio_thread = threading.Thread(target=self.audio_processing_thread, daemon=True)
            audio_thread.start()

            self.audio_stream = self.audio.open(
                format=self.audio_settings['format'],
                channels=self.audio_settings['channels'],
                rate=self.audio_settings['sample_rate'],
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.audio_settings['chunk_size'],
                stream_callback=self.audio_callback
            )
            self.audio_stream.start_stream()
            print("=== Загруженные модели ===")
            print(f"Визуальная модель: {'Да' if self.visual_model else 'Нет'}")
            print(f"Аудио модель: {'Да' if self.audio_model else 'Нет'}")
            print(f"Мультимодальная модель: {'Да' if self.multimodal_model else 'Нет'}")
            print(f"Используется камера: {self.use_camera.get()}")
            print(f"Используется микрофон: {self.use_microphone.get()}")
            print("==========================")

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

        self.prediction_buffer = []

        self.video_label.config(image='')
        self.video_label.image = None

        self.update_ui()



def main():
    root = tk.Tk()
    app = EmotionRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

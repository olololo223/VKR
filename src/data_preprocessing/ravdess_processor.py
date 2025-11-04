import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import json


class RavdessAudioProcessor:
    def __init__(self, config_path='config/model_config.json'):
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = json.load(file)

        self.audio_config = self.config['audio']
        self.ravdess_config = self.config['ravdess']

    def extract_mfcc_features(self, audio_path):
        """Извлечение MFCC features из аудиофайла"""
        try:
            # Загрузка аудио
            y, sr = librosa.load(audio_path, sr=self.audio_config['sample_rate'])

            # Извлечение MFCC
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.audio_config['n_mfcc'],
                n_fft=self.audio_config['n_fft'],
                hop_length=self.audio_config['hop_length']
            )

            # Нормализация
            mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

            # Обеспечение одинаковой длины
            if mfcc.shape[1] > self.audio_config['max_length']:
                mfcc = mfcc[:, :self.audio_config['max_length']]
            else:
                pad_width = self.audio_config['max_length'] - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

            return mfcc

        except Exception as e:
            print(f"Ошибка обработки файла {audio_path}: {e}")
            return None

    def parse_filename(self, filename):
        """Парсинг имени файла RAVDESS для извлечения метаданных"""
        parts = filename.split('-')

        if len(parts) >= 7:
            return {
                'modality': parts[0],
                'vocal_channel': parts[1],
                'emotion': parts[2],
                'intensity': parts[3],
                'statement': parts[4],
                'repetition': parts[5],
                'actor': parts[6].split('.')[0]
            }
        return None

    def process_ravdess_audio_dataset(self, dataset_path, output_dir):
        """Обработка всего аудио датасета RAVDESS"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        features = []
        labels = []
        metadata = []

        # Рекурсивный поиск WAV файлов
        wav_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))

        print(f"Найдено {len(wav_files)} аудиофайлов")

        for audio_path in tqdm(wav_files, desc="Обработка аудиофайлов"):
            # Парсинг имени файла
            filename = os.path.basename(audio_path)
            file_info = self.parse_filename(filename)

            if file_info:
                # Извлечение эмоции
                emotion_code = file_info['emotion']
                emotion_label = self.ravdess_config['emotion_map'].get(emotion_code)

                if emotion_label is not None:
                    # Извлечение MFCC features
                    mfcc_features = self.extract_mfcc_features(audio_path)

                    if mfcc_features is not None:
                        features.append(mfcc_features)
                        labels.append(emotion_label)
                        metadata.append(file_info)

        # Преобразование в numpy arrays
        features = np.array(features)
        labels = np.array(labels)

        # Добавление channel dimension для совместимости с CNN
        features = np.expand_dims(features, -1)

        print(f"Форма features: {features.shape}")
        print(f"Распределение меток: {np.bincount(labels)}")

        # Сохранение обработанных данных
        np.save(os.path.join(output_dir, 'ravdess_audio_features.npy'), features)
        np.save(os.path.join(output_dir, 'ravdess_audio_labels.npy'), labels)

        # Сохранение метаданных
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(output_dir, 'ravdess_metadata.csv'), index=False)

        return features, labels, metadata_df

    def prepare_audio_data(self, features, labels, test_size=0.2):
        """Подготовка данных для обучения"""
        from sklearn.model_selection import train_test_split

        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        print(f"Audio Data - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
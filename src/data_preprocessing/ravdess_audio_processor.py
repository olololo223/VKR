import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import pickle
from sklearn.model_selection import train_test_split



class RavdessAudioProcessor:
    def __init__(self, config_path='config/model_config.json'):
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = json.load(file)

        self.audio_config = self.config['audio']
        self.ravdess_config = self.config['ravdess']

    def extract_mfcc_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.audio_config['sample_rate'])

            y = librosa.util.normalize(y)

            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.audio_config['n_mfcc'],
                n_fft=self.audio_config['n_fft'],
                hop_length=self.audio_config['hop_length'],
                n_mels=128,
                fmin=50,
                fmax=sr/2
            )

            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            mfcc_combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

            mfcc_combined = (mfcc_combined - np.mean(mfcc_combined)) / (np.std(mfcc_combined) + 1e-8)

            if mfcc_combined.shape[1] > self.audio_config['max_length']:
                mfcc_combined = mfcc_combined[:, :self.audio_config['max_length']]
            else:
                pad_width = self.audio_config['max_length'] - mfcc_combined.shape[1]
                mfcc_combined = np.pad(mfcc_combined, ((0, 0), (0, pad_width)), mode='constant')

            return mfcc_combined

        except Exception as e:
            print(f"Ошибка обработки файла {audio_path}: {e}")
            return None

    def parse_filename(self, filename):
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

    def process_ravdess_audio_dataset(self, dataset_path, output_dir=None):
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        features = []
        labels = []
        metadata = []

        wav_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))

        for audio_path in tqdm(wav_files, desc=""):
            filename = os.path.basename(audio_path)
            file_info = self.parse_filename(filename)

            if file_info:
                emotion_code = file_info['emotion']
                emotion_label = self.ravdess_config['emotion_map'].get(emotion_code)

                if emotion_label is not None:
                    try:
                        emotion_label = int(str(emotion_label).replace(',', '').replace(' ', ''))
                    except (ValueError, TypeError) as e:
                        print(f"Ошибка: {emotion_label}, error={e}")
                        continue

                    mfcc_features = self.extract_mfcc_features(audio_path)

                    if mfcc_features is not None:
                        features.append(mfcc_features)
                        labels.append(emotion_label)
                        metadata.append(file_info)

        if len(features) == 0:
            raise ValueError("")

        features = np.array(features)
        labels = np.array(labels, dtype=np.int32)

        features = np.expand_dims(features, -1)


        if save_results and output_dir:
            np.save(os.path.join(output_dir, 'ravdess_audio_features.npy'), features)
            np.save(os.path.join(output_dir, 'ravdess_audio_labels.npy'), labels)

            metadata_df = pd.DataFrame(metadata)
            metadata_df.to_csv(os.path.join(output_dir, 'ravdess_metadata.csv'), index=False)

            with open(os.path.join(output_dir, 'ravdess_config.pkl'), 'wb') as f:
                pickle.dump({
                    'features_shape': features.shape,
                    'emotion_map': self.ravdess_config['emotion_map'],
                    'audio_config': self.audio_config
                }, f)

        return features, labels, metadata

    def load_processed_data(self, data_dir):
        features = np.load(os.path.join(data_dir, 'ravdess_audio_features.npy'))
        labels = np.load(os.path.join(data_dir, 'ravdess_audio_labels.npy'))
        metadata = pd.read_csv(os.path.join(data_dir, 'ravdess_metadata.csv'))

        return features, labels, metadata

    def prepare_audio_data(self, features, labels, test_size=0.2, val_size=0.2):
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=(test_size + val_size), random_state=42, stratify=labels
        )

        temp_size = test_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=temp_size, random_state=42, stratify=y_temp
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

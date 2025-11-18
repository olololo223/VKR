import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import json
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight



class FERProcessor:
    def __init__(self, config_path='config/model_config.json'):
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = json.load(file)

        self.image_size = tuple(self.config['data']['image_size'])
        self.num_classes = self.config['data']['num_classes']
        self.emotions = self.config['data']['emotions']

        self.emotion_to_label = {emotion: idx for idx, emotion in enumerate(self.emotions)}

    def load_fer_data_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        print(f"Загружено {len(df)} записей из CSV")

        images = []
        labels = []

        image_dir = 'data/train'

        for idx, row in df.iterrows():
            image_name = row['Image name']
            usage = row['Usage']

            emotion_scores = {
                'neutral': row['neutral'],
                'happy': row['happiness'],
                'surprise': row['surprise'],
                'sad': row['sadness'],
                'angry': row['anger'],
                'disgust': row['disgust'],
                'fear': row['fear']
            }

            max_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            label = self.emotion_to_label[max_emotion]

            emotion_folder = os.path.join(image_dir, max_emotion)
            image_path = os.path.join(emotion_folder, image_name)

            if os.path.exists(image_path):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if image is not None:
                    if image.shape[:2] != self.image_size:
                        image = cv2.resize(image, self.image_size)

                    image = image.astype('float32') / 255.0

                    images.append(image)
                    labels.append(label)

        images = np.array(images)
        labels = np.array(labels)

        images = np.expand_dims(images, -1)

        return images, labels

    def load_fer_data_from_folders(self, data_dir='data/train'):
        images = []
        labels = []

        for emotion_name, label in self.emotion_to_label.items():
            emotion_folder = os.path.join(data_dir, emotion_name)

            if not os.path.exists(emotion_folder):
                continue

            for filename in os.listdir(emotion_folder):
                if filename.lower().endswith(('.jpg')):
                    image_path = os.path.join(emotion_folder, filename)

                    try:
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                        if image is not None:
                            image = image.astype('float32') / 255.0

                            images.append(image)
                            labels.append(label)
                    except Exception as e:
                        print(f"Ошибка при загрузке {image_path}: {e}")
        images = np.array(images)
        labels = np.array(labels)

        images = np.expand_dims(images, -1)

        return images, labels

    def prepare_data(self, data_dir='data/train', test_size=0.2):
        images, labels = self.load_fer_data_from_folders(data_dir)

        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_class_weights(self, labels):
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        return dict(enumerate(class_weights))

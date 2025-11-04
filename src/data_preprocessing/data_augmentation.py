from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


class DataAugmentor:
    def __init__(self):
        self.datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def augment_data(self, X, y, multiplier=2):
        """Аугментация данных"""
        augmented_images = []
        augmented_labels = []

        for i in range(len(X)):
            image = X[i]
            label = y[i]

            # Добавляем оригинальное изображение
            augmented_images.append(image)
            augmented_labels.append(label)

            # Генерируем аугментированные версии
            image_batch = np.expand_dims(image, 0)
            for augmented_image in self.datagen.flow(image_batch, batch_size=1):
                augmented_images.append(augmented_image[0])
                augmented_labels.append(label)

                if len(augmented_images) >= len(X) * multiplier:
                    break

        return np.array(augmented_images), np.array(augmented_labels)
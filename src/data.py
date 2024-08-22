import cv2
import numpy as np
import albumentations as A
import tensorflow as tf
import random
import os

from src import consts


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, dataset_dir, batch_size, resolution, shuffle):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.image_data = self.get_image_data()
        self.n_samples = len(self.image_data)
        self.resolution = resolution
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_image_data(self):
        image_data = []
        for class_name in os.listdir(self.dataset_dir):
            for file_name in os.listdir(os.path.join(self.dataset_dir, class_name)):
                image_data.append({"file_name": file_name, "class_name": class_name})
        return image_data

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_data = self.image_data[index * self.batch_size:(index + 1) * self.batch_size]

        Xs, ys = [], []
        for sample in batch_data:
            image_path = os.path.join(self.dataset_dir, sample["class_name"], sample["file_name"])
            image = cv2.imread(filename=image_path)
            Xs.append(self.preprocess_image(image))
            ys.append(consts.CLASSES.index(sample["class_name"]))

        return tf.stack(Xs), tf.stack(ys)

    def preprocess_image(self, image):

        # Convert the image from BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Define the Albumentations transformation pipeline
        transform = A.Compose([
            A.Resize(height=self.resolution[0], width=self.resolution[1]),  # Resize to target size
            A.HorizontalFlip(p=0.5),  # Random horizontal flip
            A.RandomBrightnessContrast(p=0.5),  # Random brightness and contrast adjustment
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            # Random shift, scale, and rotate
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # Random Gaussian blur
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize with ImageNet mean and std
        ])
        # Apply the transformations
        augmented = transform(image=image)
        processed_image = augmented['image']

        return processed_image

    def on_epoch_begin(self):
        'Updates indexes after each epoch'
        random.shuffle(self.image_data)


def get_data_generators(params: dict):
    # Generators
    training_generator = DataGenerator(dataset_dir=os.path.join(params['dataset_dir'], "train"),
                                       batch_size=params['batch_size'],
                                       resolution=consts.DATA_RESOLUTION,
                                       shuffle=True)
    validation_generator = DataGenerator(dataset_dir=os.path.join(params['dataset_dir'], "train"),
                                         batch_size=params['batch_size'],
                                         resolution=consts.DATA_RESOLUTION,
                                         shuffle=False)
    return {"training_generator": training_generator, "validation_generator": validation_generator}

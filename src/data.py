import cv2
import numpy as np
import albumentations as A
import tensorflow as tf
import random
import os

from src import consts


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, dataset_dir, resolution):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.image_data = self.get_image_data()
        self.n_samples = len(self.image_data)
        self.resolution = resolution

    def get_image_data(self):
        image_data = []
        for class_name in os.listdir(self.dataset_dir):
            for file_name in os.listdir(os.path.join(self.dataset_dir, class_name)):
                image_data.append({"file_name": file_name, "class_name": class_name})
        return image_data

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_samples

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        sample = self.image_data[index]
        image_path = os.path.join(self.dataset_dir, sample["class_name"], sample["file_name"])
        image = cv2.imread(r'{}'.format(image_path.decode(encoding="utf-8")))
        x = self.preprocess_image(image)
        y = consts.CLASSES.index(sample["class_name"].decode(encoding="utf-8"))

        return x, [y]

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


def get_data_generators(params: dict):
    output_signature = (
        tf.TensorSpec(
            shape=consts.DATA_RESOLUTION,
            dtype=tf.float32), tf.TensorSpec(
            shape=[1],
            dtype=tf.int32))
    # Generators
    training_generator = tf.data.Dataset.from_generator(DataGenerator,
                                                        args=[os.path.join(params['dataset_dir'], "train"),
                                                              consts.DATA_RESOLUTION],
                                                        output_signature=output_signature).cache().shuffle(50).batch(
        params["batch_size"]).prefetch(tf.data.AUTOTUNE)

    validation_generator = tf.data.Dataset.from_generator(DataGenerator,
                                                          args=[os.path.join(params['dataset_dir'], "val"),
                                                                consts.DATA_RESOLUTION],
                                                          output_signature=
                                                          output_signature).batch(params["batch_size"])

    test_generator = tf.data.Dataset.from_generator(DataGenerator,
                                                    args=[os.path.join(params['dataset_dir'], "test"),
                                                          consts.DATA_RESOLUTION],
                                                    output_signature=output_signature).batch(params["batch_size"])
    return {"training_generator": training_generator,
            "validation_generator": validation_generator,
            "test_generator": test_generator}

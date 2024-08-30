import os
import random

import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
import albumentations as A
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.consts import CLASSES


def evaluate_model(model: tf.keras.Model, data: dict, limit: int):
    metrics_dict = {}
    misclassified_examples_dict = {}
    for phase in ['training', 'validation', 'test']:
        predictions = model.predict(data[f'{phase}_generator'])
        metrics, misclassified_examples = calculate_metrics(predictions, data[f'{phase}_generator'],
                                                            original_images=data['original_test_images'],
                                                            generate_missclassified_examples=phase == "test",
                                                            limit=limit)
        metrics_dict[phase] = metrics
        misclassified_examples_dict[phase] = misclassified_examples
    return metrics_dict, misclassified_examples_dict


def save_model_summary(output_dir, model, filename='model_summary.txt'):
    """
    Saves the summary of a TensorFlow/Keras model to a file.

    Args:
        model: The TensorFlow/Keras model.
        filename: The name of the file to save the summary. Defaults to 'model_summary.txt'.
    """
    filename = 'misclassified_examples.png'
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        # Pass the file object to the model's summary method
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def calculate_metrics(predictions: tf.Tensor, ground_truth: tf.data.Dataset, generate_missclassified_examples: bool,
                      original_images: tf.data.Dataset, limit=30):
    # Convert predictions to class labels (i.e., get the index of the max softmax score)
    predicted_classes = tf.argmax(predictions, axis=1).numpy()

    # Convert ground truth from the dataset to a numpy array
    ground_truth_labels = tf.concat([y[:, 0] for x, y in ground_truth], axis=0).numpy()

    # Calculate accuracy
    accuracy = accuracy_score(ground_truth_labels, predicted_classes)

    # Calculate precision, recall, and F1 score
    precision = precision_score(ground_truth_labels, predicted_classes, average='weighted')
    recall = recall_score(ground_truth_labels, predicted_classes, average='weighted')
    f1 = f1_score(ground_truth_labels, predicted_classes, average='weighted')

    # Identify misclassified examples
    misclassified_indices = tf.where(predicted_classes != ground_truth_labels).numpy().flatten()
    selected_missclassified_indices = random.choices(misclassified_indices, k=limit)
    misclassified_examples = []

    if generate_missclassified_examples:
        ground_truth_images = tf.concat([x for x, y in original_images], axis=0).numpy()
        # Collect the misclassified examples
        for i, (image, label) in enumerate(zip(ground_truth_images, ground_truth_labels)):
            if i in selected_missclassified_indices:
                misclassified_examples.append({
                    "index": i,
                    "image": image,  # Convert TensorFlow tensor to NumPy array
                    "true_label": label,
                    "predicted_label": predicted_classes[i]
                })

    return {
               "accuracy": accuracy,
               "precision": float(precision),
               "recall": float(recall),
               "f1_score": float(f1),
           }, misclassified_examples


def save_misclassified_images(output_dir, misclassified_examples, num_images):
    """
    Saves a grid of misclassified images with true and predicted labels.

    Args:
        misclassified_examples: A list of dictionaries containing misclassified examples.
                                Each dictionary should have 'image', 'true_label', and 'predicted_label'.
        filename: The name of the file to save the image. Defaults to 'misclassified_examples.png'.
        num_images: Number of images to display in the grid (must be a perfect square). Defaults to 9.
    """
    if len(misclassified_examples) == 0:
        print("No misclassified examples to display.")
        return
    print(f"Dumping {min(len(misclassified_examples), num_images)} unlabeled examples")
    filename = 'misclassified_examples.png'
    filepath = os.path.join(output_dir, filename)
    # Ensure num_images is a perfect square
    side = int(num_images ** 0.5)
    assert side * side == num_images, "num_images must be a perfect square (e.g., 4, 9, 16)."

    plt.figure(figsize=(12, 12))
    indices = random.choices(range(len(misclassified_examples)), k=num_images)
    for i in indices:
        example = misclassified_examples[i]
        image = example['image']
        true_label = example['true_label']
        predicted_label = example['predicted_label']

        plt.subplot(side, side, i + 1)
        plt.imshow(image)  # Convert image to correct format for plotting
        plt.title(f"True: {CLASSES[true_label]}, Pred: {CLASSES[predicted_label]}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def save_training_curves(history, output_dir: str):
    """
    Generates and saves the loss and accuracy curves from the training history.

    Args:
        history: A TensorFlow History object.
        filename: The name of the file to save the plot. Defaults to 'training_curves.png'.
    """
    filename = 'training_curves.png'
    filepath = os.path.join(output_dir, filename)
    # Plotting loss
    plt.figure(figsize=(10, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def save_experiment_results(output_dir, model, history, evaluation_results, misclassified_examples):
    model.save(os.path.join(output_dir, "model.keras"))

    with open(os.path.join(output_dir, "history.yaml"), "w") as fp:
        yaml.dump(history.history, fp)

    save_model_summary(output_dir, model)

    with open(os.path.join(output_dir, "metrics.yaml"), "w") as fp:
        yaml.dump(evaluation_results, fp)

    save_training_curves(history=history, output_dir=output_dir)
    save_misclassified_images(output_dir=output_dir,
                              misclassified_examples=misclassified_examples["test"],
                              num_images=16)

import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model: tf.keras.Model, data: dict):
    metrics_dict = {}
    for phase in ['training', 'validation', 'test']:
        predictions = model.predict(data[f'{phase}_generator'])
        metrics = calculate_metrics(predictions, data[f'{phase}_generator'])
        metrics_dict[phase] = metrics


def calculate_metrics(predictions: tf.Tensor, ground_truth: tf.data.Dataset):
    # Convert predictions to class labels (i.e., get the index of the max softmax score)
    predicted_classes = tf.argmax(predictions, axis=1).numpy()

    # Convert ground truth from the dataset to a numpy array
    ground_truth_labels = tf.concat([y for x, y in ground_truth], axis=0).numpy()

    # Calculate accuracy
    accuracy = accuracy_score(ground_truth_labels, predicted_classes)

    # Calculate precision, recall, and F1 score
    precision = precision_score(ground_truth_labels, predicted_classes, average='weighted')
    recall = recall_score(ground_truth_labels, predicted_classes, average='weighted')
    f1 = f1_score(ground_truth_labels, predicted_classes, average='weighted')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

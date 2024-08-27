import tensorflow as tf


def evaluate_model(model: tf.keras.Model, data: dict):
    metrics_dict = {}
    for phase in ['training', 'validation', 'test']:
        predictions = model.predict(data[f'{phase}_generator'])
        metrics = calculate_metrics(predictions, data[f'{phase}_generator'])
        metrics_dict[phase] = metrics


def calculate_metrics(predictions, param):
    pass
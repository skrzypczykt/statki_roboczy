from src.data import get_data_generators
import yaml
import tensorflow as tf
from tensorflow.keras import mixed_precision

from src.model_evaluation import evaluate_model
from src.model_trainer import train_model

if __name__ == "__main__":
    tf.config.optimizer.set_jit(True)
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # Parameters
    with open("config.yaml", "r") as fp:
        params = yaml.safe_load(fp)

    device_name = tf.test.gpu_device_name()
    if len(device_name) > 0:
        print("Found GPU at: {}".format(device_name))
    else:
        device_name = "/device:CPU:0"
        print("No GPU, using {}.".format(device_name))

    data = get_data_generators(params)

    with tf.device(device_name):

        model, history = train_model(data=data, params=params)

    evaluation_results = evaluate_model(model=model, data=data)

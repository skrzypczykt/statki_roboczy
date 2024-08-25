from src.data import get_data_generators
import yaml
import tensorflow as tf
from tensorflow.keras import mixed_precision

from src.model_trainer import train_model

if __name__ == "__main__":
    tf.config.optimizer.set_jit(True)
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # Parameters
    with open("config.yaml", "r") as fp:
        params = yaml.safe_load(fp)

    data = get_data_generators(params)
    # print(training_generator[0])

    model, history = train_model(data=data, params=params)

    model.evaluate(data['test_generator'])

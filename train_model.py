from src.data import get_data_generators
import yaml
import tensorflow as tf
from tensorflow.keras import mixed_precision

from src.model_evaluation import evaluate_model, save_experiment_results
from src.model_trainer import train_model
from src.utils import initialize_output_dir

if __name__ == "__main__":
    tf.config.optimizer.set_jit(True)
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # Parameters
    with open("config.yaml", "r") as fp:
        params = yaml.safe_load(fp)

    output_dir: str = initialize_output_dir(params)

    data = get_data_generators(params)

    model, history = train_model(data=data, params=params)

    evaluation_results, misclassified_examples = evaluate_model(model=model, data=data, limit=16)

    save_experiment_results(output_dir=output_dir,
                            model=model,
                            history=history,
                            evaluation_results=evaluation_results,
                            misclassified_examples=misclassified_examples)

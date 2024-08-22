from src.data import get_data_generators
import yaml

from src.model_trainer import train_model

if __name__ == "__main__":
    # Parameters
    with open("config.yaml", "r") as fp:
        params = yaml.safe_load(fp)

    data = get_data_generators(params)
    # print(training_generator[0])

    model, history = train_model(data=data, params=params)

    model.evaluate(data['validation_generator'])

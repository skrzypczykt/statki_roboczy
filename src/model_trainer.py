from src import consts
from src.model import create_transfer_learning_model


def train_model(data, params):
    model = create_transfer_learning_model(input_shape=consts.DATA_RESOLUTION, num_classes=consts.N_CLASSES)
    model.summary()  # Print the model summary

    history = model.fit(data['training_generator'], epochs=params["epochs"])
    return model, history.history
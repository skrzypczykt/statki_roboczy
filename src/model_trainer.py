import tensorflo as tf
from src import consts
from src.model import create_transfer_learning_model


def train_model(data, params):
    model = create_transfer_learning_model(input_shape=consts.DATA_RESOLUTION, num_classes=consts.N_CLASSES,
                                           dropout_rate=params["dropout_rate"])
    model.summary()  # Print the model summary

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        verbose=1,
        mode='auto',
        restore_best_weights=True,
    )
    history = model.fit(data['training_generator'],
                        validation_data=data['validation_generator'],
                        epochs=params["epochs"],
                        shuffle=False,
                        callbacks=[es])  # , steps_per_epoch=1, validation_steps=1
    return model, history

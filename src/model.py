import tensorflow as tf
from tensorflow.keras import layers, models


def create_transfer_learning_model(input_shape, num_classes, dropout_rate=0.5):
    # Load the pre-trained EfficientNetB0 model without the top classification layer
    # base_model = tf.keras.applications.EfficientNetB0(
    #     include_top=False,
    #     weights='imagenet'
    # )

    # base_model = tf.keras.applications.ConvNeXtTiny(
    #     include_top=False,
    #     weights='imagenet'
    # )

    base_model = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    # Freeze the base model's layers to prevent them from being trained
    base_model.trainable = False

    # Create the model
    model = models.Sequential([
        layers.Input(shape=input_shape),
        base_model,  # Add the pre-trained model
        layers.GlobalAveragePooling2D(),  # Pool the features
        layers.Dropout(dropout_rate),  # Add dropout for regularization
        layers.Dense(num_classes, activation='softmax')  # Final classification layer
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

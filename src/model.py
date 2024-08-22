import tensorflow as tf
from tensorflow.keras import layers, models


def create_transfer_learning_model(input_shape, num_classes):
    # Load the pre-trained EfficientNetB0 model without the top classification layer
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model's layers to prevent them from being trained
    base_model.trainable = False

    # Create the model
    model = models.Sequential([
        base_model,  # Add the pre-trained model
        layers.GlobalAveragePooling2D(),  # Pool the features
        layers.Dropout(0.5),  # Add dropout for regularization
        layers.Dense(num_classes, activation='softmax')  # Final classification layer
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


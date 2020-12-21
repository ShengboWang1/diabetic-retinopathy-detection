from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras import layers, Model, Input


def densenet121(num_classes):
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(256, 256, 3)
    )
    # Freeze the base model.
    base_model.trainable = False
    inputs = Input(shape=(256, 256, 3))

    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    x = base_model(inputs, training=False)

    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    # outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)

    return model
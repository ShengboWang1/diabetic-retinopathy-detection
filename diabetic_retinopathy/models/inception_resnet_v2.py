from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import InceptionResNetV2

def inception_resnet_v2(num_classes=2):
    base_model = InceptionResNetV2(
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
    outputs = layers.Dense(num_classes)(x)
    # outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)

    return model

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

    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    inputs = Input((256, 256, 3))

    out = base_model(inputs, training=False)
    # out = base_model.output
    out = layers.GlobalAveragePooling2D()(out)
    out = layers.Dropout(0.2)(out)
    predictions = layers.Dense(num_classes, activation='softmax')(out)
    # model = k.Model(inputs=Inp, outputs=predictions)
    model = Model(inputs, outputs=predictions)
    return model

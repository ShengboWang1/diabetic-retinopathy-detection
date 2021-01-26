from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import InceptionResNetV2, InceptionV3, MobileNet


def output(base_model, inputs, num_classes):
    # make a small uniform model on the top of following transfer models
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return outputs


def inception_resnet_v2(num_classes=2):
    base_model = InceptionResNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(256, 256, 3)
    )

    # Freeze the base model.
    base_model.trainable = False
    inputs = Input(shape=(256, 256, 3))
    outputs = output(base_model, inputs, num_classes)
    model = Model(inputs, outputs)

    return model


def inception_v3(num_classes=2):
    base_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(256, 256, 3)
    )

    # Freeze the base model.
    base_model.trainable = False
    inputs = Input(shape=(256, 256, 3))
    outputs = output(base_model, inputs, num_classes)
    model = Model(inputs, outputs)
    return model


def mobilenet(num_classes=2):
    base_model = MobileNet(
        weights='imagenet',
        include_top=False,
        input_shape=(256, 256, 3)
    )

    # Freeze the base model.
    base_model.trainable = False
    inputs = Input(shape=(256, 256, 3))
    outputs = output(base_model, inputs, num_classes)
    model = Model(inputs, outputs)
    return model

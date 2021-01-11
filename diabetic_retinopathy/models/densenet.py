from tensorflow.keras.applications.densenet import DenseNet121
import tensorflow.keras as keras
import gin


@gin.configurable
def densenet121(num_classes, problem_type, dropout_rate, layer_index, dense_units):
    # Create base model
    base_model = keras.applications.DenseNet121(
        weights='imagenet',
        input_shape=(256, 256, 3),
        include_top=False)

    base_model = keras.Model(inputs=base_model.input, outputs=base_model.get_layer(index=layer_index).output)
    # Freeze base model
    base_model.trainable = False
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    # x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(dense_units, activation=None)(x)
    x = keras.layers.LeakyReLU()(x)
    if problem_type == 'classfication':
        predictions = keras.layers.Dense(num_classes, activation='softmax')(x)
    elif problem_type == 'regression':
        predictions = keras.layers.Dense(2, activation=None)(x)
    else:
        raise ValueError
    model = keras.Model(base_model.input, predictions, name='DenseNet121')

    return model


@gin.configurable
def densenet121_bigger(dropout_rate, layer_index, dense_units):
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(256, 256, 3)
    )

    base_model.trainable = True

    out = base_model.output
    # out = base_model.output
    out = keras.layers.GlobalAveragePooling2D()(out)
    out = keras.layers.Dense(dense_units)(out)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.LeakyReLU()(out)
    out = keras.layers.Dropout(dropout_rate)(out)
    out = keras.layers.Dense(2, activation=None)(out)
    outputs = keras.layers.Activation('softmax')(out)
    # model = k.Model(inputs=Inp, outputs=predictions)
    model = keras.Model(base_model.input, outputs, name='DenseNet121_bigger')
    return model

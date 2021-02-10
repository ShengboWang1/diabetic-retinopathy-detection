from tensorflow.keras.applications.densenet import DenseNet121
import tensorflow.keras as keras
import gin


@gin.configurable
def densenet121(num_classes, problem_type, dropout_rate, layer_index, dense_units):
    """A model consisting of a densenet_121 model and a small output model on the top.

        Parameters:
            num_classes (int): number of classification classes, used in  the final dense layer
            problem_type(string): to specify a classification or a regression output
            dropout_rate(float): dropout rate
            layer_index(int): number of layer that we reserve for base model
            dense_units(int): number of dense units
        Returns:
            model(keras.Model): keras model object
        """

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
    if problem_type == 'classification':
        predictions = keras.layers.Dense(num_classes, activation='softmax')(x)
    elif problem_type == 'regression':
        predictions = keras.layers.Dense(1, activation=None)(x)
    else:
        raise ValueError
    model = keras.Model(base_model.input, predictions, name='DenseNet121')

    return model


@gin.configurable
def densenet121_bigger(dropout_rate, dense_units):
    """A model consisting of a densenet_121 model and a bigger output model on the top."""
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

from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras import Sequential, layers, regularizers

def densenet121(num_classes):
    model = Sequential()
    densenet = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(256, 256, 3)
    )
    densenet.trainable = False
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

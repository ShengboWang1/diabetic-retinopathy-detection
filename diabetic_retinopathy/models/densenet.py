from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras import Sequential, layers

def densenet121(num_classes):
    model = Sequential()
    densenet = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(256, 256, 3)
    )
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_classes, activation='softmax'))

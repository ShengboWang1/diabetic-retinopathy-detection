from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import Sequential,layers

def inception_v3(num_classes):
    model = Sequential()
    inceptionv3 = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(256, 256, 3)
    )
    model.add(inceptionv3)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

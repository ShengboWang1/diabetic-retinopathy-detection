"""Define the CNN architecture for classification with 2 conv. and 2 dense layers."""
"""
# input image dimensions
img_rows, img_cols = 256, 256
#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 5
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
"""
import tensorflow as tf
class MyModel(k.model):
    def __init__(self):


        super(MyModel, self).__init__()
        # Layer definition
        self.conv0 = tf.keras.layers.Conv2D(32, 3, 3,
                                            border_mode='valid',
                                            input_shape=(256, 256),
                                            activation='relu')
        self.conv1 = tf.keras.layers.Conv2D(32, 3, 3, activation='relu')
        self.max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.flatten = tf.keras.layers.Flatten()
        self.dense0 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        # Call layers appropriately in order to implement the forward pass
        output = self.conv0(inputs)
        output = self.conv1(output)
        output = self.max_pool_2d(output)
        output = self.flatten(output)
        output = self.dropout(output)
        output = self.dense0(output)
        output = self.dropout(output)
        output = self.dense1(output)

        return output
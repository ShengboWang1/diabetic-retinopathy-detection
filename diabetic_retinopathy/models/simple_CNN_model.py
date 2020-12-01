"""Define the CNN for classification with 2 conv. and 2 dense layers."""
import tensorflow as tf
import tensorflow.keras as k
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
        # Call layers appropriately to implement a forward pass
        output = self.conv0(inputs)
        output = self.conv1(outputs)
        output = self.max_pool_2d(outputs)
        output = self.dropout(outputs)
        output = self.flatten(outputs)
        output = self.dense0(outputs)
        output = self.dropout(outputs)
        output = self.dense1(outputs)

        return output
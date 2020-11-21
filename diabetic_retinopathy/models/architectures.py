import gin
import tensorflow as tf

from models.layers import vgg_block

@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines a VGG-like architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate

    Returns:
        (keras.Model): keras model object
    """

    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    inputs = tf.keras.Input(input_shape)
    out = vgg_block(inputs, base_filters)
    for i in range(2, n_blocks):
        out = vgg_block(out, base_filters * 2 ** (i))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')



"""
# input image dimensions
img_rows, img_cols = 200, 200
#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 5
# number of epochs to train
nb_epoch = 5
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
(X, y) = (train_data[0],train_data[1])"""

"""Define the CNN for classification with 2 conv. and 2 dense layers."""
import tensorflow as tf
class MyModel(k.model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Layer definition
        self.conv0 = tf.keras.layers.Conv2D(32, 3, 3,
                                            border_mode='valid',
                                            input_shape=(200, 200),
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
        output = self.conv0
        output = self.conv1
        output = self.max_pool_2d
        output = self.dropout
        output = self.flatten
        output = self.dense0
        output = self.dropout
        output = self.dense1

        return output




print(model.summary())



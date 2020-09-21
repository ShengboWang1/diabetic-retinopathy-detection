import gin
import tensorflow as tf

@gin.configurable
def vgg_block(inputs, filters, kernel_size):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.

    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)

    Returns:
        (Tensor): output of the VGG block
    """

    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    return out
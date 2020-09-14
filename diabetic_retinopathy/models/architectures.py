import tensorflow as tf

from models.layers import vgg_block

def vgg_like(input_shape, n_classes):
    inputs = tf.keras.Input(input_shape)
    out = vgg_block(inputs, 16)
    out = vgg_block(out, 32)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="vgg_like")
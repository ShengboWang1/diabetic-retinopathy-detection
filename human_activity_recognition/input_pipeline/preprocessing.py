import gin
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa


@gin.configurable
def preprocess(feature, label):
    """Dataset preprocessing: Normalizing and resizing"""
    # feature = tf.cast(feature, tf.float32)
    # label = tf.cast(label, tf.int32)
    return feature, label

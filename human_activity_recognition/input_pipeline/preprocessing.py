import gin
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa


@gin.configurable
def preprocess(feature, label):
    """Dataset preprocessing: Normalizing and resizing"""
    label -= 1
    return feature, label

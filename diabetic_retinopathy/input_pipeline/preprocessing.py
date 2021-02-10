import gin
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa


@gin.configurable
def preprocess(image, label, img_height, img_width, model_name):
    """Dataset preprocessing: Normalizing and resizing"""
    # original shpae (2848, 4288, 3)print(image.shape)
    image = tf.image.crop_to_bounding_box(image, 0, 266, 2848, 3426)
    # (2848, 3426, 3)
    image = tf.image.pad_to_bounding_box(image, 289, 0, 3426, 3426)
    # (3426, 2436, 3)
    image = tf.image.resize(image, size=(img_height, img_width))
    # (256,256,3)
    image = tf.cast(image, tf.float32)

    if model_name == 'vgg':
        image = image / 255.0

    elif model_name == 'resnet18':
        image = image / 255.0

    elif model_name == 'resnet34':
        image = image / 255.0

    elif model_name == 'resnet50':
        image = tf.keras.applications.resnet.preprocess_input(image)

    elif model_name == 'densenet121' or model_name == 'densenet121_bigger':
        image = tf.keras.applications.densenet.preprocess_input(image)

    elif model_name == 'inception_v3':
        image = tf.keras.applications.inception_v3.preprocess_input(image)

    elif model_name == 'mobilenet':
        image = tf.keras.applications.mobilenet.preprocess_input(image)

    elif model_name == 'inception_resnet_v2':
        image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)

    else:
        raise ValueError

    return image, label


@gin.configurable
def augment(image, label):
    """Data augmentation"""

    # Randomly rotate the image by +- 0.125pi
    random_angles = tf.random.uniform(shape=(), minval=-np.pi / 8, maxval=np.pi / 8)
    image = tfa.image.rotate(image, random_angles)

    # 50% possibility up to down flipping
    image = tf.image.random_flip_up_down(image)

    # 50% possibility left to right flipping
    image = tf.image.random_flip_left_right(image)

    # Randomly crop the image from left and right sides and scale it to the original size
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    in_h = image.shape[0]
    in_w = image.shape[1]
    scaling = tf.random.uniform([2], 0.8, 1)
    x_scaling = scaling[0]
    y_scaling = scaling[1]
    out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
    out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
    seed = np.random.randint(2020)
    image = tf.image.random_crop(image, size=[out_h, out_w, 3], seed=seed)
    image = tf.image.resize(image, size=(in_h, in_w))

    # Random shearing
    x_shear = tf.random.uniform([1], minval=-0.1, maxval=0.1, dtype=tf.float32)[0]
    y_shear = tf.random.uniform([1], minval=-0.1, maxval=0.1, dtype=tf.float32)[0]
    image = tfa.image.transform(image, [1.0, x_shear, 0, y_shear, 1.0, 0.0, 0.0, 0.0])

    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.1)

    # Random saturation
    image = tf.image.random_saturation(image, lower=0.75, upper=1.5)

    # Random hue
    image = tf.image.random_hue(image, max_delta=0.01)

    # Random contrast
    image = tf.image.random_contrast(image, lower=0.75, upper=1.5)

    return image, label

import gin
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa



@gin.configurable
def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    tf.cast(image, tf.float32) / 255.

    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))
    image = tf.keras.applications.resnet.preprocess_input(image)
    return image, label

# all the possible operations here, need to separate them afterwards

@gin.configurable
def augment(image, label):
    """Data augmentation"""

    # the possibility of 90 190 270 0 degrees are the same
    def rotation(image, label):
        image =tfa.image.rotate
        # image = tf.image.rot90(image, k=tf.random.uniform([1], minval=0, maxval=4, dtype=tf.int32)[0])
        return image, label

    # likely tp flipp the image from left to right
    def flipping(image, label):
        # 50% possibility up to down
        image = tf.image.random_flip_up_down(image)
        # 50% possibility left to right
        image = tf.image.random_flip_left_right(image)
        return image, label

    # random crop the image and scale it to the original size
    def cropping(image, label):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        in_h, in_w, channel = image.get_shape().as_list()
        scaling = tf.random.uniform([2], 0.75, 1)
        x_scaling = scaling[0]
        y_scaling = scaling[1]
        out_h = tf.cast(in_h * x_scaling, dtype=tf.int32)
        out_w = tf.cast(in_w * y_scaling, dtype=tf.int32)
        seed = np.random.randint(1234)
        image = tf.image.random_crop(image, size=[out_h, out_w, channel], seed=seed)
        image = tf.image.resize(image, size=(in_h, in_w))
        print("cropped again")
        return image, label

    # shearing with random intensity from 0 to 60
    def shearing(image, label):
        intensity = tf.random.uniform([1], minval=0, maxval=60, dtype=tf.int32)[0]
        image = tf.keras.preprocessing.image.random_shear(image, intensity, row_axis=0, col_axis=1, channel_axis=2)
        return image, label

    image, label = cropping(image, label)
    image, label = flipping(image, label)
    image, label = rotation(image, label)
    # if operation == "cropping":
    #     image, label = cropping(image, label)
    #
    # elif operation == "flipping":
    #     image, label = flipping(image, label)
    #
    # elif operation == "rotation":
    #     image, label = rotation(image, label)
    #
    # elif operation == "shearing":
    #     image, label = shearing(image, label)
    #
    # else:
    #     raise ValueError

    return image, label


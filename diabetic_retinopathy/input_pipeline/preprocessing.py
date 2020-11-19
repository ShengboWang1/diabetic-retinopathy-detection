import gin
import tensorflow as tf

@gin.configurable
def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    tf.cast(image, tf.float32) / 255.

    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))

    return image, label

def augment(image, label):
    """Data augmentation"""
    # all the possible methods here, need to separate them afterwards

    def flipping(image, label):
        # up to down
        #image = tf.image.flip_up_down(image)
        # left to right
        image = tf.image.flip_left_right(image)
        return image, label
    # random scaling
    def scaling(image, label):
        batch_size, in_h, in_w, _ = image.get_shape().as_list()
        scaling = tf.random_uniform([2], 1, 1.15)
        x_scaling = scaling[0]
        y_scaling = scaling[1]
        out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
        out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
        image = tf.image.resize_area(image, [out_h, out_w])
        return image, label

    # random cropping
    def cropping(image, label):
        batch_size, in_h, in_w, _ = tf.unstack(tf.shape(image))
        offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
        offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
        image = tf.image.crop_to_bounding_box(
            image, offset_y, offset_x, 256, 256)
        return image, label

    image = flipping(image, label)
    image = scaling(image, label)
    image = cropping(image, label)
    return image, label

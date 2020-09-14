import os
import glob

import tensorflow as tf
import tensorflow_datasets as tfds

from absl import logging

def load(name):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")
        # ...

        return


    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        # ...

        return

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True
        )

        def normalize_img(image, label):
            """Normalizes images: `uint8` -> `float32`."""
            return tf.cast(image, tf.float32) / 255., label

        ds_train = ds_train.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(128)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        ds_test = ds_test.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.batch(128)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        return ds_train, ds_test, None, ds_info

    else:
        raise ValueError
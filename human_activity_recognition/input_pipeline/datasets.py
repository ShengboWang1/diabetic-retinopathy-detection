import gin
import logging
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from input_pipeline.preprocessing import preprocess
from input_pipeline.create_tfrecord import create_tfr


@gin.configurable
def load(device_name, name, data_dir_local, data_dir_gpu, data_dir_colab):
    """Load hapt dataset"""
    if name == "hapt":
        logging.info(f"Preparing dataset {name}...")
        print(device_name)

        # Get window size
        window_size = create_tfr(device_name=device_name)

        # Find corresponding TFRecord files for differnet devices
        if device_name == 'local':
            train_filename = data_dir_local + "no0_train.tfrecord"
            val_filename = data_dir_local + "no0_val.tfrecord"
            test_filename = data_dir_local + "no0_test.tfrecord"
        elif device_name == 'iss GPU':
            train_filename = data_dir_gpu + "no0_train.tfrecord"
            val_filename = data_dir_gpu + "no0_val.tfrecord"
            test_filename = data_dir_gpu + "no0_test.tfrecord"
        elif device_name == 'Colab':
            train_filename = data_dir_colab + "no0_train.tfrecord"
            val_filename = data_dir_colab + "no0_val.tfrecord"
            test_filename = data_dir_colab + "no0_test.tfrecord"
        else:
            raise ValueError

        # Get datasets from TFRecords
        raw_ds_train = tf.data.TFRecordDataset(train_filename)
        raw_ds_val = tf.data.TFRecordDataset(val_filename)
        raw_ds_test = tf.data.TFRecordDataset(test_filename)

        feature_description = {
            'feature': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }

        def _parse_function(exam_proto):
            """Get features and labels"""
            temp = tf.io.parse_single_example(exam_proto, feature_description)
            feature = tf.io.parse_tensor(temp['feature'], out_type=tf.float64)
            label = tf.io.parse_tensor(temp['label'], out_type=tf.int64)
            return (feature, label)

        ds_train = raw_ds_train.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = raw_ds_val.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = raw_ds_test.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # for data in ds_train.take(2):
        #     print(data)
        return prepare(ds_train, ds_val, ds_test, window_size)


@gin.configurable
def prepare(ds_train, ds_val, ds_test, window_size, batch_size, caching):

    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if caching:
        ds_train = ds_train.cache("train_cache")

    # ds_train = ds_train.map(
    #     augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_train = ds_train.shuffle(1000)
    # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_val = ds_val.cache("val_cache")
    ds_val = ds_val.batch(batch_size)

    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_test = ds_test.cache("test_cache")
    ds_test = ds_test.batch(batch_size)

    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, window_size

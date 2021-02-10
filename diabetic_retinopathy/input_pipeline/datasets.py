import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from input_pipeline.preprocessing import preprocess, augment


@gin.configurable
def load(device_name, dataset_name, n_classes, data_dir_local, data_dir_gpu, data_dir_colab):
    # Load idrid dataset
    if dataset_name == "idrid":
        logging.info(f"Preparing dataset {dataset_name}...")
        # 2 classes
        print(device_name)
        # Find corresponding TFRecord files for 2/5 classes problem and for differnet devices
        if device_name == 'local':
            train_filename = "/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/idrid-" + str(n_classes) + "balanced-train.tfrecord-00000-of-00001"
            test_filename = "/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/idrid-" + str(n_classes) + "balanced-test.tfrecord-00000-of-00001"
        elif device_name == 'iss GPU':
            train_filename = "/home/RUS_CIP/st169852/st169852/dl-lab-2020-team06/diabetic_retinopathy/idrid-" + str(n_classes) + "balanced-train.tfrecord-00000-of-00001"
            test_filename = "/home/RUS_CIP/st169852/st169852/dl-lab-2020-team06/diabetic_retinopathy/idrid-" + str(n_classes) + "balanced-test.tfrecord-00000-of-00001"
        elif device_name == 'Colab':
            train_filename = "/content/drive/MyDrive/diabetic_retinopathy/idrid-" + str(n_classes) + "balanced-train.tfrecord-00000-of-00001"
            test_filename = "/content/drive/MyDrive/diabetic_retinopathy/idrid-" + str(n_classes) + "balanced-test.tfrecord-00000-of-00001"
        else:
            raise ValueError

        # Get datasets from TFRecords
        raw_ds_train = tf.data.TFRecordDataset(train_filename)
        ds_test = tf.data.TFRecordDataset(test_filename)
        ds_info = "idrid"

        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            'img_height': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            'img_width': tf.io.FixedLenFeature([], tf.int64, default_value=-1)
        }

        def _parse_function(exam_proto):
            """Get images and labels"""
            temp = tf.io.parse_single_example(exam_proto, feature_description)
            img = tf.io.decode_jpeg(temp['image'], channels=3)
            img = tf.reshape(img, [2848, 4288, 3])
            label = temp['label']
            return (img, label)

        raw_ds_train = raw_ds_train.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Split train dataset into train and validation set
        total_record_num = 750
        raw_ds_train = raw_ds_train.shuffle(total_record_num // 10)
        ds_val = raw_ds_train.take(150)
        ds_train = raw_ds_train.skip(150)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    # Load eyepacs dataset
    elif dataset_name == "eyepacs":
        logging.info(f"Preparing dataset {dataset_name}...")
        if device_name == 'local':
            data_dir = data_dir_local
        if device_name == 'iss GPU':
            data_dir = data_dir_gpu
        if device_name == 'Colab':
            data_dir = data_dir_colab

        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300:3.0.0',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):

    # Visualize the input image
    # i = 0
    # for image, label in iter(ds_train):
    #     if i < 10:
    #         plt.imshow(tf.cast(image, tf.int64))
    #         plt.axis('off')
    #         plt.savefig(os.path.join('/Users/shengbo/Documents/idrid_pictures', str(i) + 'original' + '.png'))
    #         plt.show()
    #
    #         image, label = preprocess(image, label)
    #         image = image * 255.0
    #         plt.imshow(tf.cast(image, tf.int64))
    #         plt.axis('off')
    #         plt.savefig(os.path.join('/Users/shengbo/Documents/idrid_pictures', str(i) + 'preprocess' + '.png'))
    #         plt.show()
    #
    #         image, label = augment(image, label)
    #         plt.imshow(tf.cast(image, tf.int64))
    #         plt.axis('off')
    #         plt.savefig(os.path.join('/Users/shengbo/Documents/idrid_pictures', str(i) + 'augmentation' + '.png'))
    #         plt.show()
    #     i += 1

    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if caching:
        ds_train = ds_train.cache("train_cache")

    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_train = ds_train.shuffle(300)
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
    ds_test = ds_test.batch(batch_size=103)

    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info

import os
from glob import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import gin


@gin.configurable
def create_tfr(shift_window_size, window_size, device_name):
    """Create TFRecord files according to the shift size and window size"""
    if device_name == 'local':
        base_datadir = '/Users/shengbo/Documents/Github/dl-lab-2020-team06/HAPT_dataset/'
    elif device_name == 'iss GPU':
        base_datadir = '/home/data/HAPT_dataset/'
    elif device_name == 'Colab':
        base_datadir = '/content/drive/MyDrive/HAPT_dataset/'
    else:
        raise ValueError

    def load_file(filepath, names):
        """Load file to transfer txt file into pandas dataframe"""
        dataframe = pd.read_table(filepath, names=names, delim_whitespace=True)
        return dataframe

    # Load activity labels
    activity_labels_df = load_file(base_datadir + 'activity_labels.txt',
                                   names=["number", "label"])

    # Load raw labels
    rawdata_path = base_datadir + 'RawData/'
    raw_labels_df = load_file(rawdata_path + 'labels.txt',
                              names=["experiment", "user_id",
                                     "activity", "start_pos",
                                     "end_pos"])

    # Sort all acc and gyro files and load them into pandas dataframe
    acc_pattern = 'acc_exp*.txt'
    gyro_pattern = 'gyro_exp*.txt'
    acc_files = glob(os.path.join(rawdata_path, acc_pattern))
    gyro_files = glob(os.path.join(rawdata_path, gyro_pattern))
    acc_files.sort()
    gyro_files.sort()
    file_info = pd.DataFrame()
    file_info['acc'] = [os.path.basename(x) for x in acc_files]
    file_info['gyro'] = [os.path.basename(x) for x in gyro_files]
    file_info['experiment'] = file_info['acc'].apply(lambda x: x[7:9])
    file_info['user_ID'] = file_info['acc'].apply(lambda x: x[14:16])

    def read_rawdata(experiment, user_id):
        """Read contents of single file to a dataframe with acc and gyro data."""
        read_rawdata_acc_path = rawdata_path + 'acc' + '_exp' + experiment + '_user' + user_id + '.txt'
        read_rawdata_gyro_path = rawdata_path + 'gyro' + '_exp' + experiment + '_user' + user_id + '.txt'
        rawfile_acc_df = load_file(read_rawdata_acc_path, ['a_x', 'a_y', 'a_z'])
        rawfile_gyro_df = load_file(read_rawdata_gyro_path, ['g_x', 'g_y', 'g_z'])
        rawfile_df = pd.concat([rawfile_acc_df, rawfile_gyro_df], axis=1)
        return rawfile_df

    def label_rawdata(experiment, user_id):
        """Function to read a given file and get the labels of the observations"""
        get_raw_labels_df = raw_labels_df[
            (raw_labels_df["experiment"] == int(experiment)) & (raw_labels_df["user_id"] == int(user_id))]
        label_list_df = pd.DataFrame(0, index=np.arange(read_rawdata(experiment, user_id).shape[0]), columns=['label'])
        for i in get_raw_labels_df.index:
            start_pos = get_raw_labels_df['start_pos'][i]
            end_pos = get_raw_labels_df['end_pos'][i]
            label_list_df.loc[start_pos - 1:end_pos - 1, 'label'] = get_raw_labels_df['activity'][i]
        return label_list_df

    # Split raw data into train, validation and test datasets
    # Train dataset: user-01 to user-21
    # Validation dataset: user-28 to user-30
    # Test dataset: user-22 to user-27

    train_x = pd.DataFrame()
    val_x = pd.DataFrame()
    test_x = pd.DataFrame()

    train_y = pd.DataFrame()
    val_y = pd.DataFrame()
    test_y = pd.DataFrame()

    def create_dataset(start_num, end_num, ds_x, ds_y, file_info=file_info, x={}, y={}):
        """Conbine the files to create training, validation and test set, return 6-axial data and label pandas dataframe"""
        for i in range(start_num, end_num):
            x[i] = read_rawdata(file_info['experiment'][i], file_info['user_ID'][i])
            a = len(x[i])
            # print("a")
            # print(a)
            delete_length = (a - window_size) % shift_window_size
            # print(delete_length)
            x[i] = x[i][:-delete_length]
            # print(len(x[i]))
            y[i] = label_rawdata(file_info['experiment'][i], file_info['user_ID'][i])[:-delete_length]
            # print(len(y[i]))
            ds_x = ds_x.append(x[i], ignore_index=True)
            ds_y = ds_y.append(y[i], ignore_index=True)

        ds_all = pd.concat([ds_x, ds_y], axis=1)
        # print(ds_all)
        ds_all = ds_all.drop(ds_all[ds_all.label == 0].index)
        # print(ds_all)
        ds_x = ds_all[['a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z']]
        ds_y = ds_all[['label']]

        return ds_x, ds_y

    train_x, train_y = create_dataset(start_num=0, end_num=43, ds_x=train_x, ds_y=train_y)
    val_x, val_y = create_dataset(start_num=56, end_num=61, ds_x=val_x, ds_y=val_y)
    test_x, test_y = create_dataset(start_num=44, end_num=55, ds_x=test_x, ds_y=test_y)

    def z_score(df):
        """Z-score normalization"""
        # copy the dataframe
        df_std = df.copy()
        # apply the z-score method
        for column in df_std.columns:
            df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()

        return df_std

    train_x = z_score(train_x)
    val_x = z_score(val_x)
    test_x = z_score(test_x)

    # print(train_x, train_y)
    # print(val_x, val_y)
    # print(test_x, test_y)

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).window(size=window_size, shift=shift_window_size,
                                                                             drop_remainder=True)
    train_ds = train_ds.flat_map(lambda f_acc_gyro, label: tf.data.Dataset.zip((f_acc_gyro, label))).batch(window_size,
                                                                                                           drop_remainder=True)
    val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y)).window(size=window_size, shift=shift_window_size,
                                                                       drop_remainder=True)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).window(size=window_size, shift=shift_window_size,
                                                                          drop_remainder=True)

    val_ds = val_ds.flat_map(lambda f_acc_gyro, label: tf.data.Dataset.zip((f_acc_gyro, label))).batch(window_size,
                                                                                                       drop_remainder=True)
    test_ds = test_ds.flat_map(lambda f_acc_gyro, label: tf.data.Dataset.zip((f_acc_gyro, label))).batch(window_size,
                                                                                                         drop_remainder=True)

    # print(train_ds)
    # print(val_ds)
    # print(test_ds)

    # for f_acc_gyro, label in train_ds.take(1):
    #     print(f_acc_gyro)
    #     print(f_acc_gyro)

    # The following functions can be used to convert a value to a type compatible
    # with tf.Example.

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def window_example(f_acc_gyro, label):
        """
        Creates a tf.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        f_acc_gyro = tf.io.serialize_tensor(f_acc_gyro).numpy()
        label = tf.io.serialize_tensor(label).numpy()
        feature = {
            'feature': _bytes_feature(f_acc_gyro),
            'label': _bytes_feature(label)
        }
        # Create a Features message using tf.train.Example.

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def tf_window_example(f_acc_gyro, label):
        """Specify the shape and type information that is otherwise unavailable"""
        tf_string = tf.py_function(
            window_example,
            (f_acc_gyro, label),  # pass these args to the above function.
            tf.string)  # the return type is `tf.string`.
        return tf.reshape(tf_string, ())  # The result is a scalar

    train_map_ds = train_ds.map(tf_window_example)
    val_map_ds = val_ds.map(tf_window_example)
    test_map_ds = test_ds.map(tf_window_example)

    # print(train_map_ds)
    # print(val_map_ds)
    # print(test_map_ds)

    def generator_train():
        for features in train_ds:
            yield window_example(*features)

    def generator_val():
        for features in val_ds:
            yield window_example(*features)

    def generator_test():
        for features in test_ds:
            yield window_example(*features)

    serialized_train_ds = tf.data.Dataset.from_generator(
        generator_train, output_types=tf.string, output_shapes=())

    serialized_val_ds = tf.data.Dataset.from_generator(
        generator_val, output_types=tf.string, output_shapes=())

    serialized_test_ds = tf.data.Dataset.from_generator(
        generator_test, output_types=tf.string, output_shapes=())

    if device_name == 'iss GPU':
        train_filename = '/home/RUS_CIP/st169852/st169852/dl-lab-2020-team06/human_activity_recognition/no0_train.tfrecord'
        val_filename = '/home/RUS_CIP/st169852/st169852/dl-lab-2020-team06/human_activity_recognition/no0_val.tfrecord'
        test_filename = '/home/RUS_CIP/st169852/st169852/dl-lab-2020-team06/human_activity_recognition/no0_test.tfrecord'

    elif device_name == 'local':
        train_filename = '/Users/shengbo/Documents/Github/dl-lab-2020-team06/human_activity_recognition/no0_train.tfrecord'
        val_filename = '/Users/shengbo/Documents/Github/dl-lab-2020-team06/human_activity_recognition/no0_val.tfrecord'
        test_filename = '/Users/shengbo/Documents/Github/dl-lab-2020-team06/human_activity_recognition/no0_test.tfrecord'

    elif device_name == 'Colab':
        train_filename = '/content/drive/MyDrive/human_activity_recognition/no0_train.tfrecord'
        val_filename = '/content/drive/MyDrive/human_activity_recognition/no0_val.tfrecord'
        test_filename = '/content/drive/MyDrive/human_activity_recognition/no0_test.tfrecord'

    else:
        raise ValueError

    writer = tf.data.experimental.TFRecordWriter(train_filename)
    writer.write(serialized_train_ds)

    writer = tf.data.experimental.TFRecordWriter(val_filename)
    writer.write(serialized_val_ds)

    writer = tf.data.experimental.TFRecordWriter(test_filename)
    writer.write(serialized_test_ds)

    return window_size

# create_tfr(device_name='local', shift_window_size=125, window_size=250)

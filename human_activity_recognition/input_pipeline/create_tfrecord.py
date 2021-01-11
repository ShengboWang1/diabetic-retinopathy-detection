import os
from glob import glob
import tensorflow as tf
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as display


base_datadir = '/Users/shengbo/Documents/Github/dl-lab-2020-team06/HAPT_dataset/'


# load file to transfer txt file into pandas dataframe
def load_file(filepath, names):
    dataframe = pd.read_table(filepath, names=names, delim_whitespace=True)
    return dataframe


# load activity labels
activity_labels_df = load_file(base_datadir + 'activity_labels.txt',
                               names=["number", "label"])


# load raw labels
rawdata_path = base_datadir + 'RawData/'
raw_labels_df = load_file(rawdata_path + 'labels.txt',
                          names=["experiment", "user_id",
                                 "activity", "start_pos",
                                 "end_pos"])


# sort all acc and gyro files and load them into pandas dataframe
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


# read contents of single file to a dataframe with acc and gyro data.
def read_rawdata(experiment, user_id):
    read_rawdata_acc_path = rawdata_path + 'acc' + '_exp' + experiment + '_user' + user_id + '.txt'
    read_rawdata_gyro_path = rawdata_path + 'gyro' + '_exp' + experiment + '_user' + user_id + '.txt'
    rawfile_acc_df = load_file(read_rawdata_acc_path, ['a_x', 'a_y', 'a_z'])
    rawfile_gyro_df = load_file(read_rawdata_gyro_path, ['g_x', 'g_y', 'g_z'])
    rawfile_df = pd.concat([rawfile_acc_df, rawfile_gyro_df], axis=1)
    return rawfile_df


# function to read a given file and get the labels of the observations
def label_rawdata(experiment, user_id):
    get_raw_labels_df = raw_labels_df[(raw_labels_df["experiment"] == int(experiment)) & (raw_labels_df["user_id"] == int(user_id))]
    label_list_df = pd.DataFrame(0, index=np.arange(read_rawdata(experiment, user_id).shape[0]), columns=['label'])
    for i in get_raw_labels_df.index:
        start_pos = get_raw_labels_df['start_pos'][i]
        end_pos = get_raw_labels_df['end_pos'][i]
        label_list_df.loc[start_pos-1:end_pos-1, 'label'] = get_raw_labels_df['activity'][i]
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

for i in range(43):
    train_x = train_x.append(read_rawdata(file_info['experiment'][i], file_info['user_ID'][i]), ignore_index=True)
    train_y = train_y.append(label_rawdata(file_info['experiment'][i], file_info['user_ID'][i]), ignore_index=True)

for i in range(56, 61):
    val_x = val_x.append(read_rawdata(file_info['experiment'][i], file_info['user_ID'][i]), ignore_index=True)
    val_y = val_y.append(label_rawdata(file_info['experiment'][i], file_info['user_ID'][i]), ignore_index=True)

for i in range(44, 56):
    test_x = test_x.append(read_rawdata(file_info['experiment'][i], file_info['user_ID'][i]), ignore_index=True)
    test_y = test_y.append(label_rawdata(file_info['experiment'][i], file_info['user_ID'][i]), ignore_index=True)

#print(train_x)
#print(val_x)
#print(test_x)


# input normalization, Z-score normalization
# apply the z-score method in Pandas using the .mean() and .std() methods
def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()

    return df_std

train_x = z_score(train_x)
val_x = z_score(val_x)
test_x = z_score(test_x)

print(train_x, train_y)
print(val_x, val_y)
print(test_x, test_y)


shift_window_size = 125
window_size = 250


train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).window(size=window_size, shift=shift_window_size, drop_remainder=True)
val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y)).window(size=window_size, shift=shift_window_size, drop_remainder=True)
test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).window(size=window_size, shift=shift_window_size, drop_remainder=True)

train_ds = train_ds.flat_map(lambda f_acc_gyro, label: tf.data.Dataset.zip((f_acc_gyro, label))).batch(window_size, drop_remainder=True)
val_ds = val_ds.flat_map(lambda f_acc_gyro, label: tf.data.Dataset.zip((f_acc_gyro, label))).batch(window_size, drop_remainder=True)
test_ds = test_ds.flat_map(lambda f_acc_gyro, label: tf.data.Dataset.zip((f_acc_gyro, label))).batch(window_size, drop_remainder=True)


print(train_ds)
print(val_ds)
print(test_ds)

for f_acc_gyro, label in train_ds.take(1):
    print(f_acc_gyro)
    print(f_acc_gyro)

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
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
  tf_string = tf.py_function(
    window_example,
    (f_acc_gyro, label),  # pass these args to the above function.
    tf.string)      # the return type is `tf.string`.
  return tf.reshape(tf_string, ())  # The result is a scalar


print(tf_window_example(f_acc_gyro, label))


train_map_ds = train_ds.map(tf_window_example)
val_map_ds = val_ds.map(tf_window_example)
test_map_ds = test_ds.map(tf_window_example)

print(train_map_ds)
print(val_map_ds)
print(test_map_ds)


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

train_filename = 'train.tfrecord'
val_filename = 'val.tfrecord'
test_filename = 'test.tfrecord'

writer = tf.data.experimental.TFRecordWriter(train_filename)
writer.write(serialized_train_ds)

writer = tf.data.experimental.TFRecordWriter(val_filename)
writer.write(serialized_val_ds)

writer = tf.data.experimental.TFRecordWriter(test_filename)
writer.write(serialized_test_ds)

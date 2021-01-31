import csv
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# def five2two(df):
#     for rIndex in df.index:
#         if df.loc[rIndex, 'Retinopathy grade'] == 0 or df.loc[rIndex, 'Retinopathy grade'] == 1:
#             df.loc[rIndex, 'Retinopathy grade'] = 0
#         elif df.loc[rIndex, 'Retinopathy grade'] == 2 or df.loc[rIndex, 'Retinopathy grade'] == 3 \
#                 or df.loc[rIndex, 'Retinopathy grade'] == 4:
#             df.loc[rIndex, 'Retinopathy grade'] = 1
#     return df
#
#
# retina_df = pd.read_csv('/Users/shengbo/Documents/Github/dl-lab-2020-team06/IDRID_dataset/labels/train.csv')
# test_df = pd.read_csv('/Users/shengbo/Documents/Github/dl-lab-2020-team06/IDRID_dataset/labels/test.csv')
train_path = '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/images/train/'
test_path = '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/images/test/'
#
# # Make pandas DataFrame of the original training set
# # retina_df['ImageID'] = retina_df['Image name'].map(lambda x: x.split('_')[1])
# # retina_df['path'] = retina_df['Image name'].map(lambda x: os.path.join(train_path, '{}.jpg'.format(x)))
# # retina_df['exists'] = retina_df['path'].map(os.path.exists)
# # print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')
#
# # Add some more details
# # retina_df['level_cat'] = retina_df['Retinopathy grade'].map(lambda x: to_categorical(x, 1+retina_df['Retinopathy grade'].max()))
# retina_df.dropna(inplace=True, axis='columns')
# # retina_df = retina_df[retina_df['exists']]
#
# # Check if there is redundant data
# raw_train_df = retina_df[['Image name', 'Retinopathy grade']].drop_duplicates()
#
# # Check the distribution of the original sets
# print('train', raw_train_df.shape[0])
# raw_train_df[['Retinopathy grade']].hist(figsize=(10, 5))
# plt.title('Retinopathy grade of raw train set')
# plt.show()
# test_df[['Retinopathy grade']].hist(figsize=(10, 5))
# plt.title('Retinopathy grade of raw test set')
# plt.show()
#
# # Balance the distribution in the new sets
# train_df = raw_train_df.groupby(['Retinopathy grade']).apply(lambda x: x.sample(150, replace=True)).reset_index(
#     drop=True)
# train_df[['Retinopathy grade']].hist(figsize=(10, 5))
# plt.title('Retinopathy grade of new train set')
# plt.show()
# test_df[['Retinopathy grade']].hist(figsize=(10, 5))
# plt.title('Retinopathy grade of new test set')
# plt.show()
# print(raw_train_df)
# print('New Data Size:', raw_train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
#
# # two classes
# train_df = five2two(raw_train_df)
# test_df = five2two(test_df)
#
# # five classes
# # train_df = raw_train_df
# # test_df = test_df
#
# train_df[['Retinopathy grade']].hist(figsize=(10, 5))
# plt.title('Retinopathy grade of new train set')
# plt.show()
# test_df[['Retinopathy grade']].hist(figsize=(10, 5))
# plt.title('Retinopathy grade of new test set')
# plt.show()
#
# # Shuffle
# train_df = shuffle(train_df)
# test_df = shuffle(test_df)
#
# # Write new training set and test set to csv files in the same directory
# #
# train_df.to_csv(
#     '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/labels/balanced-train-2.csv',
#     index=0, columns=['Image name', 'Retinopathy grade'])
# test_df.to_csv(
#     '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/labels/balanced-test-2.csv', index=0,
#     columns=['Image name', 'Retinopathy grade'])


# Convert csv file to dict(key-value pairs each row)
def row_csv2dict(csv_file):
    dict_club = {}
    with open(csv_file)as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            dict_club[row[0]] = row[1]
    return dict_club


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


def image_example(image_string, label):
    """
      Creates a tf.train.Example message ready to be written to a file.
      """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    image_shape = tf.image.decode_jpeg(image_string).shape
    feature = {
        'image': _bytes_feature(image_string),
        'label': _int64_feature(label),
        'img_height': _int64_feature(image_shape[0]),
        'img_width': _int64_feature(image_shape[1]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# Write messages in TFRecord
def write(record_file, image_labels, path):
    with tf.io.TFRecordWriter(record_file) as writer:
        for filename, label in image_labels.items():
            if filename != "Image name":
                print("path" + path)
                print("filename" + filename + "!")
                print("label" + label)
                filename = path + filename + ".jpg"
                image_string = open(filename, 'rb').read()
                tf_example = image_example(image_string, int(label))
                writer.write(tf_example.SerializeToString())
    writer.close()


train2_file = 'idrid-2balanced-train.tfrecord-00000-of-00001'
test2_file = 'idrid-2balanced-test.tfrecord-00000-of-00001'
train5_file = 'idrid-5balanced-train.tfrecord-00000-of-00001'
test5_file = 'idrid-5balanced-test.tfrecord-00000-of-00001'

train_image_labels = row_csv2dict(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/labels/balanced-train-5.csv')
print(train_image_labels)
write(train5_file, train_image_labels, train_path)

test_image_labels = row_csv2dict(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/labels/balanced-test-5.csv')
print(test_image_labels)
write(test5_file, test_image_labels, test_path)

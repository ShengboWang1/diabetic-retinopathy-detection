import csv 
import tensorflow as tf
# import IPython.display as display
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



base_image_dir = '/Users/shengbo/Documents/Github/dl-lab-2020-team06/IDRID_dataset/images/train/'
retina_df = pd.read_csv('/Users/shengbo/Documents/Github/dl-lab-2020-team06/IDRID_dataset/labels/train.csv')
test_df = pd.read_csv('/Users/shengbo/Documents/Github/dl-lab-2020-team06/IDRID_dataset/labels/test.csv')

# Make pandas DataFrame of the original training set
retina_df['ImageID'] = retina_df['Image name'].map(lambda x: x.split('_')[1])
retina_df['path'] = retina_df['Image name'].map(lambda x: os.path.join(base_image_dir,'{}.jpg'.format(x)))
retina_df['exists'] = retina_df['path'].map(os.path.exists)
print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')

# Add some more details
retina_df['level_cat'] = retina_df['Retinopathy grade'].map(lambda x: to_categorical(x, 1+retina_df['Retinopathy grade'].max()))
retina_df.dropna(inplace=True, axis='columns')
retina_df = retina_df[retina_df['exists']]

# Check the distribution of the training set
retina_df[['Retinopathy grade']].hist(figsize=(10, 5))
plt.title('Retinopathy grade of original train set')
# plt.show()

# Check if there is redundant data
rr_df = retina_df[['ImageID', 'Retinopathy grade']].drop_duplicates()

# Split Data into Training and Validation
train_ids, valid_ids = train_test_split(rr_df['ImageID'],
                                   test_size=0.20,
                                   random_state=2020,
                                   stratify=rr_df['Retinopathy grade'])

raw_train_df = retina_df[retina_df['ImageID'].isin(train_ids)]

valid_df = retina_df[retina_df['ImageID'].isin(valid_ids)]

def create_5_classes_csv(raw_train_df, valid_df):
    # Check the distribution of the training set after split
    print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])
    raw_train_df[['Retinopathy grade']].hist(figsize=(10, 5))
    plt.title('Retinopathy grade of raw train set')
    plt.show()

    # Check the distribution of the validation set
    valid_df[['Retinopathy grade']].hist(figsize=(10, 5))
    print(valid_df)
    plt.title('Retinopathy grade of val set')
    # plt.show()

    # Balance the distribution in the training set
    train_df = raw_train_df.groupby(['Retinopathy grade']).apply(lambda x: x.sample(120, replace=True)).reset_index(
        drop=True)
    print(train_df)
    print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
    train_df = shuffle(train_df)
    train_df[['Retinopathy grade']].hist(figsize=(10, 5))
    plt.title('Retinopathy grade of new new train set')
    plt.show()

    # Write new training set and validation set to csv files in the same directory
    train_df.to_csv(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/labels/new_train.csv',
        index=0, columns=['Image name', 'Retinopathy grade', 'path', 'level_cat'])
    valid_df.to_csv(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/labels/val.csv', index=0,
        columns=['Image name', 'Retinopathy grade', 'path', 'level_cat'])

#  change the retinopathy grade from 5 grades to 2 grades
def five2two(df):
    step = 0
    for rIndex in df.index:
        if df.loc[rIndex, 'Retinopathy grade'] == 0 or df.loc[rIndex, 'Retinopathy grade'] == 1:
            df.loc[rIndex, 'Retinopathy grade'] = 0
        elif df.loc[rIndex, 'Retinopathy grade'] == 2 or df.loc[rIndex, 'Retinopathy grade'] == 3 \
                or df.loc[rIndex, 'Retinopathy grade'] == 4:
            df.loc[rIndex, 'Retinopathy grade'] = 1
            step += 1
    print(step)
    return df

def create_2_classes_csv(raw_train_df, valid_df, test_df):
    raw_train_df = five2two(raw_train_df)
    valid_df = five2two(valid_df)
    test_df = five2two(test_df)

    # Check the distribution of the training set after split
    print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])
    raw_train_df[['Retinopathy grade']].hist(figsize=(10, 5))
    plt.title('Retinopathy grade of raw train set')
    plt.show()
    # Check the distribution of the validation set
    valid_df[['Retinopathy grade']].hist(figsize=(10, 5))
    print(valid_df)
    plt.title('Retinopathy grade of val set')
    # plt.show()

    # Balance the distribution in the training set
    train_df = raw_train_df.groupby(['Retinopathy grade']).apply(lambda x: x.sample(207, replace=True)).reset_index(
        drop=True)
    print(train_df)
    print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
    train_df = shuffle(train_df)
    train_df[['Retinopathy grade']].hist(figsize=(10, 5))
    plt.title('Retinopathy grade of new new train set')
    plt.show()

    # Write new training set , validation set and test set to csv files in the same directory

    train_df.to_csv(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/labels/new_train_2classes.csv',
        index=0, columns=['Image name', 'Retinopathy grade', 'path', 'level_cat'])

    valid_df.to_csv(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/labels/val_2classes.csv', index=0,
        columns=['Image name', 'Retinopathy grade', 'path', 'level_cat'])

    test_df.to_csv(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/labels/test_2classes.csv',
        index=0,
        columns=['Image name', 'Retinopathy grade'])


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
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Create a dictionary with features that may be relevant.
def image_example(image_string, label):
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

# Some arguments
train_file = 'idrid-train.tfrecord-00000-of-00001'
val_file = 'idrid-val.tfrecord-00000-of-00001'
test_file = 'idrid-test.tfrecord-00000-of-00001'

train_path = '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/images/train/'
test_path = '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/images/test/'

train2_file = 'idrid-2train.tfrecord-00000-of-00001'
val2_file = 'idrid-2val.tfrecord-00000-of-00001'
test2_file = 'idrid-2test.tfrecord-00000-of-00001'


# Write TFRecords files for train and test dataset
def create_train_record():
    train_image_labels = row_csv2dict(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/labels/new_train.csv')
    # print(train_image_labels)
    write(train_file, train_image_labels, train_path)


def create_val_record():
    train_image_labels = row_csv2dict(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/labels/val.csv')
    # print(train_image_labels)
    write(val_file, train_image_labels, train_path)


def create_test_record():
    test_image_labels = row_csv2dict(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/labels/test.csv')
    # print(test_image_labels)
    write(test_file, test_image_labels, test_path)

def create_train2_record():
    train_image_labels = row_csv2dict(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/labels/new_train_2classes.csv')
    # print(train_image_labels)
    write(train2_file, train_image_labels, train_path)

def create_val2_record():
    train_image_labels = row_csv2dict(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/labels/val_2classes.csv')
    # print(train_image_labels)
    write(val2_file, train_image_labels, train_path)

def create_test2_record():
    test_image_labels = row_csv2dict(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/IDRID_dataset/labels/test_2classes.csv')
    #print(test_image_labels)
    write(test2_file, test_image_labels, test_path)

create_5_classes_csv(raw_train_df, valid_df)
create_2_classes_csv(raw_train_df, valid_df, test_df)

create_train_record()
create_val_record()
create_test_record()

create_train2_record()
create_val2_record()
create_test2_record()


# Read the TFRecordDataset
def read():
    raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

    # Create a dictionary describing the features.
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'img_height': tf.io.FixedLenFeature([], tf.int64),
        'img_width': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    parsed_image_dataset

    for image_features in parsed_image_dataset:
        image_raw = image_features['image_raw'].numpy()
        # display(IPython.display.Image(filename =  image_raw))
        display.display(display.Image(data=image_raw))

def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

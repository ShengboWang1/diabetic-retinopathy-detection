import csv 
import tensorflow as tf
import IPython.display as display


# Convert csv file to dict(key-value pairs each row)
def row_csv2dict(csv_file, set):
    dict_club = {}
    with open(csv_file)as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            dict_club[row[0]] = row[1]
    l = int(len(dict_club) * 0.7)
    if set == 'test':
        return dict_club
    elif set == 'train':
        return dictcut(dict_club, 0, l)
    elif set == 'val':
        return dictcut(dict_club, l, -1)


def dictcut(dict, start, end):
    temp = list(dict.keys())
    result = {}
    temp = temp[start:end]
    for i in range(len(temp)):
        result[temp[i]] = dict.get(temp[i])
    return result

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
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
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

# some arguments
train_file = 'idrid-train.tfrecord-00000-of-00001'
val_file = 'idrid-val.tfrecord-00000-of-00001'
test_file = 'idrid-test.tfrecord-00000-of-00001'
train_path = '/Users/shengbo/Documents/Github/dl-lab-2020-team06/IDRID_dataset/images/train/'
test_path = '/Users/shengbo/Documents/Github/dl-lab-2020-team06/IDRID_dataset/images/test/'

# Write TFRecords files for train and test dataset
def create_train_record():
    train_image_labels = row_csv2dict(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/IDRID_dataset/labels/train.csv', set='train')
    print(train_image_labels)
    write(train_file, train_image_labels, train_path)

def create_val_record():
    train_image_labels = row_csv2dict(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/IDRID_dataset/labels/train.csv', set='val')
    print(train_image_labels)
    write(val_file, train_image_labels, train_path)

def create_test_record():
    test_image_labels = row_csv2dict(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/IDRID_dataset/labels/test.csv', set='test')
    print(test_image_labels)
    write(test_file, test_image_labels, test_path)

create_train_record()
create_val_record()
create_test_record()


# Read the TFRecordDataset
def read():
    raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

    # Create a dictionary describing the features.
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
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

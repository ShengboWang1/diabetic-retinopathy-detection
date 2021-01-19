import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


test_filename = '/Users/shengbo/Documents/Github/dl-lab-2020-team06/human_activity_recognition/' + "no0_test.tfrecord"

raw_ds_test = tf.data.TFRecordDataset(test_filename)


feature_description = {
            'feature': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }


def _parse_function(exam_proto):
    temp = tf.io.parse_single_example(exam_proto, feature_description)
    feature = tf.io.parse_tensor(temp['feature'], out_type=tf.float64)
    label = tf.io.parse_tensor(temp['label'], out_type=tf.int64)
    return (feature, label)


ds_test = raw_ds_test.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# multi_lstm.n_lstm = 3
# multi_lstm.n_dense = 3
# multi_lstm.rnn_units = 256
# multi_lstm.dense_units = 256
# multi_lstm.dropout_rate = 0.4209
# multi_lstm.window_size = 120
from models.multi_lstm import multi_lstm
def plot(dataset, model):
    for feature, label in dataset:
        predictions = model(feature, training=False)
        label_preds = np.argmax(predictions, -1)
        print(label_preds.shape)


model = multi_lstm(rnn_type='GRU',dense_units=256, dropout_rate=0.4209, window_size=250, rnn_units = 256, n_lstm = 3, n_dense = 3)
plot(ds_test, model)



def plot_data(dataset, model):
    i = 0
    # dataset.unbatch()
    for feature, label in dataset:
        i += 1
        if i == 1:
            acc_x_component = feature.numpy()[:, 0]
            acc_y_component = feature.numpy()[:, 1]
            acc_z_component = feature.numpy()[:, 2]
            gyro_x_component = feature.numpy()[:, 0]
            gyro_y_component = feature.numpy()[:, 1]
            gyro_z_component = feature.numpy()[:, 2]
            labels = label.numpy()
        else:
            acc_x_component = np.append(acc_x_component, feature.numpy()[:, 0])
            acc_y_component = np.append(acc_y_component, feature.numpy()[:, 1])
            acc_z_component = np.append(acc_z_component, feature.numpy()[:, 2])
            gyro_x_component = np.append(gyro_x_component, feature.numpy()[:, 3])
            gyro_y_component = np.append(gyro_y_component, feature.numpy()[:, 4])
            gyro_z_component = np.append(gyro_z_component, feature.numpy()[:, 5])
            labels = np.append(labels, label)




    # chosing colors : red for X component blue for Y component and green for Z component
    len_ds = len(acc_x_component)  # number of rows in this dataframe to be visualized(depends on 'act' variable)

    # converting row numbers into time duration (the duration between two rows is 1/50=0.02 second)
    time = [0.02 * j for j in range(len_ds)]


    acc_legend_x = 'acc_X'
    acc_legend_y = 'acc_Y'
    acc_legend_z = 'acc_Z'
    gyro_legend_x = 'gyro_X'
    gyro_legend_y = 'gyro_Y'
    gyro_legend_z = 'gyro_Z'

    color_dict = {0: 'white', 1: 'red', 2: 'orange', 3: 'yellow', 4: 'green', 5: 'pink', 6: 'brown', 7: 'violet',
                  8: 'lightgreen', 9: 'cyan', 10: 'darkblue', 11: 'tan', 12: 'cyan'}
    # Define the figure and setting dimensions width and height
    plt.figure(figsize=(20, 4))

    # j = 0
    # for label in labels:
    #     color_value = color_dict.get(label)
    #     plt.axvspan(0.02 * j, 0.02 * (j + 1), facecolor=color_value, alpha=0.5)
    #     j += 1

    color_values = []
    for label in labels:
        color_value = color_dict.get(label)
        color_values = np.append(color_values, color_value)

    for k in range(len_ds):
        plt.axvspan(0.02 * k, 0.02 * (k + 1), facecolor=color_values[k], alpha=0.5)

    plt.plot(time, acc_x_component, color='r', label=acc_legend_x)
    plt.plot(time, acc_y_component, color='b', label=acc_legend_y)
    plt.plot(time, acc_z_component, color='g', label=acc_legend_z)

    # Set the figure info defined earlier
    plt.ylabel('Acceleration in 1g')  # set Y axis info
    plt.xlabel('Time in seconds (s)')  # Set X axis info (same label in all cases)
    plt.title("acceleration signals")  # Set the title of the figure

    # localise the figure's legends
    plt.legend(loc="upper left")  # upper left corner

    # showing the figure
    plt.show()

    # ploting each signal component
    plt.figure(figsize=(20, 4))
    for k in range(len_ds):
        plt.axvspan(0.02 * k, 0.02 * (k + 1), facecolor=color_values[k], alpha=0.5)
    plt.plot(time, gyro_x_component, color='r', label=gyro_legend_x)
    plt.plot(time, gyro_y_component, color='b', label=gyro_legend_y)
    plt.plot(time, gyro_z_component, color='g', label=gyro_legend_z)

    # Set the figure info defined earlier
    plt.ylabel('Angular Velocity in radian per second [rad/s]')  # set Y axis info
    plt.xlabel('Time in seconds (s)')  # Set X axis info (same label in all cases)
    plt.title("gyroscope signals")  # Set the title of the figure

    # localise the figure's legends
    plt.legend(loc="upper left")  # upper left corner

    # showing the figure
    plt.show()

#plot_data(ds_test)


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os


def plot(len_ds, values, x, y, z, legend_x, legend_y, legend_z, title, run_paths):


    # converting row numbers into time duration (the duration between two rows is 1/50=0.02 second)
    time = [0.02 * j for j in range(len_ds)]

    plt.figure(figsize=(20, 4))
    for k in range(len_ds):
        plt.axvspan(0.02 * k, 0.02 * (k + 1), facecolor=values[k], alpha=0.5)
        #plt.axvspan(k, k + 1, facecolor=values[k], alpha=0.5)

    plt.plot(time, x, color='r', label=legend_x)
    plt.plot(time, y, color='b', label=legend_y)
    plt.plot(time, z, color='g', label=legend_z)

    # Set the figure info defined earlier
    plt.ylabel('Acceleration in 1g')  # set Y axis info
    plt.xlabel('Time in seconds (s)')  # Set X axis info (same label in all cases)
    plt.title(title)  # Set the title of the figure

    # localise the figure's legends
    plt.legend(loc="upper left")  # upper left corner
    plot_path = os.path.join(run_paths['path_plt'], title + ' visualization.png')
    plt.figure(figsize=(20, 4))
    plt.savefig(plot_path)
    # showing the figure
    plt.show()


def plot_data(model, dataset, run_paths):

    checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model)
    checkpoint.restore(tf.train.latest_checkpoint(
        '/Users/shengbo/Documents/Github/dl-lab-2020-team06/experiments/run_2021-01-17T20-18-52-373438/ckpts/'))

    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['SparseCategoricalAccuracy'])

    i = 0
    # dataset.unbatch()
    for feature, label in dataset:
        # print(feature.numpy().shape)
        # print(label.numpy().shape)
        prediction = model(feature, training=False)
        # print(prediction.numpy().shape)
        label_pred = np.argmax(prediction, -1)
        # print(label_pred.shape)
        i += 1
        if i == 1:
            acc_x_component = feature.numpy()[:, :, 0].flatten()
            print(acc_x_component.shape)
            acc_y_component = feature.numpy()[:, :, 1].flatten()
            acc_z_component = feature.numpy()[:, :, 2].flatten()
            gyro_x_component = feature.numpy()[:, :, 3].flatten()
            gyro_y_component = feature.numpy()[:, :, 4].flatten()
            gyro_z_component = feature.numpy()[:, :, 5].flatten()
            labels = label.numpy().flatten()
            label_preds = label_pred.flatten()
        elif i < 10:
            acc_x_component = np.append(acc_x_component, feature.numpy()[:, :, 0].flatten())
            print(acc_x_component.shape)
            acc_y_component = np.append(acc_y_component, feature.numpy()[:, :, 1].flatten())
            acc_z_component = np.append(acc_z_component, feature.numpy()[:, :, 2].flatten())
            gyro_x_component = np.append(gyro_x_component, feature.numpy()[:, :, 3].flatten())
            gyro_y_component = np.append(gyro_y_component, feature.numpy()[:, :, 4].flatten())
            gyro_z_component = np.append(gyro_z_component, feature.numpy()[:, :, 5].flatten())
            labels = np.append(labels, label.numpy().flatten())
            label_preds = np.append(label_preds, label_pred.flatten())
            print(labels.shape)
            print(labels)
            print(label_preds.shape)

    len_ds = len(label_preds)  # number of rows in this dataframe to be visualized(depends on 'act' variable)
    print(len_ds)


    #
    # acc_legend_x = 'acc_X'
    # acc_legend_y = 'acc_Y'
    # acc_legend_z = 'acc_Z'
    # gyro_legend_x = 'gyro_X'
    # gyro_legend_y = 'gyro_Y'
    # gyro_legend_z = 'gyro_Z'

    color_dict = {0: 'cyan', 1: 'red', 2: 'orange', 3: 'yellow', 4: 'green', 5: 'pink', 6: 'brown', 7: 'violet',
                  8: 'lightgreen', 9: 'cyan', 10: 'darkblue', 11: 'tan', 12: 'white'}

    true_color_values = []
    pred_color_values = []

    for label in labels:
        #print(label)
        true_color_value = color_dict.get(label)
        true_color_values = np.append(true_color_values, true_color_value)
    print(true_color_values.shape)

    for label in label_preds:
        pred_color_value = color_dict.get(label)
        pred_color_values = np.append(pred_color_values, pred_color_value)
    print(pred_color_values.shape)

    # Define the figure and setting dimensions width and height
    # plt.subplot(4, 1, 1)
    plot(len_ds, values=true_color_values, x=acc_x_component, y=acc_y_component, z=acc_z_component,
         legend_x='acc_X', legend_y='acc_Y', legend_z='acc_Z', title="acc signals with true labels",
         run_paths=run_paths)

    # plt.subplot(4, 1, 2)
    plot(len_ds, values=pred_color_values, x=acc_x_component, y=acc_y_component, z=acc_z_component,
         legend_x='acc_X', legend_y='acc_Y', legend_z='acc_Z', title="acc signals with predictions",
         run_paths=run_paths)

    # plt.subplot(4, 1, 3)
    plot(len_ds, values=true_color_values, x=gyro_x_component, y=gyro_y_component, z=gyro_z_component,
         legend_x='gyro_X', legend_y='gyro_Y', legend_z='gyro_Z', title="gyro signals with true labels",
         run_paths=run_paths)

    # plt.subplot(4, 1, 4)
    plot(len_ds, values=pred_color_values, x=gyro_x_component, y=gyro_y_component, z=gyro_z_component,
         legend_x='gyro_X', legend_y='gyro_Y', legend_z='gyro_Z', title="gyro signals with predictions",
         run_paths=run_paths)

#plot_data(ds_test)

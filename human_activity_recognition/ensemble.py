import tensorflow as tf
from evaluation.metrics import ConfusionMatrixMetric
from input_pipeline import datasets
from sklearn.metrics import accuracy_score
import numpy as np
import gin
from utils import utils_params, utils_misc
import logging
from models.multi_rnn import multi_rnn
import seaborn as sns
import matplotlib.pyplot as plt
from input_pipeline import plot_data


def ensemble_averaging(ds_test, models, num_models):
    """ Combine models by averaging

    Parameters:
        ds_test (dataset): tensorflow dataset object
        models (tuple: 3 keras.model): the models models that we use for ensemble
        num_models (int): number of models that we use for ensemble

    Returns:
        accuracy_score(float): accuracy of this ensemble model
        test_cm(keras.metric): confusion matrix
        total_labels(np array): true labels of the dataset
        total_pred_labels(np array): predictions of ensemble model
    """
    test_cm = ConfusionMatrixMetric(num_classes=12)

    i = 0
    for test_features, test_labels in ds_test:
        average_predictions = 0
        for model in models:
            test_predictions = model(test_features, training=False)
            average_predictions += test_predictions
            # print(average_predictions.numpy().shape)
        average_predictions = average_predictions / num_models
        # print(average_predictions.numpy().shape())
        average_labels = np.argmax(average_predictions, -1)
        average_labels = average_labels.flatten()

        _ = test_cm.update_state(test_labels.numpy().flatten(), average_labels)

        i += 1
        if i == 1:
            total_labels = test_labels.numpy().flatten()
            total_pred_labels = average_labels
        else:
            total_labels = np.append(total_labels, test_labels.numpy().flatten())
            total_pred_labels = np.append(total_pred_labels, average_labels)

    return accuracy_score(total_labels, total_pred_labels), test_cm, total_labels, total_pred_labels


# generate folder structures
run_paths = utils_params.gen_run_folder()

# set loggers
utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

# gin-config
gin.parse_config_files_and_bindings(['/content/drive/MyDrive/human_activity_recognition/configs/config.gin'],
                                    [])
utils_params.save_config(run_paths['path_gin'], gin.config_str())

# setup pipeline
ds_train, ds_val, ds_test, window_size = datasets.load(device_name='Colab')

model1 = multi_rnn(rnn_type='GRU', window_size=window_size, kernel_initializer='glorot_uniform')
model1.build(input_shape=(None, 256, 256, 3))
checkpoint1 = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model1)
checkpoint1.restore(
    tf.train.latest_checkpoint('/content/drive/MyDrive/experiments/run_2021-02-04T17-48-05-968689/ckpts'))
model1.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')

model2 = multi_rnn(rnn_type='GRU', window_size=window_size, kernel_initializer='he_normal')
model2.build(input_shape=(None, 256, 256, 3))
checkpoint2 = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model2)
checkpoint2.restore(
    tf.train.latest_checkpoint('/content/drive/MyDrive/experiments/run_2021-02-04T18-46-09-416019/ckpts'))
model2.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')

model3 = multi_rnn(rnn_type='GRU', window_size=window_size, kernel_initializer='glorot_normal')
checkpoint3 = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model3)
checkpoint3.restore(
    tf.train.latest_checkpoint('/content/drive/MyDrive/experiments/run_2021-02-04T18-59-10-306075/ckpts'))
model3.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')

models = [model1, model2, model3]
model1.summary()

acc, test_cm, total_labels, total_pred_labels = ensemble_averaging(ds_test, models, num_models=3)

print(acc)

template = 'Confusion Matrix:\n{}'
logging.info(template.format(test_cm.result().numpy()))
template = 'Sparese Categorical Accuracy:{}'
logging.info(template.format(acc))

print(test_cm.result().numpy())
plt.figure(figsize=(18, 16))
plt.savefig('/content/drive/MyDrive/human_activity_recognition/cm.png')
hm = sns.heatmap(test_cm.result().numpy(), annot=True, fmt='g')
plt.show()

print(total_pred_labels)


def plot_colormap():
    # Display colormap
    plt.figure(figsize=(20, 4))
    activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING',
                       'LAYING', 'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE',
                       'LIE_TO_STAND']
    # color_dict = {0: 'lightpink', 1: 'lightgreen', 2: 'orange', 3: 'yellow', 4: 'cyan', 5: 'greenyellow',
    #               6: 'red', 7: 'violet',
    #               8: 'sandybrown', 9: 'lightskyblue', 10: 'blueviolet', 11: 'deepskyblue', 12: 'white'}
    colors = ['lightpink', 'lightgreen', 'orange', 'yellow', 'cyan', 'greenyellow', 'red',
              'violet', 'sandybrown', 'lightskyblue', 'blueviolet', 'deepskyblue']
    x = np.arange(0, 12, 1)
    plt.bar(x, height=1, width=1, align='center', color=colors)
    plt.xticks(x, activity_labels, rotation=60)
    plt.yticks([])
    plt.title('Colormap')
    plt.margins(0)
    ax = plt.gca()
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    # change figure size
    fig = plt.gcf()
    fig.set_size_inches(8, 4)
    plt.tight_layout()
    plot_path = os.path.join(run_paths['path_plt'], ' colormap.png')
    plt.savefig(plot_path)
    plt.show()


def plot(len_ds, values, x, y, z, legend_x, legend_y, legend_z, title, run_paths):
    # converting row numbers into time duration (the duration between two rows is 1/50=0.02 second)
    # time = [0.02 * j for j in range(len_ds)]

    plt.figure(figsize=(20, 4))
    for k in range(len_ds):
        # plt.axvspan(0.02 * k, 0.02 * (k + 1), facecolor=values[k], alpha=0.5)
        plt.axvspan(k, k + 1, facecolor=values[k], alpha=0.5)

    plt.plot(x, color='r', label=legend_x)
    plt.plot(y, color='b', label=legend_y)
    plt.plot(z, color='g', label=legend_z)

    plt.title(title)  # Set the title of the figure

    # localise the figure's legends
    plt.legend(loc="upper left")  # upper left corner
    plot_path = os.path.join(run_paths['path_plt'], title + ' visualization.png')
    # plt.figure(figsize=(20, 4))
    plt.savefig(plot_path)
    # showing the figure
    # plt.show()


def plot_data(dataset, run_paths, total_labels, total_pred_labels):

    i = 0
    # dataset.unbatch()
    for feature, label in dataset:

        i += 1
        if i == 1:
            acc_x_component = feature.numpy()[:, :, 0].flatten()
            print(acc_x_component.shape)
            acc_y_component = feature.numpy()[:, :, 1].flatten()
            acc_z_component = feature.numpy()[:, :, 2].flatten()
            gyro_x_component = feature.numpy()[:, :, 3].flatten()
            gyro_y_component = feature.numpy()[:, :, 4].flatten()
            gyro_z_component = feature.numpy()[:, :, 5].flatten()
            # labels = label.numpy().flatten()
            # label_preds = label_pred.flatten()
        elif i < 2:
            acc_x_component = np.append(acc_x_component, feature.numpy()[:, :, 0].flatten())
            print(acc_x_component.shape)
            acc_y_component = np.append(acc_y_component, feature.numpy()[:, :, 1].flatten())
            acc_z_component = np.append(acc_z_component, feature.numpy()[:, :, 2].flatten())
            gyro_x_component = np.append(gyro_x_component, feature.numpy()[:, :, 3].flatten())
            gyro_y_component = np.append(gyro_y_component, feature.numpy()[:, :, 4].flatten())
            gyro_z_component = np.append(gyro_z_component, feature.numpy()[:, :, 5].flatten())

    len_ds = len(acc_x_component)  # number of rows in this dataframe to be visualized(depends on 'act' variable)
    print(len_ds)
    print(acc_x_component)

    color_dict = {0: 'lightpink', 1: 'lightgreen', 2: 'orange', 3: 'yellow', 4: 'cyan', 5: 'greenyellow',
                  6: 'red', 7: 'violet',
                  8: 'sandybrown', 9: 'lightskyblue', 10: 'blueviolet', 11: 'deepskyblue', 12: 'white'}

    true_color_values = []
    pred_color_values = []

    k = 0
    for label in total_labels:
        k += 1
        # print(label)
        if k <= len_ds:
            true_color_value = color_dict.get(label)
            true_color_values = np.append(true_color_values, true_color_value)
    print(true_color_values.shape)

    k = 0
    for label in total_pred_labels:
        k += 1
        if k <= len_ds:
            pred_color_value = color_dict.get(label)
            pred_color_values = np.append(pred_color_values, pred_color_value)
    print(pred_color_values)

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

    plot_colormap()


plot_data(ds_test, run_paths, total_labels, total_pred_labels)

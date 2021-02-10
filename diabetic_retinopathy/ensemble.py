import tensorflow as tf
from models.resnet_1 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from evaluation.metrics import ConfusionMatrixMetric
from models.architectures import vgg_like
from input_pipeline import datasets
from sklearn.metrics import accuracy_score
import numpy as np
import gin
from utils import utils_params, utils_misc
import logging
from evaluation.eval import plot_roc


def ensemble_averaging(ds_test, models, num_models):
    """ Combine models by averaging the predictions

    Parameters:
        ds_test (dataset): tensorflow dataset object
        models (tuple: 3 keras.model): the models models that we use for ensemble
        num_models (int): number of models that we use for ensemble

    Returns:
        accuracy_score(float): accuracy of this ensemble model
        test_cm(keras.metric): confusion matrix
        test_labels(np array): true labels of the dataset
        ensembled_predictions(Tensor): predictions of ensemble model
    """

    test_cm = ConfusionMatrixMetric(num_classes=2)
    for test_images, test_labels in ds_test:
        total_pred_predictions = 0
        for model in models:
            predictions = model(test_images, training=False)
            # label_pred = np.argmax(predictions, -1)
            # print("predictions")
            # print(predictions)
            total_pred_predictions += predictions
        ensembled_predictions = total_pred_predictions / num_models
        # print("ensembled_predictions")
        # print(ensembled_predictions)
        label_pred = np.argmax(ensembled_predictions, -1)
        _ = test_cm.update_state(test_labels, ensembled_predictions)

    return accuracy_score(test_labels, label_pred), test_cm, test_labels, ensembled_predictions


def ensemble_voting(ds_test, models, num_models):
    """ Combine models by voting

    Parameters:
        ds_test (dataset): tensorflow dataset object
        models (tuple: 3 keras.model): the models models that we use for ensemble
        num_models (int): number of models that we use for ensemble

    Returns:
        accuracy_score(float): accuracy of this ensemble model
        test_cm(keras.metric): confusion matrix
        test_labels(np array): true labels of the dataset
        ensembled_predictions(Tensor): predictions of ensemble model
    """
    test_cm = ConfusionMatrixMetric(num_classes=2)
    for test_images, test_labels in ds_test:
        total_pred_labels = 0
        for model in models:
            predictions = model(test_images, training=False)
            # label_pred = np.argmax(predictions, -1)
            # print("predictions")
            # print(predictions)
            label_pred = np.argmax(predictions, -1)
            total_pred_labels += label_pred
        # only binary problem suitable

        ensembled_labels = total_pred_labels > (num_models / 2)
        ensembled_labels = ensembled_labels.astype(int)
        one_hot_predictions = np.eye(2)[ensembled_labels]
        _ = test_cm.update_state(test_labels, one_hot_predictions)

        print("ensembled_labels")
        print(ensembled_labels)

    return accuracy_score(test_labels, ensembled_labels), test_cm, test_labels, one_hot_predictions


# generate folder structures
run_paths = utils_params.gen_run_folder()

# set loggers
utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

# gin-config
gin.parse_config_files_and_bindings(['/content/drive/MyDrive/diabetic_retinopathy/configs/config.gin'],
                                    [])
utils_params.save_config(run_paths['path_gin'], gin.config_str())

# setup pipeline
ds_train, ds_val, ds_test, ds_info = datasets.load(device_name='Colab', dataset_name='idrid', n_classes=2)

model1 = ResNet18(problem_type='classification', num_classes=2)
model1.build(input_shape=(None, 256, 256, 3))
checkpoint1 = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model1)
checkpoint1.restore(tf.train.latest_checkpoint('/content/drive/MyDrive/experiments/run_2021-01-26T13-16-28-380039/ckpts'))
model1.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')

model2 = ResNet34(problem_type='classification', num_classes=2)
model2.build(input_shape=(None, 256, 256, 3))
checkpoint2 = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model2)
checkpoint2.restore(tf.train.latest_checkpoint('/content/drive/MyDrive/experiments/run_2021-01-25T18-54-18-524245/ckpts'))
model2.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')


model3 = vgg_like(input_shape=(256, 256, 3), n_classes=2)
checkpoint3 = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model3)
checkpoint3.restore(tf.train.latest_checkpoint('/content/drive/MyDrive/experiments/run_2021-01-26T18-46-27-336202/ckpts'))
model3.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')

models = [model1, model2, model3]

acc, cm, test_labels, predictions = ensemble_voting(ds_test, models, num_models=3)

# Show accuracy
template = 'accuracy{}'
logging.info(template.format(acc))

# Confusion matrix
template = 'Confusion Matrix:\n{}'
logging.info(template.format(cm.result().numpy()))

# Compute some other metrics from the confusion matrix
sensitivity, specificity, precision, f1 = cm.process_confusion_matrix()
template = 'Sensitivity: {}, Specificity: {}, Precision: {}, F1: {}'
logging.info(template.format(sensitivity, specificity, precision, f1))
print(cm.result().numpy())
print("sensitivity, specificity")
print(sensitivity)
print(specificity)
print(precision)
print(f1.numpy())

# Plot roc and compute auc
plot_path = '/content/drive/MyDrive/diabetic_retinopathy/'
plot_roc(labels=test_labels, predictions=predictions[:, 1], plot_path=plot_path)

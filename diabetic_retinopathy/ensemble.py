import tensorflow as tf
from models.resnet_1 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from evaluation.metrics import ConfusionMatrixMetric
from models.architectures import vgg_like
from input_pipeline import datasets
from sklearn.metrics import accuracy_score
import numpy as np
import gin
from absl import app, flags
from utils import utils_params, utils_misc
import logging


def ensemble_averaging(ds_test, models, num_models):
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

    return accuracy_score(test_labels, label_pred), test_cm


def ensemble_voting(ds_test, models, num_models):
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

    return accuracy_score(test_labels, ensembled_labels), test_cm

# generate folder structures
run_paths = utils_params.gen_run_folder()

# set loggers
utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

# gin-config
gin.parse_config_files_and_bindings(['/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/configs/config.gin'],
                                    [])
utils_params.save_config(run_paths['path_gin'], gin.config_str())

# setup pipeline
ds_train, ds_val, ds_test, ds_info = datasets.load(device_name='local', dataset_name='idrid')

model1 = ResNet18(problem_type='classification', num_classes=2)
model1.build(input_shape=(None, 256, 256, 3))
checkpoint1 = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model1)
checkpoint1.restore(tf.train.latest_checkpoint('/Users/shengbo/Documents/Github/dl-lab-2020-team06/experiments/run_2021-01-23T19-34-36-063044/ckpts'))
model1.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')

model2 = ResNet34(problem_type='classification', num_classes=2)
model2.build(input_shape=(None, 256, 256, 3))
checkpoint2 = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model2)
checkpoint2.restore(tf.train.latest_checkpoint('/Users/shengbo/Documents/Github/dl-lab-2020-team06/experiments/run_2021-01-25T11-18-32-536508/ckpts'))
model2.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')


model3 = ResNet34(problem_type='classification', num_classes=2)
model3.build(input_shape=(None, 256, 256, 3))
checkpoint3 = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model3)
checkpoint3.restore(tf.train.latest_checkpoint('/Users/shengbo/Documents/Github/dl-lab-2020-team06/experiments/run_2021-01-25T18-54-18-524245/ckpts'))
model3.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')

models = [model1, model2, model3]

acc, test_cm = ensemble_voting(ds_test, models, num_models=2)

print(acc)

template = 'Confusion Matrix:\n{}'
logging.info(template.format(test_cm.result().numpy()))

# Compute sensitivity and specificity from the confusion matrix
sensitivity, specificity = test_cm.process_confusion_matrix()
template = 'Sensitivity: {}, Specificity: {}'
logging.info(template.format(sensitivity, specificity))
print(test_cm.result().numpy())
print("sensitivity, specificity")
print(sensitivity)
print(specificity)


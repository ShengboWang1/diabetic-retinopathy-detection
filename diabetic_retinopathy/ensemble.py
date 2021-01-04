import tensorflow as tf
from models.resnet import resnet18, resnet34, resnet50, resnet50_original
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
            print("predictions")
            print(predictions)
            total_pred_predictions += predictions
        ensembled_predictions = total_pred_predictions / num_models
        print("ensembled_predictions")
        print(ensembled_predictions)
        label_pred = np.argmax(ensembled_predictions, -1)
        # divided_number = (num_models+1)/2
        # ensembled_label = total_pred_label // divided_number
        # print("ensembled_predictions")
        # print(ensembled_predictions)
    return accuracy_score(test_labels, label_pred)

def ensemble_voting(ds_test, models, num_models):
    for test_images, test_labels in ds_test:
        total_pred_labels = 0
        for model in models:
            predictions = model(test_images, training=False)
            # label_pred = np.argmax(predictions, -1)
            print("predictions")
            print(predictions)
            label_pred = np.argmax(predictions, -1)
            total_pred_labels += label_pred
        ensembled_labels = total_pred_labels / num_models
        print("ensembled_labels")
        print(ensembled_labels)

    return accuracy_score(test_labels, ensembled_labels)

# generate folder structures
run_paths = utils_params.gen_run_folder()

# set loggers
utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

# gin-config
gin.parse_config_files_and_bindings(['/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/configs/config.gin'],
                                    [])
utils_params.save_config(run_paths['path_gin'], gin.config_str())

# setup pipeline
ds_train, ds_val, ds_test, ds_info = datasets.load()

model1 = vgg_like(input_shape=[256, 256, 3], n_classes=2)
model1.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')
model2 = resnet18()
model2.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')
model3 = resnet34()
model3.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')

models = [model1, model2, model3]

acc = ensemble_voting(ds_test, models, num_models=3)

print(acc)


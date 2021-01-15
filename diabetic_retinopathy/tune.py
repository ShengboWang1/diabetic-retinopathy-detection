import logging
import gin
from ray import tune
from input_pipeline.datasets import load
from models.architectures import vgg_like
from train import Trainer
from utils import utils_params, utils_misc
from models.resnet_1 import ResNet18
from models.inception_resnet_v2 import inception_resnet_v2
from models.architectures import vgg_like
from models.densenet import densenet121_bigger, densenet121
from models.inception_v3 import inception_v3
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('device_name', 'local', 'Prepare different paths for local, iss GPU and Colab')
flags.DEFINE_string('dataset_name', 'idrid', 'Specify whether to use idrid or eyepacs')

def train_func(config):
    # Hyperparameters
    bindings = []
    for key, value in config.items():
        bindings.append(f'{key}={value}')

    # generate folder structures
    run_paths = utils_params.gen_run_folder(','.join(bindings))

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    if FLAGS.device_name == 'local':
        gin.parse_config_files_and_bindings(
            ['/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/configs/config.gin'], bindings)
    elif FLAGS.device_name == 'iss GPU':
        gin.parse_config_files_and_bindings(
            ['/home/RUS_CIP/st169852/st169852/dl-lab-2020-team06/diabetic_retinopathy/configs/config.gin'], bindings)
    elif FLAGS.device_name == 'Colab':
        gin.parse_config_files_and_bindings(['/content/drive/MyDrive/diabetic_retinopathy/configs/config.gin'], bindings)
    else:
        raise ValueError

    # gin.parse_config_files_and_bindings(['/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/configs/config.gin'], bindings)
    # gin.parse_config_files_and_bindings(['/home/RUS_CIP/st169852/st169852/dl-lab-2020-team06/diabetic_retinopathy/configs/config.gin'], bindings)
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load(device_name=FLAGS.device_name, dataset_name=FLAGS.dataset_name)

    # model
    # model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)
    model = vgg_like(input_shape=(256, 256, 3), n_classes=2)
    # model = densenet121()

    trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy)


config = {
        "Trainer.total_steps": tune.grid_search([20000]),
        "vgg_like.base_filters": tune.choice([8, 16]),
        "vgg_like.n_blocks": tune.choice([2, 3, 4, 5]),
        "vgg_like.dense_units": tune.choice([32, 64]),
        "vgg_like.dropout_rate": tune.uniform(0, 0.9),
    }

if FLAGS.device_name == 'local':
    resources_per_trial = {'gpu': 0, 'cpu': 1}
elif FLAGS.device_name == 'iss GPU':
    resources_per_trial = {'gpu': 1, 'cpu': 10}
elif FLAGS.device_name == 'Colab':
    resources_per_trial = {'gpu': 1, 'cpu': 2}
else:
    raise ValueError
analysis = tune.run(
    train_func, num_samples=50, resources_per_trial=resources_per_trial,
    config=config
    )

print("Best config: ", analysis.get_best_config(metric="val_accuracy", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()

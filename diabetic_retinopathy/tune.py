import logging
import gin
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from input_pipeline.datasets import load
from models.architectures import vgg_like
from train import Trainer
from utils import utils_params, utils_misc
from models.resnet_1 import ResNet18
from models.inception_resnet_v2 import inception_resnet_v2
from models.architectures import vgg_like
from models.densenet import densenet121_bigger, densenet121
from models.inception_v3 import inception_v3

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
    # gin.parse_config_files_and_bindings(['/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/configs/config.gin'], bindings)
    gin.parse_config_files_and_bindings(['./configs/config.gin'], bindings)
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load()

    # model
    # model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)
    # model = vgg_like(input_shape=(256, 256, 3), n_classes=2)
    model = densenet121()

    trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy)


hyperopt = HyperOptSearch(metric="val_accuracy", mode="max")
asha_scheduler = ASHAScheduler(metric="val_accuracy", mode="max", grace_period=5, max_t=100)
config={
        "Trainer.total_steps": tune.grid_search([10000]),
        "densenet121.layer_index": tune.choice([100, 200, 300, 400]),
        "densenet121.dense_units": tune.choice([8, 16, 32, 64, 128, 256]),
        "densenet121.dropout_rate": tune.uniform(0, 0.5),
    }

analysis = tune.run(
    train_func, num_samples=30, resources_per_trial={'gpu': 0, 'cpu': 1},
    config=config
    )

print("Best config: ", analysis.get_best_config(metric="val_accuracy", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()

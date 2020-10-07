import os
import gin
from ray import tune

from input_pipeline.datasets import load
from models.architectures import vgg_like
from train import Trainer
from utils import utils_params


def train_func(config):
    # Hyperparameters
    bindings = []
    for key, value in config.items():
        bindings.append(f'{key} = {value}')

    # generate folder structures
    run_paths = utils_params.gen_run_folder()
    print(run_paths)

    # gin-config
    gin.parse_config_files_and_bindings(['/mnt/home/repos/dl-lab-skeleton/diabetic_retinopathy/configs/config.gin'], bindings)

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load()

    # model
    model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)

    trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy)


analysis = tune.run(
    train_func, num_samples=20, resources_per_trial={'gpu': 1, 'cpu': 4},
    config={
        "Trainer.total_steps": tune.grid_search([1e7]),
        "vgg_like.base_filters": tune.choice([8, 16]),
        "vgg_like.n_blocks": tune.choice([2, 3, 4, 5]),
        "vgg_like.dense_units": tune.choice([32, 64]),
        "vgg_like.dropout_rate": tune.uniform(0, 1),
    })

print("Best config: ", analysis.get_best_config(metric="val_accuracy"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()

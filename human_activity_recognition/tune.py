import logging
import gin
from ray import tune
from input_pipeline.datasets import load
from train import Trainer
from utils import utils_params, utils_misc
from models.multi_lstm import multi_lstm
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('device_name', 'local', 'Prepare different paths for local, iss GPU and Colab')


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
            ['/Users/shengbo/Documents/Github/dl-lab-2020-team06/human_activity_recognition/configs/config.gin'], bindings)
    elif FLAGS.device_name == 'iss GPU':
        gin.parse_config_files_and_bindings(
            ['/home/RUS_CIP/st169852/st169852/dl-lab-2020-team06/human_activity_recognition/configs/config.gin'], bindings)
    elif FLAGS.device_name == 'Colab':
        gin.parse_config_files_and_bindings(['/content/drive/MyDrive/human_activity_recognition/configs/config.gin'], bindings)
    else:
        raise ValueError

    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load()

    # model
    model = multi_lstm()


    trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy)

config={
        "Trainer.total_steps": tune.grid_search([50000]),
        "multi_lstm.dense_units": tune.choice([16, 32, 64, 128, 256]),
        "multi_lstm.n_lstm": tune.choice([1, 2, 3]),
        "multi_lstm.n_dense": tune.choice([1, 2, 3]),
        "multi_lstm.lstm_units": tune.choice([16, 32, 64, 128, 256]),
        "multi_lstm.dropout_rate": tune.uniform(0.1, 0.8),
    }

analysis = tune.run(
    train_func, num_samples=100, resources_per_trial={'gpu': 1, 'cpu': 10},
    config=config
    )

print("Best config: ", analysis.get_best_config(metric="val_accuracy", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()

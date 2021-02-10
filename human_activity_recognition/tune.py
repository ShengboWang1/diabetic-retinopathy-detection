import logging
import gin
from ray import tune
from input_pipeline.datasets import load
from train import Trainer
from utils import utils_params, utils_misc
from models.multi_rnn import multi_rnn
from absl import app, flags


device_name = 'iss GPU'


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
    if device_name == 'local':
        gin.parse_config_files_and_bindings(
            ['/Users/shengbo/Documents/Github/dl-lab-2020-team06/human_activity_recognition/configs/config.gin'], bindings)
    elif device_name == 'iss GPU':
        gin.parse_config_files_and_bindings(
            ['/home/RUS_CIP/st169852/st169852/dl-lab-2020-team06/human_activity_recognition/configs/config.gin'], bindings)
    elif device_name == 'Colab':
        gin.parse_config_files_and_bindings(['/content/drive/MyDrive/human_activity_recognition/configs/config.gin'], bindings)
    else:
        raise ValueError

    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, window_size = load(device_name=device_name)

    # model
    model = multi_rnn(rnn_type='GRU', window_size=window_size)

    trainer = Trainer(model, ds_train, ds_val, run_paths)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy)


config = {
        "Trainer.total_steps": tune.grid_search([5000]),
        #"create_tfr.window_size": tune.choice([250, 200, 150, 100]),
        #"create_tfr.shift_window_size": tune.choice([125, 100, 75, 50]),
        #"multi_rnn.dense_units": tune.choice([512]),
        #"multi_rnn.n_lstm": tune.choice([1, 2, 3]),
        #"multi_rnn.n_dense": tune.choice([1, 2, 3]),
        #"multi_rnn.rnn_units": tune.choice([16, 32, 64, 128, 256, 512]),
        #"multi_rnn.dropout_rate": tune.uniform(0.1, 0.8),
        "multi_rnn.kernel_initializer": tune.choice(["glorot_uniform", "glorot_normal", "random_uniform", "he_uniform", "he_normal", "lecun_normal", "lecun_uniform", "TruncatedNormal"])
    }

if device_name == 'local':
    resources_per_trial = {'gpu': 0, 'cpu': 1}
elif device_name == 'iss GPU':
    resources_per_trial = {'gpu': 1, 'cpu': 48}
elif device_name == 'Colab':
    resources_per_trial = {'gpu': 1, 'cpu': 2}
else:
    raise ValueError

# resources_per_trial = {'gpu': 1, 'cpu': 2}

analysis = tune.run(
    train_func, num_samples=30, resources_per_trial=resources_per_trial,
    config=config
    )

print("Best config: ", analysis.get_best_config(metric="val_accuracy", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()

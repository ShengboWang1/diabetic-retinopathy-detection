import gin
import logging
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.simple_rnn import simple_rnn
from models.cnn_lstm import cnn_lstm
from models.multi_rnn import multi_rnn


FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('device_name', 'local', 'Prepare different paths for local, iss GPU and Colab')
flags.DEFINE_string('model_name', 'multi_rnn', 'Prepare different models')
flags.DEFINE_string('kernel_initializer', 'he_normal', 'try different kernel initializers for ensemble learning')


@gin.configurable
def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    if FLAGS.device_name == 'local':
        gin.parse_config_files_and_bindings(['/Users/shengbo/Documents/Github/dl-lab-2020-team06/human_activity_recognition/configs/config.gin'], [])
    elif FLAGS.device_name == 'iss GPU':
        gin.parse_config_files_and_bindings(['/home/RUS_CIP/st169852/st169852/dl-lab-2020-team06/human_activity_recognition/configs/config.gin'], [])
    elif FLAGS.device_name == 'Colab':
        gin.parse_config_files_and_bindings(['/content/drive/MyDrive/human_activity_recognition/configs/config.gin'], [])
    else:
        raise ValueError

    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, window_size = datasets.load(device_name=FLAGS.device_name)

    # select a suitable model
    if FLAGS.model_name == 'simple_rnn':
        model = simple_rnn(rnn_type='GRU', window_size=window_size, kernel_initializer=FLAGS.kernel_initializer)
    elif FLAGS.model_name == 'cnn_lstm':
        model = cnn_lstm(window_size=window_size, kernel_initializer=FLAGS.kernel_initializer)
    elif FLAGS.model_name == 'multi_rnn':
        model = multi_rnn(rnn_type='GRU', window_size=window_size, kernel_initializer=FLAGS.kernel_initializer)
    else:
        raise ValueError

    # train or evaluate
    if FLAGS.train:
        model.summary()
        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue

    else:
        evaluate(model,
                 ds_test,
                 run_paths)


if __name__ == "__main__":
    app.run(main)

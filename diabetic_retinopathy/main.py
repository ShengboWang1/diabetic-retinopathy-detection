import gin
import logging
from absl import app, flags
from train import Trainer
from evaluation.eval import Evaluator
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.resnet import resnet18
from models.resnet import resnet34
from models.resnet import resnet50
import tensorflow as tf
# from models.architectures import vgg_like
# from tensorflow.keras.applications.resnet import ResNet50

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')


def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # model vgg
    # model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)
    # model = vgg_like(input_shape=[256, 256, 3], n_classes=5)

    # model resnet
    model = resnet34()
    model.build(input_shape=(32, 256, 256, 3))
    model.summary()

    evaluator = Evaluator(model, ds_test, ds_info, run_paths)
    evaluator.evaluate()
    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue
        evaluator = Evaluator(model, ds_test, ds_info, run_paths)
        evaluator.evaluate()


    else:
        evaluator = Evaluator(model, ds_test, ds_info, run_paths)
        evaluator.evaluate()


if __name__ == "__main__":
    app.run(main)

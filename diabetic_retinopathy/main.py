import gin
import logging
import tensorflow as tf
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.resnet import resnet18, resnet34, resnet50, resnet50_original
from models.inception_resnet_v2 import inception_resnet_v2
from models.architectures import vgg_like
from models.densenet import densenet121
from models.inception_v3 import inception_v3

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')


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
    # model = vgg_like(input_shape=[256, 256, 3], n_classes=2)

    # model resnet
    # model = resnet18()
    # model = resnet34()
    # model = resnet34()
    # model = inception_v3(num_classes=2)
    model = densenet121(num_classes=2)
    # model = inception_resnet_v2(2)
    # model.build(input_shape=(16, 256, 256, 3))


    if FLAGS.train:
        model.summary()
        trainer = Trainer(model, ds_train, ds_test, ds_info, run_paths)
        for _ in trainer.train():
            continue
        model_to_be_restored = densenet121(num_classes=2)
        checkpoint = tf.train.Checkpoint(myModel=model_to_be_restored)
        evaluate(model_to_be_restored,
                 checkpoint,
                 ds_test,
                 ds_info,
                 run_paths)
    else:
        model_to_be_restored = densenet121(num_classes=5)
        checkpoint = tf.train.Checkpoint(myModel=model_to_be_restored)
        evaluate(model_to_be_restored,
                 checkpoint,
                 ds_test,
                 ds_info,
                 run_paths)


if __name__ == "__main__":
    app.run(main)

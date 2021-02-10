import gin
import logging
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.resnet import resnet18, resnet34, resnet50, resnet50_original
from models.resnet_1 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.transfer_models import inception_resnet_v2, inception_v3, mobilenet
from models.architectures import vgg_like
from models.densenet import densenet121, densenet121_bigger

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('model_name', 'resnet18', 'Name of the model')
flags.DEFINE_string('device_name', 'local', 'Prepare different paths for local, iss GPU and Colab')
flags.DEFINE_string('problem_type', 'classification', 'Specify whether to solve a regression or a classification problem')
flags.DEFINE_string('dataset_name', 'idrid', 'Specify whether to use idrid or eyepacs')
flags.DEFINE_integer('num_classes', 5, '2 or 5 classes classification problem')


@gin.configurable
def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    if FLAGS.device_name == 'local':
        gin.parse_config_files_and_bindings(['/Users/shengbo/Documents/Github/dl-lab-2020-team06/diabetic_retinopathy/configs/config.gin'], [])
    elif FLAGS.device_name == 'iss GPU':
        gin.parse_config_files_and_bindings(['/home/RUS_CIP/st169852/st169852/dl-lab-2020-team06/diabetic_retinopathy/configs/config.gin'], [])
    elif FLAGS.device_name == 'Colab':
        gin.parse_config_files_and_bindings(['/content/drive/MyDrive/diabetic_retinopathy/configs/config.gin'], [])
    else:
        raise ValueError
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load(device_name=FLAGS.device_name, dataset_name=FLAGS.dataset_name, n_classes=FLAGS.num_classes)

    # select model as you wish
    if FLAGS.model_name == 'vgg':
        model = vgg_like(input_shape=(256, 256, 3), n_classes=FLAGS.num_classes)
        # model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)

    elif FLAGS.model_name == 'resnet18':
        model = ResNet18(problem_type=FLAGS.problem_type, num_classes=FLAGS.num_classes)
        model.build(input_shape=(None, 256, 256, 3))

    elif FLAGS.model_name == 'resnet34':
        model = ResNet34(problem_type=FLAGS.problem_type, num_classes=FLAGS.num_classes)
        model.build(input_shape=(None, 256, 256, 3))

    elif FLAGS.model_name == 'resnet50':
        model = ResNet50(problem_type=FLAGS.problem_type, num_classes=FLAGS.num_classes)
        model.build(input_shape=(None, 256, 256, 3))

    elif FLAGS.model_name == 'resnet101':
        model = ResNet101(problem_type=FLAGS.problem_type, num_classes=FLAGS.num_classes)
        model.build(input_shape=(None, 256, 256, 3))

    elif FLAGS.model_name == 'resnet152':
        model = ResNet152(problem_type=FLAGS.problem_type, num_classes=FLAGS.num_classes)
        model.build(input_shape=(None, 256, 256, 3))

    elif FLAGS.model_name == 'densenet121':
        model = densenet121(FLAGS.num_classes, problem_type=FLAGS.problem_type)

    elif FLAGS.model_name == 'densenet121_bigger':
        model = densenet121_bigger()

    elif FLAGS.model_name == 'inception_resnet_v2':
        model = inception_resnet_v2()

    elif FLAGS.model_name == 'inception_v3':
        model = inception_v3()

    elif FLAGS.model_name == 'mobilenet':
        model = mobilenet()

    else:
        raise ValueError

    model.summary()
    logging.info(f"A {FLAGS.problem_type} Problem")
    if FLAGS.problem_type == 'classification':
        logging.info(f"{FLAGS.num_classes} classes")

    if FLAGS.train:
        logging.info(f"Training model {FLAGS.model_name}...")
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, problem_type=FLAGS.problem_type)
        for _ in trainer.train():
            continue

        evaluate(model,
                 ds_test,
                 ds_info,
                 FLAGS.num_classes,
                 run_paths)
    else:
        logging.info(f"Evaluate model {FLAGS.model_name}...")
        evaluate(model,
                 ds_test,
                 ds_info,
                 FLAGS.num_classes,
                 run_paths)


if __name__ == "__main__":
    app.run(main)

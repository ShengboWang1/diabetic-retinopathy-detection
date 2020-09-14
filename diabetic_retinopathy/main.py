import gin
from absl import logging

from train import train
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params
from models.architectures import vgg_like

@gin.configurable
def main(mode):

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    logging.get_absl_handler().use_absl_log_file(run_paths['path_logs_train'])
    logging.set_verbosity(logging.INFO)

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load('mnist')
    logging.info(ds_info)

    # model
    model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)
    logging.info(model.summary())

    if mode == 'train':
        train(model,
              ds_train,
              ds_val,
              ds_info,
              run_paths)
    elif mode == 'eval':
        evaluate()
    else:
        raise ValueError

if __name__ == "__main__":
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    main()
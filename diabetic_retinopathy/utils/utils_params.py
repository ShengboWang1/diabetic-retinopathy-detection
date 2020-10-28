import os
import datetime


def gen_run_folder(path_model_id=''):
    run_paths = dict()

    if not os.path.isdir(path_model_id):
        path_model_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'experiments'))
        date_creation = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')
        run_id = 'run_' + date_creation
        if path_model_id:
            run_id += '_' + path_model_id
        run_paths['path_model_id'] = os.path.join(path_model_root, run_id)
    else:
        run_paths['path_model_id'] = path_model_id

    run_paths['path_logs_train'] = os.path.join(run_paths['path_model_id'], 'logs', 'run.log')
    #run_paths['path_logs_eval'] = os.path.join(run_paths['path_model_id'], 'logs', 'eval', 'run.log')
    run_paths['path_ckpts_train'] = os.path.join(run_paths['path_model_id'], 'ckpts')
    #run_paths['path_ckpts_eval'] = os.path.join(run_paths['path_model_id'], 'ckpts', 'eval')
    run_paths['path_gin'] = os.path.join(run_paths['path_model_id'], 'config_operative.gin')

    # Create folders
    for k, v in run_paths.items():
        if any([x in k for x in ['path_model', 'path_ckpts']]):
            if not os.path.exists(v):
                os.makedirs(v, exist_ok=True)

    # Create files
    for k, v in run_paths.items():
        if any([x in k for x in ['path_logs']]):
            if not os.path.exists(v):
                os.makedirs(os.path.dirname(v), exist_ok=True)
                with open(v, 'a'):
                    pass  # atm file creation is sufficient

    return run_paths


def save_config(path_gin, config):
    with open(path_gin, 'w') as f_config:
        f_config.write(config)

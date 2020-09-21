import os
import datetime

def gen_run_folder(path_model_id=''):
    run_paths = dict()

    print(path_model_id)

    if not os.path.isdir(path_model_id):
        path_model_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'experiments'))
        date_creation = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        run_id = 'run_' + date_creation
        run_paths = dict()
        run_paths['path_model_id'] = os.path.join(path_model_root, run_id)
    else:
        run_paths['path_model_id'] = path_model_id
    print('------------------------')
    print(run_paths['path_model_id'])
    print('------------------------')
    run_paths['path_logs_train'] = os.path.join(run_paths['path_model_id'], 'logs')
    #run_paths['path_logs_eval'] = os.path.join(run_paths['path_model_id'], 'logs', 'eval', 'run.log')
    run_paths['path_ckpts_train'] = os.path.join(run_paths['path_model_id'], 'ckpts')
    #run_paths['path_ckpts_eval'] = os.path.join(run_paths['path_model_id'], 'ckpts', 'eval')
    run_paths['path_gin'] = os.path.join(run_paths['path_model_id'], 'config_operative.gin')

    # Create folders
    for k, v in run_paths.items():
        if any([x in k for x in ['path_model', 'path_logs', 'path_ckpts']]):
            if not os.path.exists(v):
                os.makedirs(v, exist_ok=True)

    return run_paths

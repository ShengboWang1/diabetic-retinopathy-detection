# Team06
- Shengbo Wang (st169852)
- Junmao Liao (st165800)

# How to run the code
## Diabetic retinopathy recognition
Train ResNet18 with idrid dataset:

`python3 main.py --train=True  --model_name='resnet18' --num_classes=2 --problem_type='classification' --device_name='iss GPU' --dataset_name='idrid'`

Evaluate the performance:

`python3 main.py --train=False  --model_name='resnet18' --num_classes=2 --problem_type='classification' --device_name='iss GPU' --dataset_name='idrid'`

Try another model:

`python3 main.py --train=True  --model_name='vgg' --num_classes=2 --problem_type='classification' --device_name='iss GPU' --dataset_name='idrid'`

Ensemble learning and evaluate:

`python3 ensemble.py`

Create TFRecord files:

`python3 inputpipeline/create_tfrecord.py`

5-classes classification:

`python3 main.py --train=True  --model_name='resnet18' --num_classes=5 --problem_type='classification' --device_name='iss GPU' --dataset_name='idrid'`

Regression:

`python3 main.py --train=True  --model_name='resnet18' --num_classes=5 --problem_type='regression' --device_name='iss GPU' --dataset_name='idrid'`

Eyepacs:

`python3 main.py --train=True  --model_name='resnet18' --num_classes=5 --problem_type='regression' --device_name='iss GPU' --dataset_name='eyepacs'`


## Human activity recognition
Train model:

`python3 main.py --train=True  --model_name='multi-rnn' --device_name='iss GPU' --kernel_initializer='he_normal'`

Evaluate model:

`python3 main.py --train=False  --model_name='multi-rnn' --device_name='iss GPU' --kernel_initializer='he_normal'`

Ensemble learning, evaluate and visualization:

`python3 ensemble.py`

# Results
## Diabetic retinopathy recognition






## Human activity recognition

Hyperparameter optimization:

| Trial | GRU layers | Desne layers | GRU units | Dense units | Window size | Shift size | Dropout rate | Val accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2 | 3 | 512 | 256 | 250 | 125 | 0.471 | 92.9% |
| 2 | 2 | 3 | 128 | 64 | 250 | 125 | 0.387 | 90.8% |
| 3 | 1 | 2 | 32 | 128 | 250 | 75 | 0.566 | 85.1% |
| 4 | 1 | 1 | 256 | 128 | 100 | 50 | 0.454 | 85.8% |
| 5 | 1 | 1 | 256 | 128 | 250 | 125 | 0.248 | 88.4% |




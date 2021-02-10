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

Ensemble learning:
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
Ensemble learning:
`python3 ensemble.py`

# Results
to do

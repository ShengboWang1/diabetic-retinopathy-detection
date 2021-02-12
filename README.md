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
Please see details in paper poster and presentation slides.
## Diabetic retinopathy recognition

Results of different models for binary classification:
The best accuracy on test set could be 89.32%(Ensemble model)

|  | Accuracy | Sensitivity/Recall | Specificity | Precision | F1 score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| VGG | 88.34% | 0.875 | 0.897 | 0.933 | 0.903 |
| ResNet18 | 86.41% | 0.828 | 0.9231 | 0.946 | 0.883 |
| ResNet34 | 86.41% | 0.797 | 0.974 | 0.981 | 0.879 |
| MobileNet | 73.44% | 0.821 | 0.821 | 0.870 | 0.797 |
| InceptionV3 | 80.58% | 0.875 | 0.692 | 0.824 | 0.849 |
| InceptionResNetV2 | 80.58% | 0.797 | 0.821 | 0.879 | 0.836 |
| DenseNet | 82.52% | 0.833 | 0.814 | 0.862 | 0.847 |
| Voting | 89.32% | 0.844 | 0.974 | 0.982 | 0.908 |
| Averaging | 89.32% | 0.844 | 0.974 | 0.982 | 0.908 |

Deep visualization:
![deep_visual](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team06/blob/master/results/01_dr/03_deep%20visualization.JPG)


Confusion matrix of multi-class classification(resnet18):
![con_mat](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team06/blob/master/results/01_dr/04_confusion%20matrix.JPG)




## Human activity recognition
Results of ensemble model with He_normal Glorot_uniform and Glorot_normal initializers:
| Architecture | LSTM | GRU |
| :---: | :---: | :---: |
| Test Accuracy | 91.73% | 94.48% | 

Corresponding Hyperparameters for the best GRU-based model:

| GRU layers | Dense layers | GRU units | Dense units | Window size | Shift size | Dropout rate | Val accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 3 | 512 | 256 | 250 | 125 | 0.471 | 92.9% |


Visualization of test label and predictions:

![01_acc_true](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team06/blob/master/results/02_har/01_acc%20signals%20with%20true%20labels%20visualization.png)
![02_acc_pred](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team06/blob/master/results/02_har/02_acc%20signals%20with%20predictions%20visualization.png)
![03_gyro_true](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team06/blob/master/results/02_har/03_gyro%20signals%20with%20true%20labels%20visualization.png)
![04_gyro_pred](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team06/blob/master/results/02_har/04_gyro%20signals%20with%20predictions%20visualization.png)
![05_colormap](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team06/blob/master/results/02_har/05_colormap.png)


Confusion Matrix:
![cm_6](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team06/blob/master/results/02_har/cm_6.png)


Normalised confusion matrix:
![normal_cm_6](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team06/blob/master/results/02_har/normal_cm_6.png)



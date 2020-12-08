import tensorflow as tf
import logging
import gin
from evaluation.metrics import ConfusionMatrix
from train import Trainer

def evaluate(model, checkpoint, ds_test, ds_info, run_paths):
    checkpoint.restore(tf.train.latest_checkpoint('./checkpoint/train'))
    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')
    for test_images, test_labels in ds_test:
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
    template = 'Test Loss: {}, Test Accuracy: {}'
    print(template.format(test_loss, test_accuracy * 100))
    return test_accuracy


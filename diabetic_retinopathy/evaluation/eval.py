import tensorflow as tf
from evaluation.metrics import ConfusionMatrixMetric
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def evaluate(model, checkpoint, ds_train, ds_val, ds_test, ds_info, run_paths):
    test_CM = ConfusionMatrixMetric(2)
    val_CM = ConfusionMatrixMetric(2)
    # Restore the model from the corresponding checkpoint
    # checkpoint.restore(tf.train.latest_checkpoint('./checkpoint/checkpoint/train/20201217-033207'))
    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['SparseCategoricalAccuracy'])


    for val_images, val_labels in ds_val:
        val_loss, val_accuracy = model.evaluate(val_images, val_labels, verbose=1)
        predictions = model(val_images, training=True)
        label_pred = np.argmax(predictions, -1)
        _ = val_CM.update_state(val_labels, predictions)
    template = 'val Loss: {}, val Accuracy: {}'
    print(template.format(val_loss, val_accuracy * 100))
    template = 'Confusion Matrix:\n{}'
    print(template.format(val_CM.result().numpy()))


    for test_images, test_labels in ds_test:
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
        predictions = model(test_images, training=True)
        label_pred = np.argmax(predictions, -1)
        _ = test_CM.update_state(test_labels, predictions)

    template = 'Test Loss: {}, Test Accuracy: {}'
    print(template.format(test_loss, test_accuracy * 100))
    template = 'Confusion Matrix:\n{}'
    print(template.format(test_CM.result().numpy()))
    return test_accuracy



import tensorflow as tf
import logging
from evaluation.metrics import ConfusionMatrixMetric
import matplotlib.pyplot as plt
import sklearn
import os
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn import metrics


def evaluate(model, ds_test, ds_info, run_paths):
    test_cm = ConfusionMatrixMetric(num_classes=2)

    # Restore the model from the corresponding checkpoint
    checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model)
    checkpoint.restore(
        tf.train.latest_checkpoint("/content/drive/MyDrive/experiments/run_2021-01-01T15-51-31-698506/ckpts/"))

    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['SparseCategoricalAccuracy'])
    plot_path = os.path.join(run_paths['path_plt'], 'roc.png')
    print(plot_path)

    # Compute accuracy and loss for test set and the corresponding confusion matrix
    for test_images, test_labels in ds_test:
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
        test_predictions = model(test_images, training=False)
        label_preds = np.argmax(test_predictions, -1)
        _ = test_cm.update_state(test_labels, test_predictions)

        # label_preds = np.argmax(predictions, -1)
        labels = test_labels.numpy()
        binary_true = np.squeeze(labels)
        binary_pred = np.squeeze(label_preds)

        binary_accuracy = metrics.accuracy_score(binary_true, binary_pred)
        binary_confusion_matrix = metrics.confusion_matrix(binary_true, binary_pred)
        tf.print(binary_accuracy)
        tf.print(binary_confusion_matrix)
        plot_roc(labels=test_labels, predictions=test_predictions[:, 1], plot_path=plot_path)

    print('Accuracy on Test Data: %2.2f%%' % (accuracy_score(test_labels, label_preds)))
    print(classification_report(test_labels, label_preds))

    template = 'Test Loss: {}, Test Accuracy: {}'
    logging.info(template.format(test_loss, test_accuracy * 100))

    template = 'Confusion Matrix:\n{}'
    logging.info(template.format(test_cm.result().numpy()))

    # Compute sensitivity and specificity from the confusion matrix
    sensitivity, specificity = test_cm.process_confusion_matrix()
    template = 'Sensitivity: {}, Specificity: {}'
    logging.info(template.format(sensitivity, specificity))

    return
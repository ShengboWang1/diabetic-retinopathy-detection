import tensorflow as tf
import logging
from evaluation.metrics import ConfusionMatrixMetric
import matplotlib.pyplot as plt
import sklearn
import os
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn import metrics
from visualization import visualize


def evaluate(model, ds_test, ds_info, num_classes, run_paths):
    test_cm = ConfusionMatrixMetric(num_classes=num_classes)

    # Restore the model from the corresponding checkpoint

    #checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model)
    # checkpoint.restore(tf.train.latest_checkpoint(run_paths['path_ckpts_train']))

    #checkpoint.restore(tf.train.latest_checkpoint('/Users/shengbo/Documents/Github/dl-lab-2020-team06/experiments/run_2021-01-23T19-34-36-063044/ckpts'))

    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['SparseCategoricalAccuracy'])

    plot_path = os.path.join(run_paths['path_plt'], 'roc.png')
    print(plot_path)

    # Compute accuracy and loss for test set and the corresponding confusion matrix
    for test_images, test_labels in ds_test:
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
        test_predictions = model(test_images, training=False)
        label_preds = np.argmax(test_predictions, -1)
        _ = test_cm.update_state(test_labels, test_predictions)

        plot_roc(labels=test_labels, predictions=test_predictions[:, 1], plot_path=plot_path)

    # print('Accuracy on Test Data: %2.2f%%' % (accuracy_score(test_labels, label_preds)))
    print(classification_report(test_labels, label_preds))

    template = 'Test Loss: {}, Test Accuracy: {}'
    logging.info(template.format(test_loss, test_accuracy * 100))

    template = 'Confusion Matrix:\n{}'
    logging.info(template.format(test_cm.result().numpy()))

    # Compute sensitivity and specificity from the confusion matrix
    sensitivity, specificity = test_cm.process_confusion_matrix()
    template = 'Sensitivity: {}, Specificity: {}'
    logging.info(template.format(sensitivity, specificity))

    # visualize(model, layerName='sequential_6', save_path=run_paths)

    return


def plot_roc(labels, predictions, plot_path, **kwargs):
    """plot the ROC image of the corresponding checkpoint"""
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    area = sklearn.metrics.roc_auc_score(labels, predictions)
    # Plot fp and tp
    plt.plot(100 * fp, 100 * tp, label='ROC curve (area = %0.2f)' % area, linewidth=2, **kwargs)

    # Settings
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.legend(loc="lower right")
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

    # Save and show
    plt.savefig(plot_path)
    plt.show()


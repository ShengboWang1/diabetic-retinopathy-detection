import tensorflow as tf
from evaluation.metrics import ConfusionMatrixMetric
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import gin

@gin.configurable
def evaluate(model, checkpoint, ds_val, ds_test, ds_info, run_paths):
    test_cm = ConfusionMatrixMetric(num_classes=2)
    val_cm = ConfusionMatrixMetric(num_classes=2)
    # Restore the model from the corresponding checkpoint
    checkpoint.restore(tf.train.latest_checkpoint('./checkpoint/checkpoint/train/20201222-164014'))
    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['SparseCategoricalAccuracy'])

    #

    for val_images, val_labels in ds_val:
        val_loss, val_accuracy = model.evaluate(val_images, val_labels, verbose=1)
        val_predictions = model(val_images, training=True)
        label_pred = np.argmax(val_predictions, -1)
        _ = val_cm.update_state(val_labels, label_pred)

    template = 'val Loss: {}, val Accuracy: {}'
    print(template.format(val_loss, val_accuracy * 100))
    template = 'Confusion Matrix:\n{}'
    print(template.format(val_cm.result().numpy()))

    # Compute accuracy and loss for test set and the corresponding confusion matrix
    step = 0
    for test_images, test_labels in ds_test:
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
        test_predictions = model(test_images, training=True)
        # test_predictions = model.predict(test_images, batch_size=16)
        label_preds = np.argmax(test_predictions, -1)
        #_ = test_cm.update_state(test_labels, label_preds)
        if step == 0:
            all_test_labels = test_labels
            all_label_preds = label_preds
            all_test_predictions = test_predictions
        else:
            all_test_labels = np.concatenate((all_test_labels, test_labels), axis=0)
            all_label_preds = np.concatenate((all_label_preds, label_preds), axis=0)
            all_test_predictions = np.concatenate((all_test_predictions, test_predictions), axis=0)
        step += 1
    _ = test_cm.update_state(all_test_labels, all_test_predictions[:, 1])
    accuracy = np.sum(np.equal(all_label_preds, all_test_labels)) / all_test_labels.shape[0]
    print("new_acc")
    print(accuracy)
    # a = all_test_predictions[:, 1]
    plot_roc(labels=all_test_labels, predictions=all_test_predictions[:, 1])

    template = 'Test Loss: {}, Test Accuracy: {}'
    print(template.format(test_loss, test_accuracy * 100))

    template = 'Confusion Matrix:\n{}'
    print(template.format(test_cm.result().numpy()))

    # Compute sensitivity and specificity from the confusion matrix
    sensitivity, specificity = test_cm.process_confusion_matrix()
    template = 'Sensitivity: {}, Specificity: {}'
    print(template.format(sensitivity, specificity))

    return test_accuracy


def plot_roc(labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    area = sklearn.metrics.roc_auc_score(labels, predictions)
    plt.plot(100 * fp, 100 * tp, label='ROC curve (area = %0.2f)' % area, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.legend(loc="lower right")
    # plt.xlim([-0.5, 20])
    # plt.ylim([80, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()



import tensorflow as tf
import logging
from evaluation.metrics import ConfusionMatrixMetric
import matplotlib.pyplot as plt
import sklearn
import os
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
# #
# pred_Y = retina_model.predict(test_X, batch_size = 32, verbose = True)
# pred_Y_cat = np.argmax(pred_Y, -1)
# test_Y_cat = np.argmax(test_Y, -1)
# print('Accuracy on Test Data: %2.2f%%' % (accuracy_score(test_Y_cat, pred_Y_cat)))
# print(classification_report(test_Y_cat, pred_Y_cat))


def evaluate(model, checkpoint, ds_test, ds_info, run_paths):
    test_cm = ConfusionMatrixMetric(num_classes=2)

    # Restore the model from the corresponding checkpoint
    # checkpoint.restore(tf.train.latest_checkpoint(run_paths['path_ckpts_train']))

    # checkpoint.restore(tf.train.latest_checkpoint('./checkpoint/checkpoint/train/20201218-024936/'))
    # checkpoint.restore(tf.train.latest_checkpoint('Users/shengbo/Documents/Github/dl-lab-2020-team06/experiments/run_2020-12-30T04-30-55-717166/ckpts'))
    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['SparseCategoricalAccuracy'])
    plot_path = os.path.join(run_paths['path_plt'], 'roc.png')
    print(plot_path)

    # Compute accuracy and loss for test set and the corresponding confusion matrix
    for test_images, test_labels in ds_test:
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
        test_predictions = model(test_images, training=False)
        label_preds = np.argmax(test_predictions, -1)
        _ = test_cm.update_state(test_labels, test_predictions[:, 1])
        # plot_roc(labels=test_labels, predictions=test_predictions[:, 1], plot_path=plot_path)

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

    # #
    # val_cm = ConfusionMatrixMetric(num_classes=2)
    # for val_images, val_labels in ds_val:
    #     val_loss, val_accuracy = model.evaluate(val_images, val_labels, verbose=1)
    #     val_predictions = model(val_images, training=False)
    #     label_pred = np.argmax(val_predictions, -1)
    #     _ = val_cm.update_state(val_labels, label_pred)
    #
    # template = 'val Loss: {}, val Accuracy: {}'
    # logging.info(template.format(val_loss, val_accuracy * 100))
    # template = 'Confusion Matrix:\n{}'
    # logging.info(template.format(val_cm.result().numpy()))

    #
    # #
    # for test_images, test_labels in ds_test:
    #     test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
    #     test_predictions = model(test_images, training=True)
    #     # test_predictions = model.predict(test_images, batch_size=16)
    #     label_preds = np.argmax(test_predictions, -1)
    #     #_ = test_cm.update_state(test_labels, label_preds)
    #     #
    #     if step == 0:
    #         all_test_labels = test_labels
    #         all_label_preds = label_preds
    #         all_test_predictions = test_predictions
    #     else:
    #         all_test_labels = np.concatenate((all_test_labels, test_labels), axis=0)
    #         all_label_preds = np.concatenate((all_label_preds, label_preds), axis=0)
    #         all_test_predictions = np.concatenate((all_test_predictions, test_predictions), axis=0)
    #     step += 1
    # _ = test_cm.update_state(all_test_labels, all_test_predictions[:, 1])
    # accuracy = np.sum(np.equal(all_label_preds, all_test_labels)) / all_test_labels.shape[0]
    # print("new_acc")
    # print(accuracy)
    # # a = all_test_predictions[:, 1]

import tensorflow as tf
import logging
from evaluation.metrics import ConfusionMatrixMetric
import matplotlib.pyplot as plt
import sklearn
import os
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix


def evaluate(model, ds_test, run_paths):
    test_cm = ConfusionMatrixMetric(num_classes=12)

    # Restore the model from the corresponding checkpoint

    checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model)
    # checkpoint.restore(tf.train.latest_checkpoint(run_paths['path_ckpts_train']))

    checkpoint.restore(tf.train.latest_checkpoint('/Users/shengbo/Documents/Github/dl-lab-2020-team06/experiments/run_2021-01-17T20-18-52-373438/ckpts/'))

    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['SparseCategoricalAccuracy'])
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')
    # plot_path = os.path.join(run_paths['path_plt'], 'roc.png')
    # print(plot_path)

    # Compute accuracy and loss for test set and the corresponding confusion matrix
    for test_features, test_labels in ds_test:
        t_loss, t_accuracy = model.evaluate(test_features, test_labels, verbose=1)
        print("predictions")

        test_predictions = model(test_features, training=False)
        print(test_predictions.numpy().flatten().shape)
        print(test_labels.shape)
        label_preds = np.argmax(test_predictions, -1)
        _ = test_cm.update_state(test_labels.numpy().flatten(), label_preds.flatten())
        test_loss(t_loss)
        test_accuracy(t_accuracy)
        sns.set()
        sns.heatmap(confusion_matrix(test_labels.numpy().flatten(), label_preds.flatten()),
                    annot=True, fmt="d", cbar=False, cmap=plt.cm.Blues)

        # plot_roc(labels=test_labels, predictions=test_predictions[:, 1], plot_path=plot_path)

    # print('Accuracy on Test Data: %2.2f%%' % (accuracy_score(test_labels, label_preds)))
    # print(classification_report(test_labels, label_preds))

    template = 'Test Loss: {}, Test Accuracy: {}'
    logging.info(template.format(test_loss.result(), test_accuracy.result() * 100))

    template = 'Confusion Matrix:\n{}'
    logging.info(template.format(test_cm.result().numpy()))

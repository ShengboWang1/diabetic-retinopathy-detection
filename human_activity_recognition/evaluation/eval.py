import tensorflow as tf
import logging
from evaluation.metrics import ConfusionMatrixMetric
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import seaborn as sns


def evaluate(model, ds_test, run_paths):
    test_cm = ConfusionMatrixMetric(num_classes=12)

    # Restore the model from the corresponding checkpoint
    checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model)
    # checkpoint.restore(tf.train.latest_checkpoint(run_paths['path_ckpts_train']))
    # he_normal
    # checkpoint.restore(tf.train.latest_checkpoint('/content/drive/MyDrive/experiments/run_2021-02-04T18-59-10-306075/ckpts/'))
    checkpoint.restore(tf.train.latest_checkpoint('/content/drive/MyDrive/experiments/run_2021-02-06T16-38-25-745037/ckpts/'))
    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['SparseCategoricalAccuracy'])

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

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

    print('Accuracy on Test Data: %2.2f%%' % (accuracy_score(test_labels, label_preds)))
    print(classification_report(test_labels, label_preds))

    template = 'Test Loss: {}, Test Accuracy: {}'
    logging.info(template.format(test_loss.result(), test_accuracy.result() * 100))

    template = 'Confusion Matrix:\n{}'
    logging.info(template.format(test_cm.result().numpy()))

    # Plot confusion matrix
    plot_path = os.path.join(run_paths['path_plt'], ' confusion_matrix.png')
    cm = np.array(test_cm.result().numpy().tolist())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt='.1%')
    plt.xlabel('True', fontsize=15)  # x-axis label with fontsize 15
    plt.ylabel('Predict', fontsize=15)  # y-axis label with fontsize 15
    plt.savefig(plot_path)
    plt.show()

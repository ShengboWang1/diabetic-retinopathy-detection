import tensorflow as tf
from evaluation.metrics import ConfusionMatrix, CM
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def evaluate(model, checkpoint, ds_val, ds_test, ds_info, run_paths):
    test_CM = CM()
    checkpoint.restore(tf.train.latest_checkpoint('./checkpoint/checkpoint/train/20201217-033207'))
    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')
    for val_images, val_labels in ds_val:
        val_loss, val_accuracy = model.evaluate(val_images, val_labels, verbose=1)
    template = 'val Loss: {}, val Accuracy: {}'
    print(template.format(val_loss, val_accuracy * 100))
    for test_images, test_labels in ds_test:
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
        predictions = model(test_images, training=True)
        label_pred = tf.math.argmax(predictions, axis=1)
        #test_CM.update_state(test_labels, label_pred)

    template = 'Test Loss: {}, Test Accuracy: {}'
    print(template.format(test_loss, test_accuracy * 100))

    pred_Y = model.predict(test_images, batch_size=16, verbose=True)
    pred_Y_cat = np.argmax(pred_Y, -1)
    test_Y_cat = np.argmax(test_labels, -1)
    # print('Accuracy on Test Data: %2.2f%%' % (accuracy_score(test_Y_cat, pred_Y_cat)))
    print(classification_report(test_Y_cat, pred_Y_cat))

    # template = 'Confusion Matrix:{}'
    #print(template.format(test_CM.result().numpy()))
    return test_accuracy



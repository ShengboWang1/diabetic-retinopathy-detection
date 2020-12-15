import tensorflow as tf
from evaluation.metrics import ConfusionMatrix, CM


def evaluate(model, checkpoint, ds_test, ds_info, run_paths):
    test_CM = CM()
    # checkpoint.restore(tf.train.latest_checkpoint('./checkpoint/train'))
    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')
    for test_images, test_labels in ds_test:
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
        predictions = model(test_images, training=True)
        label_pred = tf.math.argmax(predictions, axis=1)
        test_CM.update_state(test_labels, label_pred)

    template = 'Test Loss: {}, Test Accuracy: {}'
    print(template.format(test_loss, test_accuracy * 100))
    template = 'Confusion Matrix:{}'
    print(template.format(test_CM.result().numpy()))
    return test_accuracy


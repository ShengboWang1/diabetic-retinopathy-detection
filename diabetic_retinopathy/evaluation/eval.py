import tensorflow as tf
import logging
import gin
from evaluation.metrics import ConfusionMatrix


# def evaluate(model, checkpoint, ds_test, ds_info, run_paths):
class Evaluator(object):
    def __init__(self, model, ds_test, ds_info, run_paths):
        self.model = model
        self.ds_test = ds_test
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.checkpoint = tf.train.Checkpoint(myModel=self.model)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.test_summary_writer = tf.summary.create_file_writer('./test_summary')


    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
        # pred_label = tf.math.argmax(predictions, axis=1)
        # tp, tn, fp, fn = ConfusionMatrix.update_state(labels, pred_label)

    def evaluate(self):
        self.checkpoint.restore(tf.train.latest_checkpoint('./checkpoint/train'))
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        for images, labels in self.ds_test:
            self.test_step(images, labels)

        with self.test_summary_writer.as_default():
            tf.summary.scalar('test_loss', self.test_loss.result(), step=0)
            tf.summary.scalar('test_accuracy', self.test_accuracy.result(), step=0)
            # tf.summary.image('Confusion Matrix test', fig_CM, step=0)
            template = 'Test Loss: {}, Test Accuracy: {}'
            print(template.format(self.test_loss.result(), self.test_accuracy.result() * 100))
        return self.test_accuracy.result().numpy()


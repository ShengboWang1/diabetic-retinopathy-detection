import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow.python.keras import backend as K
import sys

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init(self, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        # ...
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.tn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None, *args, **kwargs):
        # ...
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        # tp = tf.math.count_nonzero(y_pred * y_true, dtype=tf.float32)
        # fp = tf.math.count_nonzero(y_pred * (1 - y_true), dtype=tf.float32)
        # fn = tf.math.count_nonzero((1 - y_pred) * y_true, dtype=tf.float32)

        tp_values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        tp_values = tf.cast(tp_values, self.dtype)
        tn_values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        tn_values = tf.cast(tn_values, self.dtype)
        fp_values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        fp_values = tf.cast(fp_values, self.dtype)
        fn_values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        fn_values = tf.cast(fn_values, self.dtype)

        def check_sample_weight(values):
            if sample_weight is not None:
                sample_weight = tf.cast(sample_weight, self.dtype)
                sample_weight = tf.broadcast_to(sample_weight, tp_values.shape)
                values = tf.multiply(values, sample_weight)
            return values

        self.tp.assign_add(tf.reduce_sum(check_sample_weight(tp_values)))
        self.tn.assign_add(tf.reduce_sum(check_sample_weight(tn_values)))
        self.fp.assign_add(tf.reduce_sum(check_sample_weight(fp_values)))
        self.fn.assign_add(tf.reduce_sum(check_sample_weight(fn_values)))

    def result(self):
        # ..

        return self.tp, self.tn, self.fp, self.fn


class ConfusionMatrixMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """

    def __init__(self, num_classes, **kwargs):
        super(ConfusionMatrixMetric, self).__init__(name='confusion_matrix_metric',
                                                    **kwargs)  # handles base args (e.g., dtype)
        self.num_classes = num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes, num_classes), initializer="zeros")

    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        # self.total_cm.assign_add(self.confusion_matrix(y_true, y_pred))
        # return self.total_cm
        # convert predictions from probability to boolean
        y_pred = tf.math.argmax(y_pred, axis=1)
        # y_true = tf.cast(y_true, tf.bool)
        # apply confusion matrix
        cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=self.num_classes)
        self.total_cm.assign_add(cm)

    def result(self):
        # return self.process_confusion_matrix()
        return self.total_cm


    def confusion_matrix(self, y_true, y_pred):
        """
        Make a confusion matrix
        """
        ##### y_pred = tf.argmax(y_pred, 1)
        cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=self.num_classes)
        return cm

    def process_confusion_matrix(self):
        "returns sensitivity and specificity along with overall accuracy"
        cm = self.total_cm
        diag_part = tf.linalg.diag_part(cm)
        # precision = diag_part / (tf.reduce_sum(cm, 0) + tf.constant(1e-15))
        # recall = diag_part / (tf.reduce_sum(cm, 1) + tf.constant(1e-15))
        # f1 = 2 * precision * recall / (precision + recall + tf.constant(1e-15))
        sensitivity_specificity = diag_part / (tf.reduce_sum(cm, 0) + tf.constant(1e-15))
        sensitivity = sensitivity_specificity.numpy()[1]
        specificity = sensitivity_specificity.numpy()[0]
        return sensitivity, specificity

    def fill_output(self, output):
        results = self.result()
        for i in range(self.num_classes):
            output['precision_{}'.format(i)] = results[0][i]
            output['recall_{}'.format(i)] = results[1][i]
            output['F1_{}'.format(i)] = results[2][i]



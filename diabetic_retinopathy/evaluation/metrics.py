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

# class ConfusionMatrix(tf.keras.metrics.Metric):
#
#     def __init(self, name="confusion_matrix", **kwargs):
#         super(ConfusionMatrix, self).__init__(name=name, **kwargs)
#         # confusion matrix
#         self.noc = noc
#         self.confusion_matrix = self.add_weight(
#             name="confusion_matrix",
#             shape=(noc, noc)
#         initializer = "zeros", dtype = tf.int32)
#
#     def update_state(self, *args, **kwargs):
#         # update state
#         confusion_matrix = tf.math.confusion_matrix(y_true, tf.argmax(y_pred, axis=1), num_classes=self.noc)
#         return self.confusion_matrix.assign_add(confusion_matrix)
#
#     def result(self):
#         # result
#         diag = tf.linalg.diag_part(self.confusion_matrix)
#         rowsums = tf.math.reduce_sum(self.confusion_matrix, axis=1)
#         result = tf.math.reduce_mean(diag / rowsums, axis=0)
#
#         return result


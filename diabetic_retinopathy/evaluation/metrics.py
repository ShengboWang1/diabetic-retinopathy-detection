import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow.python.keras import backend as K
import sys

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init(self, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        # confusion matrix
        self.noc = noc
        self.confusion_matrix = self.add_weight(
            name="confusion_matrix",
            shape=(noc, noc)
        initializer = "zeros",
                      dtype = tf.int32
        )

    def update_state(self, *args, **kwargs):
        # update state
        confusion_matrix = tf.math.confusion_matrix(y_true, tf.argmax(y_pred, axis=1), num_classes=self.noc)
        return self.confusion_matrix.assign_add(confusion_matrix)

    def result(self):
        # result
        diag = tf.linalg.diag_part(self.confusion_matrix)
        rowsums = tf.math.reduce_sum(self.confusion_matrix, axis=1)
        result = tf.math.reduce_mean(diag / rowsums, axis=0)

        return result



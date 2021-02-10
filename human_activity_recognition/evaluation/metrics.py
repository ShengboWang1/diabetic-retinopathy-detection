import tensorflow as tf


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
        # y_pred = tf.math.argmax(y_pred, axis=1)
        # y_true = tf.cast(y_true, tf.bool)
        # apply confusion matrix
        cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=self.num_classes)
        cm = tf.transpose(cm)
        self.total_cm.assign_add(cm)

    def result(self):
        # return self.process_confusion_matrix()
        return self.total_cm

    def confusion_matrix(self, y_true, y_pred):
        """
        Make a confusion matrix
        """
        # y_pred = tf.argmax(y_pred, 1)
        cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=self.num_classes)
        return cm

    def process_confusion_matrix(self):
        """returns sensitivity and specificity along with overall accuracy"""
        cm = self.total_cm
        diag_part = tf.linalg.diag_part(cm)
        # precision = diag_part / (tf.reduce_sum(cm, 0) + tf.constant(1e-15))
        # recall = diag_part / (tf.reduce_sum(cm, 1) + tf.constant(1e-15))
        # f1 = 2 * precision * recall / (precision + recall + tf.constant(1e-15))
        sensitivity_specificity = diag_part / (tf.reduce_sum(cm, 0) + tf.constant(1e-15))
        sensitivity = sensitivity_specificity.numpy()[1]
        specificity = sensitivity_specificity.numpy()[0]
        return sensitivity, specificity

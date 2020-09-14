import tensorflow as tf

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init(self, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        # ...

    def update_state(self, *args, **kwargs):
        # ...

    def result(self):
        # ...

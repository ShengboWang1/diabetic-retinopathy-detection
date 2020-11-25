import tensorflow as tf
from keras.callbacks import ModelCheckpoint

def evaluate(model, checkpoint, ds_test, ds_info, run_paths):
    results=model.evaluate(
         x=x_test,
         y=y_test,
         batch_size=batch_size
         verbose=1,
         sample_weight=None)
    tf.keras.callbacks.ModelCheckpoint(
        filepath=run_paths,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
        save_freq='epoch',
        options=None,
        **kwargs)

    return
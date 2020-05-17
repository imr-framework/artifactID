import tensorflow as tf


class SliceUpdateCallback(tf.keras.callbacks.Callback):
    def __init__(self, data_generator):
        super().__init__()
        self.data_generator = data_generator

    def on_epoch_end(self, epoch, logs=None):
        print('Updating slices for cross-validation...')
        self.data_generator.update_slices_cross_validation()

from tensorflow.keras.callbacks import Callback
from time import clock


class TimeCallback(Callback):
    def __init__(self):
        self.start = clock()

    def on_epoch_begin(self, epoch, logs=None):
        self.start = clock()

    def on_epoch_end(self, epoch, logs={}):
        logs['time'] = clock() - self.start

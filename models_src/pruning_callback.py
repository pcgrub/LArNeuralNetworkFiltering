"""
Multiplies a models weights with a mask to avoid the updating of pruned weights.

author: Clemens Grubitz (mailto:clemens@grubitz.eu)
"""

from keras.callbacks import Callback

class PruningCallback(Callback):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def on_epoch_end(self, epoch, logs=None):
        self.model = self.mask.apply_mask(self.model)


from models_src import TrainingData
from keras.callbacks import Callback
import numpy as np


"""Make a prediction between epochs and save characteristic values"""

class PredictionCallback(Callback):
    def __init__(self, test, scale):
        super().__init__()
        self.upper_percentile_sig = []
        self.lower_percentile_sig = []
        self.stds_sig = []
        self.means = []
        self.medians = []
        self.testing_data = test
        self.scale = scale


    def on_epoch_end(self, epoch, logs=None):
        x_test, y_test = self.testing_data
        results = self.scale*self.model.predict(x_test).flatten()
        y_true = self.scale*y_test.flatten()
        difference = results-y_true
        
        diff_sig = 
        self.means.append(np.mean(difference))
        self.medians.append(np.median(difference))
        self.stds.append(np.std(difference))
        self.lower_percentile.append(np.percentile(difference, 1))
        self.lower_percentile.append(np.percentile(difference, 99))

"""
Make a prediction between epochs and save characteristic values

The characteristic values are stored on training end into an hdf5.file
author: Clemens Grubitz (mailto:clemens@grubitz.eu)
"""
from keras.callbacks import Callback
import numpy as np
import pandas as pd


class AnalysisCallback(Callback):
    def __init__(self, testing_data, scale, pu_threshold, save_path):
        super().__init__()

        # pass data used for validation process
        self.testing_data = testing_data
        self.scale = scale
        self.pu_threshold = pu_threshold
        columns = ['epoch',
                   'mean_sig', 'median_sig', 'std_sig',
                   '1percentile_sig', '99percentile_sig',
                   'mean_pu', 'median_pu', 'std_pu',
                   '1percentile_pu', '99percentile_pu',
                   'loss', 'validation_loss']
        self.analysis_df = pd.DataFrame(columns=columns)
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        x_test, y_test = self.testing_data
        results = self.scale*self.model.predict(x_test).flatten()
        y_true = self.scale*y_test.flatten()
        difference = results-y_true
        diff_sig = difference[y_true>self.pu_threshold]
        diff_pu = difference[y_true<= self.pu_threshold]

        loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        self.analysis_df = self.analysis_df.append(
            {'mean_sig' : np.mean(diff_sig),
             'median_sig' : np.median(diff_sig),
             'std_sig' : np.std(diff_sig),
             '1percentile_sig' : np.percentile(diff_sig, 1),
             '99percentile_sig' : np.percentile(diff_sig, 99),

             'mean_pu' : np.mean(diff_pu),
             'median_pu' : np.median(diff_pu),
             'std_pu' : np.std(diff_pu),
             '1percentile_pu' : np.percentile(diff_pu, 1),
             '99percentile_pu' : np.percentile(diff_pu, 99),

             'epoch': epoch,
             'loss': loss,
             'validation_loss': val_loss

             }, ignore_index=True)


    def on_train_end(self ,logs={}):
        self.analysis_df.to_csv(self.save_path + 'analysis.csv')

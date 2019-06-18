from keras.callbacks import Callback
import numpy as np
import pandas as pd

"""Make a prediction between epochs and save characteristic values"""

class AnalysisCallback(Callback):
    def __init__(self, test, scale, pu_threshold):
        super().__init__()
        self.test = test
        self.scale = scale
        self.pu_threshold = pu_threshold
        columns = ['epoch',
                   'mean_sig', 'median_sig', 'std_sig',
                   '1percentile_sig', '99percentile_sig',
                   'mean_pu', 'median_pu', 'std_pu',
                   '1percentile_pu', '99percentile_pu']
        self.analysis_df = pd.DataFrame(columns=columns)

    def on_epoch_end(self, epoch, logs=None):
        x_test, y_test = self.testing_data
        results = self.scale*self.model.predict(x_test).flatten()
        y_true = self.scale*y_test.flatten()
        difference = results-y_true
        diff_sig = difference[y_true>self.pu_threshold]
        diff_pu = difference[y_true<= self.pu_threshold]

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

             'epoch': epoch
             },
            ignore_index=True)


    def on_training_end(self, logs={}):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        self.analysis_df['loss'] = loss
        self.analysis_df['val_loss'] = val_loss
        self.analysis_df.to_hdf(
            '/home/cgrubitz/nn_models/output/test_analysis.h5',
            key='analysis_df')

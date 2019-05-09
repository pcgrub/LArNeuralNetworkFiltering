"""Doc_string"""


#import os
#from keras.models import load_model
#import numpy as np
from models_src.training_data import TrainingData
#from models_src.pruning_callback import PruningCallback
from models_src.pruning_run import PruningRun
#from models_src.weight_mask import WeightMask


def main():
    """Simple script to prune a model"""
    model_path = ('saved_models/GatedRecurrentTLFN/' +
                  'Normal-30-adam-150_epochs-6d-' +
                  'OFdataset/Graph/GatedRecurrentTLFN-(30, 30)/' +
                  'run2/GatedRecurrentTLFN-(30, 30).h5')

    thresholds = [0.03, 0.01, 0.01]
    epochs = [1, 1, 1]
    layers = [3, 4, 5]

    slice_len = 30
    delay = 6

    td1 = TrainingData()
    training_data = td1.window_dim_1_sized_td(slice_len, delay)

    p_run = PruningRun(model_path, training_data)
    for i in range(2):
        p_run.prune_layer(layers[i], thresholds[i], epochs[i])

if __name__ == "__main__":
    main()

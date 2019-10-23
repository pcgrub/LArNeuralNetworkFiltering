"""Doc_string"""


#import os
#from keras.models import load_model
from models_src.training_data import TrainingData
from models_src.pruning_run import PruningRun


def main():
    """Simple script to prune a model"""

    # path of model that should be pruned
    model_path = ('saved_models/PATH_TO_MODEL/model.h5')

    # weights below this threshold will be set to zero
    # thresholds can be defined per layer
    thresholds = [0.03, 0.01, 0.01]

    # specify training epochs for retraining
    epochs = [1, 1, 1]
    # define the layer index that should be pruned
    # only feedforward layers can be pruned!!!
    layers = [3, 4, 5]

    # TrainingData section
    # specify input dimension of the sliding window using 'slice_len'
    slice_len = 30

    # output delay for AREUS data
    delay = 6

    td1 = TrainingData()
    training_data = td1.window_dim_1_sized_td(slice_len, delay)

    # Pruning runs for each layer
    p_run = PruningRun(model_path, training_data)
    for i, layer in enumerate(layers):
        p_run.prune_layer(layer, thresholds[i], epochs[i])

        # when no retraining is needed
        #p_run.prune_layer_no_retraining(layer, thresholds[i])

if __name__ == "__main__":
    main()

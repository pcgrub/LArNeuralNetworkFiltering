from models_src import *
import sys


def main():
    model = GatedRecurrentTLFNreg

    # Training Parameters:
    runs = 1
    epochs = 1

    delay = 6
    slice_len = 30

    training_params = (slice_len, delay)

    # Model specific parameters
    dim = slice_len

    n = 2
    l_neurons_per_layer = 30
    params = (dim, l_neurons_per_layer)

    comments = 'l2reg0.01-30-adam-150_epochs-6d-OFdataset'


    td1 = TrainingData()
    training_data = td1.window_dim_1_sized_td(*training_params)

    run1 = RunSingleModel(model, runs, epochs, params,
                          training_data, comments)

    run1.run()


if __name__ == "__main__":
    main()

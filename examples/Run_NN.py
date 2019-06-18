import sys

# Path to models_src needs to be changed if file or module is moved
sys.path.insert(0, '..')

from models_src import *
def main():
    model = GatedRecurrentTLFN

    # Training Parameters:
    runs = 1
    epochs = 150

    delay = 6
    slice_len = 30

    training_params = (slice_len, delay)

    # Model specific parameters
    dim = slice_len

    # neurons in hidden layer
    l_neurons_per_layer = 30
    params = (dim, l_neurons_per_layer)

    comments = 'OFdataset'

    # TrainingData path and file prefixes
    td1 = TrainingData(path='/ZIH.fast/users/ML_berthold_grubitz/data/TestCell/OFMaxFinder/', prefix='EMB_EMMiddle_0.5125X0.0125_OF_')
    training_data = td1.window_dim_1_sized_td(*training_params)


    # Scaling factor is used for later in an AnalysisCallback
    scale = td1.eT_scale
    run1 = RunSingleModel(model, runs, epochs, params,
                          training_data, comments, scale)

    run1.run()


if __name__ == "__main__":
    main()

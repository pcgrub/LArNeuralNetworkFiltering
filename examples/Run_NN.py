import sys

# Path to models_src needs to be changed if file or module is moved
sys.path.insert(0, '..')

from models_src import *
def main():
    model = igru_regression

    # Training Parameters:
    runs = 1
    epochs = 5

    # training data parameters
    delay = 6
    slice_len = 30

    training_params = (slice_len, delay)

    # neurons in hidden layer
    l_neurons_per_layer = 30
    params = (slice_len, l_neurons_per_layer)

    comments = 'some_comment'

    # TrainingData path and file prefixes
    td1 = TrainingData(path='testdata/TestCell/OFMaxFinder/',
                       prefix='EMB_EMMiddle_0.5125X0.0125_OF_')
    training_data = td1.window_dim_1_sized_td(*training_params)


    # Scaling factor is used for later in an AnalysisCallback
    scale = td1.eT_scale
    run1 = RunSingleModel(model, runs, epochs, params,
                          training_data, comments, scale)

    run1.run()


if __name__ == "__main__":
    main()

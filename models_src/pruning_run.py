""" Class implementing a pruning run """

import os
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.callbacks import TensorBoard
from models_src.pruning_callback import PruningCallback
from models_src.weight_mask import WeightMask


# Limit number of threads to one master-thread and one worker-thread
tf_config = K.tf.ConfigProto(intra_op_parallelism_threads=1,
                             inter_op_parallelism_threads=1)
K.set_session(K.tf.Session(config=tf_config))


class PruningRun():
    """
    Doc_string
    """
    def __init__(self, model_path, training_data):
        self.model_path = model_path
        self.model = load_model(self.model_path)
        self.train, self.test = training_data
        self.mask = WeightMask(self.model)

    def prune_layer(self, layer_index, threshold, epochs):
        """
        Prunes layer weights or biases to be lower than a certain threshold

        Parameters:
        layer_index - layer that is supposed to be pruned
        threshold - weights below this value will be pruned
        epochs - number of training epochs after pruning one weight

        Returns:
        Model with pruned weights

        """
        number_of_pruned_values = 0
        path_for_saving, file_name = os.path.split(self.model_path)
        log_dir = path_for_saving + '/Pruning/Graph/'

        tb_call = TensorBoard(log_dir=log_dir, histogram_freq=0, write_images=True,
                              write_graph=True)

        while True:

            # Get all weights in the respective layer with index 'layer_index'
            # Only only absolute valuies are relevant in this pruning method
            prunable_weights = np.abs(self.model.get_weights()[layer_index])

            # find the maximum number of weights to be pruned
            max_weights = len(prunable_weights.flatten())

            # Removing weights that are already 0 using a mask
            masked = np.ma.MaskedArray(prunable_weights, prunable_weights == 0)

            # Finding the index of the smallest weight
            prune_index = np.unravel_index(np.ma.argmin(masked), prunable_weights.shape)

            # Finding the value of the smallest weight
            pruned_value = prunable_weights[prune_index]

            # weight has to be non 0 and below threshold to be removed
            # in the case of non-converging process the pruning will be cancelled after all
            # weights are pruned
            if 0 < pruned_value < threshold and number_of_pruned_values < max_weights:

                # Set weight in layer 'layer_index' and location 'prune_index' to 0
                self.mask.prune_weight(layer_index, *prune_index)
                model = self.mask.apply_mask(self.model)

                # During the process the WeightMask has to be applied
                # over and over to avoid updating weights
                pruning_call = PruningCallback(self.mask)

                # Retraining the Network
                self.model.compile(loss='mean_squared_error', optimizer='adam')
                self.model.fit(*self.train, validation_data=self.test, epochs=epochs, verbose=2,
                               callbacks=[pruning_call, tb_call])
                number_of_pruned_values += 1

            # Pruning is finished
            else:
                break

          # Final retraining
        pruning_call = PruningCallback(self.mask)
        self.model.fit(*self.train, validation_data=self.test, epochs=20, verbose=2,
                       callbacks=[pruning_call, tb_call])

        rem_weights = self.get_remaining_weights_number()

        self.model.save(path_for_saving + 'Remaining_weights'
                        + str(rem_weights) + '_' + file_name)


    def get_remaining_weights_number(self):
        """
        Returns:
        The number of remaining nonzero weights in the mask
        """
        weights = np.asarray(self.mask.get_mask())
        return np.count_nonzero(weights)

"""
TrainingData class

This class processes the data used for training and testing.

Parameters:
path - containing files with 10000 Training samples each
prefix - file prefix for each of these files

Defaults:
path = /ZIH.fast/users/ML_berthold_grubitz/data/TestCell/ofMaxFinder/
prefix = EMB_EMMiddle_0.5125X0.0125_of_


Methods:
of_data - Returns the output the Optimal-Filter+Maxfinder would give
          on the training and testing set

delayed_td - Returns training data where the output is delayed by n samples

chunk_td - Generating unprocessed training data

window_dim_1_sized_td - Generates processed training data, where an input of
                        length 'slice_len' is mapped to an output of length 1.
                        The output gets delayed to match specifications,
                        i.e the output samples should be given after 'delay-1'
                        samples of the corresponding detector pulse.

create_input_files - Finding all files with a certain prefix and returning
                     data from those files.

"""

import os
import numpy as np
import h5py

class TrainingData():
    def __init__(self, path='/ZIH.fast/users/ML_berthold_grubitz/data/TestCell/OFMaxFinder/',
                 prefix='EMB_EMMiddle_0.5125X0.0125_OF_'):
        sets = self.create_input_files(path=path, prefix=prefix) 
        self.eT_scale = np.amax(sets)
        self.normalized = sets / self.eT_scale


    def process_input(self, input_path):
        """
        Extracting relevant columns from an HDF5-input file

        The columns are:
        'dig_eT' - Digitizer Output, E_T [GeV]
        'hit_eT' - real deposited enegery, E_T [GeV]
        'OFMax_eT' - Energy reconstruction using Optimal-Filter+Maximum-Finder,
        E_T [GeV]

        Returns:
        numpy-ndarray containing the three clolumns
        """
        input_file = h5py.File(input_path, 'r')
        dig_eT = input_file['sequence_dig_eT'][()]
        hit_eT = input_file['sequence_hit_eT'][()]
        ofMax_eT = input_file['sequence_OFMax_eT'][()]
        return np.column_stack((dig_eT, hit_eT, ofMax_eT))


    def of_data(self, total_length):
        """
        Returns the output the Optimal-Filter+Maxfinder would give
        on the training and testing set

        Parameters:
        total_length - length of output data sets

        Returns:
        Pair training set and testing set, each being pair of input and output
        data.
        """
        of_sets = np.asarray(self.normalized[:, :, 2])
        of_train, of_test = np.split(np.asarray(of_sets), 2, axis=1)
        nf, samples = of_train.shape
        of_train = of_train.reshape(nf, samples, 1)
        of_test = of_test.reshape(nf, samples, 1)


        nof = of_test.shape[0]
        new_length = int(total_length / nof)
        of_train = of_train[:, -new_length:, :]
        of_test = of_test[:, -new_length:, :]

        of_train = np.concatenate(of_train)
        of_test = np.concatenate(of_test)

        # Reshape into 3D array for NN:
        of_train = of_train.reshape(total_length, 1, 1)
        of_test = of_test.reshape(total_length, 1, 1)

        return of_train, of_test


    def delayed_td(self, n):
        """
        Returns training data where the output is delayed by n samples

        Parameters:
        n - number of samples the output should be delayed

        Returns:
        Pair training set and testing set, each being pair of input and output
        data.
        """
        (x_train, y_train), (x_test, y_test) = self.chunk_td()

        y_train = np.roll(y_train, n, axis=1)
        y_test = np.roll(y_test, n, axis=1)

        y_train[:, :n, :] = 0
        y_test[:, :n, :] = 0

        return (x_train, y_train), (x_test, y_test)

    def chunk_td(self):
        """
        Generating unprocessed training data.

        Returns:
        Pair training set and testing set, each being pair of input and output
        data.
        """

        d_norm = np.asarray(self.normalized[:, :, 0])
        h_norm = np.asarray(self.normalized[:, :, 1])

        # Split in half for testing and training data
        x_train, x_test = np.split(d_norm, 2, axis=1)
        y_train, y_test = np.split(h_norm, 2, axis=1)
        nf, samples = x_train.shape

        x_train = x_train.reshape(nf, samples, 1)
        x_test = x_test.reshape(nf, samples, 1)
        y_train = y_train.reshape(nf, samples, 1)
        y_test = y_test.reshape(nf, samples, 1)

        return (x_train, y_train), (x_test, y_test)

    def window_dim_1_sized_td(self, slice_len, delay):
        """
        Generates processed training data, where an input of length 'slice_len'
        is mapped to an output of length 1. The output gets delayed to match
        specifications, .i.e the output samples should be given after 'delay-1'
        samples of the corresponding detector pulse.

        Parameters:
        delay - number of samples the output should be delayed
        slice_len - number of samples given as input

        Returns:
        Pair training set and testing set, each being pair of input and output
        data.
        """
        (x_train, y_train), (x_test, y_test) = self.delayed_td(delay)

        # Process input into overlapping slices with length "slice_len"
        x_train = self.split_up(x_train, slice_len)
        x_test = self.split_up(x_test, slice_len)

        # shorten corresponding output data by taking
        # samples away from the front
        nof, nos, dim = y_test.shape
        total_length = len(x_test)
        new_length = int(total_length / nof)
        y_train = y_train[:, -new_length:, :]
        y_test = y_test[:, -new_length:, :]

        y_train = np.concatenate(y_train)
        y_test = np.concatenate(y_test)

        # Reshape into 3D array for NN:
        y_train = y_train.reshape(total_length, 1, 1)
        y_test = y_test.reshape(total_length, 1, 1)

        return (x_train, y_train), (x_test, y_test)


    def split_up(self, sequence, slice_len):
        """
        Generating overlapping windows

        Parameters:
        sequence - long sequence of samples
        slice_len - size of each windows

        Returns:
        Array of overlapping windows with length 'slice_len'
        """
        nof, nos = sequence.shape [:-1]
        new_length = nos - slice_len + 1

        # split up each file and put in list
        temp_list = []
        for n in range(nof):
            temp_array = np.ones((new_length, slice_len, 1))
            for i in range(new_length):
                temp_array[i, :, 0] = sequence[n, i:(i + slice_len), 0]
            temp_list.append(temp_array)

        return np.concatenate(temp_list)


    def create_input_files(self, path, prefix):
        """
        Finding all files with a certain prefix and returning
        data from those files

        Parameters:
        path - containing files with 10000 Training samples each
        prefix - file prefix for each of these files

        Returns:
        list with each element being a three column matrix according to
        'self.process_inputs'

        """

        sets = []
        for file in os.listdir(path):
            if file.startswith(prefix):
                file_name = os.path.join(path, file)
                sets.append(self.process_input(file_name))

        return sets

    def classification_td(self, slice_len, delay, threshold):
        """
        Generate Training data for the output of triggering decisions.

        Parameters:
        delay - number of samples the output should be delayed
        slice_len - number of samples given as input
        threshold - Above this value the trigger returns 1, below 0

        Returns:
        Training Data with 30sample windows input and trigger output 0 or 1.
        """
        (x_train, y_train), (x_test, y_test) = self.window_dim_1_sized_td(slice_len, delay)

        trigger_train = y_train > threshold
        trigger_test = y_test > threshold

        y_train = trigger_train.astype(int)
        y_test = trigger_test.astype(int)

        return (x_train, y_train), (x_test, y_test)


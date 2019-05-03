"""Doc_string"""

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
        input_file = h5py.File(input_path, 'r')
        dig_eT = input_file['sequence_dig_eT'][()]
        hit_eT = input_file['sequence_hit_eT'][()]
        OFMax_eT = input_file['sequence_OFMax_eT'][()]
        return np.column_stack((dig_eT, hit_eT, OFMax_eT))


    def OF_data(self, total_length):
        OF_sets = np.asarray(self.normalized[:, :, 2])
        OF_train, OF_test = np.split(np.asarray(OF_sets), 2, axis=1)
        nf, samples = OF_train.shape
        OF_train = OF_train.reshape(nf, samples, 1)
        OF_test = OF_test.reshape(nf, samples, 1)


        nof, nos, dim = OF_test.shape
        new_length = int(total_length / nof)
        OF_train = OF_train[:, -new_length:, :]
        OF_test = OF_test[:, -new_length:, :]

        OF_train = np.concatenate(OF_train)
        OF_test = np.concatenate(OF_test)

        # Reshape into 3D array for NN:
        OF_train = OF_train.reshape(total_length, 1, 1)
        OF_test = OF_test.reshape(total_length, 1, 1)

        return OF_train, OF_test


    def delayed_td(self, n):
        (X_train, Y_train), (X_test, Y_test) = self.chunk_td()

        Y_train = np.roll(Y_train, n, axis=1)
        Y_test = np.roll(Y_test, n, axis=1)

        Y_train[:, :n, :] = 0
        Y_test[:, :n, :] = 0

        return (X_train, Y_train), (X_test, Y_test)

    def chunk_td(self):
        d_norm = np.asarray(self.normalized[:, :, 0])
        h_norm = np.asarray(self.normalized[:, :, 1])

        # Split in half for testing and training data
        X_train, X_test = np.split(d_norm, 2, axis=1)
        Y_train, Y_test = np.split(h_norm, 2, axis=1)
        nf, samples = X_train.shape

        X_train = X_train.reshape(nf, samples, 1)
        X_test = X_test.reshape(nf, samples, 1)
        Y_train = Y_train.reshape(nf, samples, 1)
        Y_test = Y_test.reshape(nf, samples, 1)

        return (X_train, Y_train), (X_test, Y_test)

    def window_dim_1_sized_td(self, slice_len, delay):
        (X_train, Y_train), (X_test, Y_test) = self.delayed_td(delay)

        # Process input into overlapping slices with length "slice_len"
        X_train = self.split_up(X_train, slice_len)
        X_test = self.split_up(X_test, slice_len)

        # shorten corresponding output data by taking
        # samples away from the front
        nof, nos, dim = Y_test.shape
        total_length = len(X_test)
        new_length = int(total_length / nof)
        Y_train = Y_train[:, -new_length:, :]
        Y_test = Y_test[:, -new_length:, :]

        Y_train = np.concatenate(Y_train)
        Y_test = np.concatenate(Y_test)

        # Reshape into 3D array for NN:
        Y_train = Y_train.reshape(total_length, 1, 1)
        Y_test = Y_test.reshape(total_length, 1, 1)

        return (X_train, Y_train), (X_test, Y_test)

    def split_up(self, sequence, slice_len):
        nof, nos, dim = sequence.shape
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
        """Finding all files with a certain prefix and returning
        respective paths in list
        default (path='/ZIH.fast/users/ML_berthold_grubitz/data/TestCell/')
        default (prefix='EMB_EMMiddle_0.5125X0.0125_5GeV_')
        """

        sets = []
        for file in os.listdir(path):
            if file.startswith(prefix):
                file_name = os.path.join(path, file)
                sets.append(self.process_input(file_name))

        return sets



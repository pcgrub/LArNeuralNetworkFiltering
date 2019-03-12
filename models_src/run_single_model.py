import numpy as np
# from keras.models import Model
# import h5py
from .training_data import TrainingData
from keras import backend as K
from keras.callbacks import TensorBoard
import os

# Limit number of threads to one master-thread and one worker-thread
tf_config = K.tf.ConfigProto(intra_op_parallelism_threads=1,
                             inter_op_parallelism_threads=1)
K.set_session(K.tf.Session(config=tf_config))


class RunSingleModel:

    def __init__(self, model, runs, epochs, params, training_params, comments):
        '''
        Construct and initialize a model including all parameters required:
        Parameters include:
        'model' - a compilable Keras Model
        'epochs' - number of training epochs
        'params' - additional parameters for the Model
        'training_params' - paramaters required for processing training data
        'comments' - info to be passed into the foldernames of output-data
        '''
        self.name = str(model.__name__)
        self.model = model
        self.epochs = epochs
        self.params = params
        self.training_params = training_params
        self.comments = comments

        # create a title including model name and parameters
        # for output files
        self.sim_title = self.name + '-' + str(params)

        # The highest energy measured in the hit samples is stored
        # to be able to scale the data for being smaller or equal to 1
        # in the neural network processing and to be rescaled afterwards
        self.scale = 0

        # create empty file list for training data files
        self.file_list = []

        # fill the file list with default files
        self.create_input_files()

        # Number of Runs
        self.runs = runs

        # create empty value_loss history array with an entry for each epoch
        # self.value_loss = np.zeros((1, epochs))

    def run(self):
        """carry out one or multiple runs of training"""
        # create overall path for trained model and Tensorboard Graphs
        if self.comments == '':
            model_path = './saved_models/' + self.name + '/'

        # create a subdirectory if comments are passed
        else:
            model_path = './saved_models/' + self.name + '/' \
                + self.comments + '/'
        os.makedirs(model_path, exist_ok=True)
        print(self.sim_title)


        # create training and testing data
        train, test = self.create_training_data()

        # Since Multiple runs should be in a new directory each:
        # run numbers are generated taking into account
        # previous directories created for runs
        runs_completed = 0
        total_run_number = 1


        # iterate until the number of runs specified have been carried out,
        # regardless of how many runs have been carried out previousky
        while runs_completed<self.runs:
            log_dir = model_path + 'Graph/' + self.sim_title \
                                + '/run' + str(total_run_number) + '/'

        # Check whether a run number still exists
            if not os.path.isdir(log_dir):
                print('run number: ' + str(runs_completed + 1))
                self.run_model(train, test, log_dir)
                runs_completed += 1

        # Otherwise increment run number until a free one is found
            total_run_number += 1

    def run_model(self, train, test, log_dir):
        """one training run isolated two save memory"""

        # instantiate Neural Network model with parameters
        current_model = self.model(*self.params)

        # Enable TensorBoard analytic files
        tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_images=True,
                                 write_graph=True)
        # Training
        current_model.fit(*train, validation_data=test, epochs=self.epochs, verbose=2,
                  callbacks=[tbCallBack])

        # save models and weights
        current_model.save(log_dir + self.sim_title + ".h5")
        #del current_model

    def create_input_files(self,
                    path='/ZIH.fast/users/ML_berthold_grubitz/data/TestCell/',
                    prefix='EMB_EMMiddle_0.5125X0.0125_5GeV_'):
        """Finding all files with a certain prefix and returning
        respective paths in list
        default (path='/ZIH.fast/users/ML_berthold_grubitz/data/TestCell/')
        default (prefix='EMB_EMMiddle_0.5125X0.0125_5GeV_')
        """
        files_list = []
        for file in os.listdir(path):
            if file.startswith(prefix):
                files_list.append(os.path.join(path, file))
        self.file_list = files_list

    def create_training_data(self):
        '''Creates training data arrays

        Using the TrainingData-class an array is created containing the
        training data using the training_parameters passed to the local object.

        Returns:
            np.ndarray -- numpy-array containing training data in format
            (n_of_files, input_dim, 1)
        '''
        max_file_length = self.training_params[-1]
        training_data_params = self.training_params[:-1]

        # Data and Models:
        td_class = TrainingData(self.file_list[:max_file_length])

        # Get the energy-scale (highest possible energy in hit samples)
        #  from the training data
        self.scale = td_class.eT_scale

        # create training data with input windows and and one samples as output
        training_data = td_class.window_dim_1_sized_td(*training_data_params)
        # training_data = td_class.chunk_td()
        return training_data

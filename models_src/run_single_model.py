"""
Class implementing a number of training run for one specific model

author: Clemens Grubitz (mailto:clemens@grubitz.eu)
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from models_src.analysis_callback import AnalysisCallback

# Limit number of threads to one master-thread and one worker-thread
NUM_THREADS=1
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)

class RunSingleModel:
    def __init__(self, model, runs, epochs, params,
                 training_data, comments, scale):
        '''
        Construct and initialize a model including all parameters
        required:

        Parameters include:
        'model' - a compilable Keras Model
        'epochs' - number of training epochs
        'params' - additional parameters for the Model
        'training_params' - parameters required for processing
        training data
        'comments' - info to be passed into the foldernames of
        output-data
        '''
        self.name = str(model.__name__)
        self.model = model
        self.epochs = epochs
        self.params = params
        self.training_data = training_data
        self.comments = comments

        # create a title including model name and parameters
        # for output files
        self.sim_title = self.name + '-' + str(params)

        # The highest energy measured in the hit samples is stored
        # to be able to scale the data for being smaller or equal to 1
        # in the neural network processing and to be rescaled
        # afterwards
        self.scale = scale

        # Number of Runs
        self.runs = runs


    def run(self):
        """carry out one or multiple runs of training"""
        # create overall path for trained model and Tensorboard
        # Graphs
        if self.comments == '':
            model_path = './saved_models/' + self.name + '/'

        # create a subdirectory if comments are passed
        else:
            model_path = './saved_models/' + self.name + '/' \
                + self.comments + '/'
        os.makedirs(model_path, exist_ok=True)
        print(self.sim_title)

        # create training and testing data
        train, test = self.training_data

        # Since Multiple runs should be in a new directory each:
        # run numbers are generated taking into account
        # previous directories created for runs
        runs_completed = 0
        total_run_number = 1


        # iterate until the number of runs specified have been
        # carried out,  regardless of how many runs have been
        # carried out previously
        while runs_completed < self.runs:
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
        tb_call = TensorBoard(log_dir=log_dir, histogram_freq=0,
                              write_images=True, write_graph=True)
        ana_call = AnalysisCallback(test, self.scale, 0.8, log_dir)
        # Training
        current_model.fit(*train, validation_data=test,
                          epochs=self.epochs, verbose=2,
                          callbacks=[tb_call, ana_call])

        # save models and weights
        current_model.save(log_dir + self.sim_title + ".h5")

        # export json file
        model_json = current_model.to_json()
        with open(log_dir + self.sim_title + ".json", "w") \
             as json_file:
            json_file.write(model_json)

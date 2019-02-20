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

    def __init__(self, model, epochs, params, training_params, comments):
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
        self.model = model(*params)
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

        # create empty value_loss history array with an entry for each epoch
        # self.value_loss = np.zeros((1, epochs))

    def run(self):
        '''Execute 1 Training-run. Write Loss-files

        '''

        #if self.comments == '':
        #    loss_path = './losses/' + self.name + '/'
        #else:
        #    loss_path = './losses/' + self.name + '/' + self.comments + '/'

        train, test = self.create_training_data()
        #self.value_loss = self.run_model(train, test, self.model)
        self.run_model(train, test, self.model)
        #os.makedirs(loss_path, exist_ok=True)
        #det_file_name = loss_path + 'detailedlosses_' + self.sim_title + '.txt'
        #np.savetxt(det_file_name, self.value_loss, delimiter=',')

    def run_model(self, train, test, model, run=0):

        if self.comments == '':
            model_path = './saved_models/' + self.name + '/'
        else:
            model_path = './saved_models/' + self.name + '/' \
                + self.comments + '/'

        os.makedirs(model_path, exist_ok=True)
        print(self.sim_title)

        log_dir = model_path + 'Graph/' + self.sim_title \
                             + '/run' + str(run) + '/'

        tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0,
                                 write_graph=True, write_images=True)

        history = model.fit(*train, validation_data=test,
                            epochs=self.epochs, verbose=2,
                            callbacks=[tbCallBack])
        self.model.save(model_path + self.sim_title + ".h5")
        # return np.asarray(history.history['val_loss'])

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
        # Data and Models:
        td_class = TrainingData(self.file_list)

        # Get the energy-scale (highest possible energy in hit samples)
        #  from the training data
        self.scale = td_class.eT_scale

        # create training data with input windows and and one samples as output
        training_data = td_class.window_dim_1_sized_td(*self.training_params)
        # training_data = td_class.chunk_td()
        return training_data

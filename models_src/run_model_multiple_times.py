import numpy as np
from .run_single_model import RunSingleModel
from keras import backend as K
import os

# Limit number of threads to one master-thread and one worker-thread
tf_config = K.tf.ConfigProto(intra_op_parallelism_threads=1,
                             inter_op_parallelism_threads=1)
K.set_session(K.tf.Session(config=tf_config))


class RunModelMultipleTimes(RunSingleModel):
    def __init__(self, model, runs, epochs, params, training_params, comments):
        super().__init__(model, epochs, params, training_params, comments)
        self.runs = runs

    def run(self):

        # if self.comments == '':
        #    loss_path = './losses/' + self.name + '/'
        # else:
        #     loss_path = './losses/' + self.name + '/' + self.comments + '/'
        # os.makedirs(loss_path, exist_ok=True)

        train, test = self.create_training_data()
        # details_per_run = np.zeros((self.runs, self.epochs))

        for i in range(self.runs):
            print('run number: ' + str(i + 1))
            # details_per_run[i, :] = self.run_model(train, test, self.model)
            self.run_model(train, test, self.model, i+1)


        # CODE OBSOLETE DUE TO TENSORBOARD
        #mean_hist = np.mean(details_per_run[:, -1])
        #std_hist = np.mean(details_per_run[:, -1])

        #det_file_name = loss_path + 'detaillosses_' + self.sim_title + '.txt'
        #np.savetxt(det_file_name, details_per_run, delimiter=',')

        #tp_to_string = ",".join([str(x) for x in self.training_params])
        #p_to_string = ",".join([str(x) for x in self.params])
        #params_to_string = p_to_string + ',' + str(self.runs) + ',' \
        #                                                      + tp_to_string
        #line = str(mean_hist) + ',' + str(std_hist) \
        #                           + ',' + params_to_string + '\n'

        #output_file = open(loss_path + 'endlosses_' + self.name + '.txt', 'a')
        #output_file.write(line)
        #output_file.close()

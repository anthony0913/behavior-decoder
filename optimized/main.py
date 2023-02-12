import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.cluster import AffinityPropagation, SpectralClustering, KMeans
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class Optimizer:
    def __init__(self, data, params, freqs, constraints, scheme="power",
                 lower_freq=2, upper_freq=40):
        #self.data = data #Array containing time series data about the total session
        #self.params = params #Array containing cleaned trial parameters

        self.primary = self.process_data(scheme, data, params)
        self.config = freqs

    def process_data(self, scheme, data, params):
        primary = np.zeros((np.shape(data)[0],np.shape(data)[1],np.shape(params)[0]))
        for trial in range(np.shape(params)[0]):
            gamma = data[params[0],params[1],:]
            gamma = np.real(np.fft.fft(gamma, axis=0))
            if scheme=="power":
                primary[:,:,trial] = gamma[self.freqs,:]
        #rescale before returning
        return primary

    def optimize(self):
        training_input, testing_input, training_output, testing_output = train_test_split(
            data, exp_out, test_size=0.25, stratify = exp_out
        )
        testing_input = np.vstack((testing_input, extra_data))
        testing_output = np.hstack((testing_output, extra_exp_out))
        return 0

class Batcher:
    def __init__(self, data, params, constraints, output_column=2, start=5, end=7):
        #generate dictionary to contain params of each run by freqs
        self.start = start
        self.end = end
        cleaned_params = self.clean_params(params, output_column)

        self.acc = 0
        self.archive = np.zeros(1)

    def clean_params(self, params, output_column, constraints=None):
        #output style >>> [start_time | end_time | output]
        output = np.zeros((1,3))
        for trial in range(np.shape(params)[0]):
            valid_trial = True
            for constraint in constraints:
                if params[trial,constraint] != constraints[constraint]:
                    valid_trial = False
                    break
            if valid_trial:
                output = np.append(output, params[trial, [self.start, self.end, output_column]],axis=0)
        return output[1:,:]

    def gen_power_matrix(self, length, specify=None):
        #use np.nonzero later
        log = np.zeros((1, length))
        best_log = np.zeros((1, length))

        for i in range(math.factorial(length)):
            stop = False
            index = 0
            while not stop:
                if log[index]==0:
                    log[index]=1
                    stop=True
                index+=1

        #keep all models of same accuracy and remove all models of lower accuracy

    def update_archive(self, new_config):
        new_acc = new_config.acc
        if new_acc > self.acc:
            self.acc = new_acc
            self.archive = new_config.config
        elif new_acc == self.acc:
            self.archive = np.append(self.archive, new_config.config, axis=0)
        elif new_acc < self.acc:
            pass
        #nevermind don't use hashmaps lol
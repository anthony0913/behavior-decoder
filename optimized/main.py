import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, output_column=2):
        #generate dictionary to contain params of each run by freqs
        print()
        '''
        output style
        start_time | end_time | output
        VVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
        '''
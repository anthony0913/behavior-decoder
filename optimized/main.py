import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.cluster import AffinityPropagation, SpectralClustering, KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class Optimizer:
    def __init__(self, data, params, freqs, constraints, iterations=100):
        #self.data = data #Array containing time series data about the total session
        self.params = params #Array containing cleaned trial parameters
        self.config = freqs

        #Model evaluation
        self.acc_mean ,self.acc_stdev = self.optimize()

    def gen_reduced_matrix(self, data, params):
        '''
        Generates a reduced matrix separating individual trials
        axis 0 - time series -> frequency components
        axis 1 - neuron
        axis 2 - trial
        '''
        reduced_matrix = np.zeros((np.shape(data)[0],np.shape(data)[1],np.shape(params)[0]))
        for trial in range(np.shape(params)[0]):
            primitive = data[params[0],params[1],:]
            primitive = np.real(np.fft.fft(primitive, axis=0))
            reduced_matrix[:,:,trial] = primitive[self.freqs,:]
        #rescale before returning
        return reduced_matrix, params[:,2]

    def shuffle(self, params):
        #Split trials evenly wrt output type into reduced_trials, dump remaining trials into extra_trials
        length = min(np.sum(self.params,axis=0)[2], np.shape(params)[0])
        params = np.shuffle(params)
        pos_out, neg_out = 0, 0

        reduced_trials = np.zeros(0)
        extra_params = np.zeros(0)

        for trial in range(np.shape(params)[0]):
            if (params[trial,2]==1 and pos_out <= length) or \
                    params[trial,2]==0 and neg_out <= length:
                reduced_trials = np.append(reduced_trials, params[trial], axis=0)
            else:
                extra_trials = np.append(extra_trials, params[trial], axis=0)
        return reduced_trials, extra_trials

    def optimize(self):
        '''
        Logging format
        Accuracy | Noise control
        '''
        log = np.zeros(self.iterations)
        for iteration in range(self.iterations):
            #Generating necessary components for fitting and testing model
            reduced_trials, extra_trials = self.shuffle(self.params)
            primary_matrix, primary_output = self.gen_reduced_matrix(reduced_trials)
            extra_matrix, extra_output =self.gen_reduced_matrix(extra_trials)

            #Creating the testing/training set split
            training_input, testing_input, training_output, testing_output = train_test_split(
                primary_matrix, primary_output, test_size=0.25, stratify = [0,1] #50/50 split
            )
            testing_input = np.vstack((testing_input, extra_matrix))
            testing_output = np.hstack((testing_output, extra_output))

            #SVD fit
            classifier = SVC(random_state=0, cache_size=7000, kernel="linear")
            classifier.fit(training_input, training_output)

            #Predictions
            predicted_output = classifier.predict(testing_input)

            #Logging
            log[iteration, 0] = accuracy_score(testing_output, predicted_output)
            #add noise prediction accuracies

        return np.mean(log), np.std(log)

class Batcher:
    def __init__(self, data, params, constraints, length, output_column=2, start_col=5, end_col=7):
        self.data = data
        self.constraints = constraints
        self.cleaned_params = self.clean_params(params, start_col, end_col, output_column)

        self.power_iteration(length)

        print(self.acc)

    def clean_params(self, params, start_col, end_col, output_column, constraints=None):
        #output style >>> [start_time | end_time | output]
        output = np.zeros((1,3))
        for trial in range(np.shape(params)[0]):
            valid_trial = True
            for constraint in constraints:
                if params[trial,constraint] != constraints[constraint]:
                    valid_trial = False
                    break
            if valid_trial:
                output = np.append(output, params[trial, [start_col, end_col, output_column]],axis=0)
        return output[1:,:]

    def power_iteration(self, length):
        #use np.nonzero later
        #wait don't use np.nonzero (better archiving)
        log = np.zeros((1, length))

        self.acc = 0

        for configuration in range(math.factorial(length)):
            stop = False
            index = 0
            while not stop:
                #binary counter
                if log[index]==0:
                    log[index]=1
                    stop=True
                else:
                    log[index]=0
                index+=1
            #Iterated updating of model archive
            self.update_archive(Optimizer(self.data, self.cleaned_params, np.nonzero(log), self.constraints))

    def continuous_iteration(self):
        #just do it with lower and upper freqs
        pass

    def update_archive(self, new_config):
        #Update archived configurations and corresponding accuracy
        new_acc = new_config.acc_mean
        if new_acc > self.acc:
            #Remove previously archive and accuracy
            self.acc = new_acc
            self.archive = new_config.config
            self.stdevs = new_config.acc_stdev
        elif new_acc == self.acc:
            #Add current configuration to list of configurations corresponding to current accuracy
            self.archive = np.append(self.archive, new_config.config, axis=0)
            self.stdevs = np.append(self.stdevs, new_config.acc_stdev)
        elif new_acc < self.acc:
            #Ignore current configuration
            pass
        #nevermind don't use hashmaps lol

class Pooler:
    #Running multiple sessions in parallel
    def __init__(self):
        pass


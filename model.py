import numpy as np
import matplotlib.pyplot as plt
import math
import csv

from sklearn.cluster import AffinityPropagation, SpectralClustering, KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class Optimizer:
    def __init__(self, data, params, freqs, iterations=100, shuffles=5):
        #self.data = data #Array containing time series data about the total session
        self.params = params #Array containing cleaned trial parameters
        self.shuffles = shuffles
        self.config = freqs
        self.iterations = iterations

        #Model evaluation
        self.acc_mean, self.acc_stdev = self.optimize(data)

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
        np.random.shuffle(params)
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

    def optimize(self, data):
        '''
        Logging format
        Accuracy | Noise control
        '''
        log = np.zeros((self.iterations, 1+self.shuffles))
        for iteration in range(self.iterations):
            #Generating necessary components for fitting and testing model
            reduced_trials, extra_trials = self.shuffle(self.params)
            primary_matrix, primary_output = self.gen_reduced_matrix(data, reduced_trials)
            extra_matrix, extra_output =self.gen_reduced_matrix(data, extra_trials)

            #Creating the testing/training set split
            training_input, testing_input, training_output, testing_output = train_test_split(
                primary_matrix, primary_output, test_size=0.25, stratify = [0,1] #50/50 split
            )
            testing_input = np.vstack((testing_input, extra_matrix))
            testing_output = np.hstack((testing_output, extra_output))

            #SVD fit
            classifier = SVC(random_state=0, cache_size=7000, kernel="linear")
            classifier.fit(training_input, training_output)

            #Predictions and logging
            predicted_output = classifier.predict(testing_input)
            log[iteration, 0] = accuracy_score(testing_output, predicted_output)

            interval = int(math.floor(100 / self.shuffles))
            for shuffles in range(self.shuffles):
                distribution = shuffles / self.shuffles
                correct_output = np.random.choice([0,1], size=np.shape(predicted_output)[0], p=[1-distribution, distribution])
                log[iteration, shuffles+1] = accuracy_score(predicted_output, correct_output)

            baseline = classifier.predict(np.random.rand(np.shape(testing_output)))
            half_output = np.random.choice([0,1], size=np.shape(predicted_output)[0], p=[0.5, 0.5])
            log[iteration, -1] = accuracy_score(baseline, half_output)

        return np.mean(log), np.std(log)

class Batcher:
    def __init__(self, data, params, constraints, length, output_classes,
                 output_column=2, start_col=5, end_col=7):
        self.data = data
        self.constraints = constraints
        self.cleaned_params = self.clean_params(params, start_col, end_col,
                                                output_column, output_classes, constraints=constraints)

        self.power_iteration(length)
        self.get_statistics()

    def clean_params(self, params, start_col, end_col, output_column, output_classes, constraints=None):
        #output style >>> [start_time | end_time | output]
        output = np.zeros((1,3))
        for trial in range(np.shape(params)[0]):
            valid_trial = True
            for constraint in constraints:
                if params[trial,constraint] != constraints[constraint]:
                    valid_trial = False
                    break
            if valid_trial:
                output = np.vstack((output, params[trial, [start_col, end_col, output_column]]))
        for trial in range(1,np.shape(output)[0]):
            output[trial, -1] = output_classes[output[trial, -1]]#converts output from an object to a numerical value
        return output[1:,:]

    def power_iteration(self, length):
        #Initial values
        log = np.zeros(length)
        self.acc = 0

        for configuration in range(math.factorial(length)):
            stop = False
            index = 0
            while not stop:
                #binary counter
                print(log[index])
                if log[index]==0:
                    log[index]=1
                    stop=True
                else:
                    log[index]=0
                index+=1
            #Iterated updating of model archive
            self.update_archive(Optimizer(data=self.data, params=self.cleaned_params, freqs=np.nonzero(log)))

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

    def get_statistics(self):
        print("Complete with maximum accuracy as " + self.acc)

class Pooler:
    #Running multiple sessions in parallel
    def __init__(self, data_paths, params_paths, parallel=False):
        self.data_paths = data_paths
        self.param_paths = params_paths

        if not parallel:
            self.data, self.params = self.single_run(self.data_paths[0], self.param_paths[0])

    def single_run(self, data_path, param_path):
        #Extracts data and params from csv into a numpy array formatted for the Batcher class

        with open(data_path, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            data_array = [row for row in reader][1:]
        data = np.array(data_array, dtype=np.float32)[:, 1:]

        with open(param_path, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            params_array = list(reader)[1:]
        params = np.array(params_array, dtype=object)

        return data, params
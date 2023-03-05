import numpy as np
import matplotlib.pyplot as plt
import math
import csv

from sklearn.cluster import AffinityPropagation, SpectralClustering, KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import shuffle
from collections import defaultdict
from tqdm import tqdm


class Optimizer:
    def __init__(self, freqs, iterations=100):
        self.freqs = freqs  # current configuration of freqs accepted
        self.iterations = iterations

    def gen_reduced_matrix(self, data, params):
        '''
        Generates a reduced matrix separating individual trials
        flattened reduced matrix - change documentation here later

        axis 0 - time series -> frequency components
        axis 2 - trial
        axis 1 - neuron
        '''
        reduced_matrix = np.zeros((np.shape(self.freqs)[0],np.shape(data)[1],np.shape(params)[0]))
        for trial in range(np.shape(params)[0]):
            #primitive is the corresponding block of session time series data
            primitive = data[int(params[trial,0]):int(params[trial,1]),:]
            primitive = np.fft.rfft(primitive, axis=0).real
            reduced_matrix[:,:,trial] = primitive[self.freqs,:]
        #rescale before returning
        reduced_matrix = np.reshape(reduced_matrix, (np.shape(params)[0],-1))
        return reduced_matrix, params[:,2]

    def split(self, params):
        # Randomly shuffle the input params array.
        np.random.shuffle(params)

        # Separate the positive and negative output trials in the shuffled params array into two separate arrays.
        pos_trials = params[params[:, 2] == 1]
        neg_trials = params[params[:, 2] == 0]

        # Compute the difference between the number of positive and negative output trials.
        diff = len(pos_trials) - len(neg_trials)

        # If the difference is positive, select the first half of the excess positive output trials to be set aside as evaluation trials.
        # If the difference is negative, select the first half of the excess negative output trials to be set aside as evaluation trials.
        if diff > 0:
            eval_pos_trials = pos_trials[:diff // 2]
            eval_neg_trials = np.zeros_like(eval_pos_trials)
        elif diff < 0:
            eval_neg_trials = neg_trials[-diff // 2:]
            eval_pos_trials = np.zeros_like(eval_neg_trials)
        else:
            eval_pos_trials = np.zeros_like(pos_trials)
            eval_neg_trials = np.zeros_like(neg_trials)

        # Concatenate the remaining positive and negative output trials into a `training_trials` array.
        training_trials = np.concatenate([pos_trials[diff // 2:], neg_trials[diff // 2:]])

        # Return `training_trials` and `evaluation_trials`.
        return training_trials, np.concatenate([eval_pos_trials, eval_neg_trials])

    def optimize(self, training_matrix, training_output, eval_matrix, eval_output):
        # Cross-validation with StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        best_model, best_acc = None, 0
        accs = []
        for freqs in skf.split(training_matrix, training_output):
            freqs = self.freqs[freqs[1]]
            classifier = SVC(random_state=0, cache_size=7000, kernel="linear")
            acc = np.mean(cross_val_score(classifier, training_matrix[:, freqs], training_output, cv=skf, n_jobs=-1))
            accs.append(acc)
            if acc > best_acc:
                best_acc = acc
                best_model = classifier.set_params(**{'kernel': 'linear', 'C': 1, 'gamma': 'scale'})

        # Use the best model to predict the output for the evaluation trials
        predicted_output = best_model.predict(eval_matrix)

        # Calculate the mean accuracy and standard deviation of the SVM classifier evaluated on the evaluation set.
        mean_acc = np.mean(accuracy_score(eval_output, predicted_output))
        stdev_acc = np.std(accuracy_score(eval_output, predicted_output))

        # Return the mean accuracy and standard deviation of the SVM classifier evaluated on the evaluation set.
        return mean_acc, stdev_acc

class Batcher:
    def __init__(self, data, params, constraints, length, output_classes,
                 output_column=2, start_col=5, end_col=7):
        self.data = data
        self.constraints = constraints
        self.cleaned_params = self.clean_params(params, start_col, end_col,
                                                output_column, output_classes, constraints=constraints)

        # Split the cleaned params array into training and evaluation sets.
        self.training_trials, self.eval_trials = self.split(self.cleaned_params)

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

    def split(self, params):
        # Randomly shuffle the input params array.
        params = shuffle(params)

        # Separate the positive and negative output trials in the shuffled params array into two separate arrays.
        pos_trials = params[params[:, 2] == 1]
        neg_trials = params[params[:, 2] == 0]

        # Compute the difference between the number of positive and negative output trials.
        diff = len(pos_trials) - len(neg_trials)

        # If the difference is positive, select the first half of the excess positive output trials to be set aside as evaluation trials.
        # If the difference is negative, select the first half of the excess negative output trials to be set aside as evaluation trials.
        if diff > 0:
            eval_pos_trials = pos_trials[:diff // 2]
            eval_neg_trials = np.zeros_like(eval_pos_trials)
        elif diff < 0:
            eval_neg_trials = neg_trials[-diff // 2:]
            eval_pos_trials = np.zeros_like(eval_neg_trials)
        else:
            eval_pos_trials = np.zeros_like(pos_trials)
            eval_neg_trials = np.zeros_like(neg_trials)

        # Concatenate the remaining positive and negative output trials into a `training_trials` array.
        training_trials = np.concatenate([pos_trials[diff // 2:], neg_trials[diff // 2:]])

        # Return `training_trials` and `evaluation_trials`.
        return training_trials, np.concatenate([eval_pos_trials, eval_neg_trials])


    def power_iteration(self, length):
        #Initial values
        log = np.zeros(length)
        self.archive = defaultdict(list)
        self.acc = 0

        for configuration in tqdm(range(2**length-1)):
            stop = False
            index = 0
            # binary counter
            while not stop:
                if log[index]==0:
                    log[index]=1
                    stop=True
                else:
                    log[index]=0
                index+=1

            # Call the `optimize` method of the `Optimizer` class on the training and evaluation sets for each iteration.
            #optimizer = Optimizer(data=self.data, params=self.training_trials, freqs=np.nonzero(log)[0], iterations=100, shuffles=5)
            optimizer = Optimizer(freqs=np.nonzero(log)[0], iterations=100)
            mean_acc = optimizer.best_acc

            # Update the record of configurations and corresponding accuracy.
            self.archive[mean_acc].append(np.nonzero(log)[0])
            if mean_acc > self.acc:
                self.acc = mean_acc

            # Iterate through each mean accuracy in descending order.
            for acc in sorted(self.archive.keys(), reverse=True):
                # If the current accuracy is less than the previously recorded maximum accuracy, break the loop.
                if acc < self.acc:
                    break

                # Update the list of configurations that resulted in the current maximum accuracy.
                self.archive[self.acc].extend(self.archive[acc])

            # Remove the configurations that did not result in the current maximum accuracy.
            del self.archive[acc]

    def continuous_iteration(self):
        #just do it with lower and upper freqs
        pass

    def update_archive(self, new_config):
        #Update archived configurations and corresponding accuracy
        new_acc = new_config.acc_mean
        if new_acc > self.acc:
            #Remove previously archive and accuracy
            self.acc = new_acc
            self.archive = new_config.freqs
            self.stdevs = new_config.acc_stdev
        elif new_acc == self.acc:
            #Add current configuration to list of configurations corresponding to current accuracy
            self.archive = np.append(self.archive, new_config.freqs, axis=0)
            self.stdevs = np.append(self.stdevs, new_config.acc_stdev)
        elif new_acc < self.acc:
            #Ignore current configuration
            pass
        #nevermind don't use hashmaps lol

    def get_statistics(self):
        print("Complete with maximum accuracy as " + str(self.acc) + " using models:\n")
        best_models = []
        for freqs in self.archive[self.acc]:
            optimizer = Optimizer(data=self.data, params=self.training_trials, freqs=freqs, iterations=100, shuffles=5)
            mean_acc = optimizer.best_acc
            stdev_acc = np.std([optimizer.optimize()[1] for i in range(10)])

            best_models.append((freqs, mean_acc, stdev_acc))

        for model in best_models:
            print("Configuration (freqs):", model[0])
            print("Mean accuracy:", model[1])
            print("Standard deviation of accuracy:", model[2])

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
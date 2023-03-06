import numpy as np
#import matplotlib.pyplot as plt
#import math
import csv

#from sklearn.cluster import AffinityPropagation, SpectralClustering, KMeans
from sklearn.svm import SVC
#from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import shuffle
from collections import defaultdict
from tqdm import tqdm


class Optimizer:
    def __init__(self, data, params, freqs, folds=10, skip=False):
        self.freqs = freqs  # current configuration of freqs accepted
        self.folds = folds
        self.skip = skip
        self.train_mat, self.train_out = self.gen_reduced_matrix(data, params, self.freqs)
        if not self.skip:
            self.optimize()

    def gen_reduced_matrix(self, data, params, freqs):
        '''
        Generates a reduced matrix separating individual trials
        flattened reduced matrix - change documentation here later

        axis 0 - time series -> frequency components
        axis 2 - trial
        axis 1 - neuron
        '''
        reduced_matrix = np.zeros((np.shape(freqs)[0],data.shape[1],params.shape[0]))
        for trial in range(np.shape(params)[0]):
            #primitive is the corresponding block of session time series data
            primitive = data[int(params[trial,0]):int(params[trial,1]),:]
            #primitive = np.fft.rfft(primitive, axis=0).real[1:,:]

            primitive = np.fft.fft(primitive, axis=0).real
            reduced_matrix[:,:,trial] = primitive[freqs,:]
        reduced_matrix = reduced_matrix.reshape(params.shape[0],np.shape(freqs)[0]*data.shape[1])
        #if self.skip: print(reduced_matrix[:,0])
        return reduced_matrix, params[:,2]

    def optimize(self):
        pos_trials = self.train_mat[self.train_out == 1]
        neg_trials = self.train_mat[self.train_out == 0]
        if len(pos_trials) > len(neg_trials):
            pos_trials, pos_val = train_test_split(pos_trials, test_size=len(pos_trials) - len(neg_trials),
                                                   stratify=self.train_out[self.train_out == 1])
            val_data = pos_val
            val_labels = np.ones(len(val_data))
        else:
            neg_trials, neg_val = train_test_split(neg_trials, test_size=len(neg_trials) - len(pos_trials),
                                                   stratify=self.train_out[self.train_out == 0])
            val_data = neg_val
            val_labels = np.zeros(len(val_data))
        train_data = np.concatenate([pos_trials, neg_trials])
        train_labels = np.concatenate([np.ones(len(pos_trials)), np.zeros(len(neg_trials))])

        # Cross-validation with StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        accs = []
        for train_index, test_index in skf.split(train_data, train_labels):
            classifier = SVC(random_state=0, cache_size=7000, kernel="linear")

            current_data = np.reshape(train_data[train_index],(np.shape(train_index)[0],-1))
            current_labels = train_labels[train_index].astype(int)

            score_data = np.reshape(train_data[test_index],(np.shape(test_index)[0],-1))
            score_data = np.vstack((score_data, val_data))
            score_labels = train_labels[test_index].astype(int)
            score_labels = np.hstack((score_labels, val_labels))

            classifier.fit(current_data, current_labels)
            acc = classifier.score(score_data, score_labels)
            accs.append(acc)
        # Calculate the mean and standard deviation of the SVM classifier evaluated on the test set.
        self.acc_mean = np.mean(accs)
        self.acc_stdev = np.std(accs)

class Batcher:
    def __init__(self, data, params, constraints, length, output_classes,
                 output_column=2, start_col=5, end_col=7, folds=100, resamples=10):
        self.data = data
        self.folds = folds
        self.length = length
        self.constraints = constraints
        self.cleaned_params = self.clean_params(params, start_col, end_col,
                                                output_column, output_classes, constraints=constraints).astype(int)
        self.evaluate(resamples)

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

        # If the difference is positive, select the second half of the excess positive output trials to be set aside as evaluation trials.
        # If the difference is negative, select the second half of the excess negative output trials to be set aside as evaluation trials.
        if diff > 0:
            eval_trials = pos_trials[abs(diff) // 2:]
        elif diff < 0:
            eval_trials = neg_trials[abs(diff) // 2:]
        else:
            print("Error: Please perform train test split manually")

        # Concatenate the remaining positive and negative output trials into a `training_trials` array.
        training_trials = np.concatenate([pos_trials[:abs(diff) // 2], neg_trials[:abs(diff) // 2]])

        # Return `training_trials` and `evaluation_trials`.
        return training_trials, eval_trials

    def evaluate(self, resamples):
        best_models = np.zeros((resamples, self.length))
        statistics = np.zeros((resamples, 2))
        accuracies = np.zeros((resamples, 1))
        for resample in range(resamples):
            self.training_trials, self.eval_trials = self.split(self.cleaned_params)
            print("Current resample: " + str(resample+1))
            output = self.power_iteration()
            best_models[resample, :] = output[0] #Models are characterized by the present freqs
            statistics[resample, :] = output[1] #Model accuracies
            accuracies[resample, :] = 0 #Based off eval trials

            # Train an SVM model on the best model (freqs) generated by power_iteration() using the same number of trials from each class
            pos_trials = self.training_trials[self.training_trials[:, 2] == 1]
            neg_trials = self.training_trials[self.training_trials[:, 2] == 0]
            num_trials = min(len(pos_trials), len(neg_trials))
            pos_idx = np.random.choice(len(pos_trials), num_trials, replace=False)
            neg_idx = np.random.choice(len(neg_trials), num_trials, replace=False)

            train_trials = np.concatenate([pos_trials[pos_idx], neg_trials[neg_idx]])
            generator = Optimizer(self.data, train_trials, np.nonzero(best_models[resample, :].astype(int))[0], skip=True)

            train_data = generator.train_mat
            train_labels = generator.train_out

            svm_model = SVC(random_state=0, cache_size=7000, kernel="linear")
            svm_model.fit(train_data, train_labels)

            generator = Optimizer(self.data, self.eval_trials, np.nonzero(best_models[resample, :].astype(int))[0], skip=True)
            eval_mat = generator.train_mat
            eval_out = generator.train_out

            # Evaluate the trained SVM model on the eval_trials
            acc = svm_model.score(eval_mat, eval_out)
            accuracies[resample] = acc
            print("Accuracy on eval_trials using best model: ", acc)

        for resample in range(resamples):
            print(best_models[resample], statistics[resample], accuracies[resample])
        print("\nMean:", np.mean(accuracies), "| Stdev:", np.std(accuracies))


    def power_iteration(self):
        #Initial values
        log = np.zeros(self.length)
        archive = defaultdict(list)
        best_acc = 0

        for configuration in tqdm(range(2**self.length-1)):
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
            optimizer = Optimizer(data=self.data, params=self.training_trials, freqs=np.nonzero(log)[0])#, shuffles=5)
            mean_acc = optimizer.acc_mean
            stdev_acc = optimizer.acc_stdev

            # Update the record of configurations and corresponding accuracy.
            archive[mean_acc].append(np.nonzero(log)[0])
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_stdev = stdev_acc
                #print(mean_acc, np.nonzero(log)[0])

            # Iterate through each mean accuracy in descending order.
            for acc in sorted(archive.keys(), reverse=True):
                # If the current accuracy is less than the previously recorded maximum accuracy, break the loop.
                if acc < acc:
                    break

                # Update the list of configurations that resulted in the current maximum accuracy.
                archive[acc].extend(archive[acc])

            # Remove the configurations that did not result in the current maximum accuracy.
            del archive[acc]
        self.get_statistics(best_acc, archive)
        return log, [best_acc, best_stdev]

    def continuous_iteration(self):
        #just do it with lower and upper freqs
        pass

    def get_statistics(self, best_acc, archive):
        print("Complete with maximum accuracy as " + str(best_acc) + " using models:\n")
        best_models = []
        for freqs in archive[best_acc]:
            optimizer = Optimizer(data=self.data, params=self.training_trials, freqs=freqs, folds=100, shuffles=5)
            mean_acc = optimizer.acc_mean
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
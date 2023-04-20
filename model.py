import numpy as np
import csv

from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm

import multiprocessing
import visualization as vis
import matplotlib.pyplot as plt

class Optimizer:
    def __init__(self, data, params, kernel="poly", length=10, resamples=10, minimum=100, pca=2, visualize=False):
        self.data = data
        self.params = params
        self.kernel = kernel
        self.length = length
        self.resamples = resamples
        self.minimum = minimum
        self.pca = pca

        if visualize:
            self.gen_vis(self.params)
        else:
            self.acc = self.optimize()
            self.cacc = self.optimize(randomize=True)

    def gen_vis(self, params, skip=0):
        pos_trials = params[params[:, 2] == 1]
        neg_trials = params[params[:, 2] == 0]
        pos = self.gamma_mat(pos_trials, skip)
        neg = self.gamma_mat(neg_trials, skip)
        pos = self.sort(pos)
        neg = self.sort(neg)
        #vis.Display("pos", pos)
        #vis.Display("neg", neg)
        g = -5000 * np.ones((1, pos.shape[1]))
        temp = np.vstack((pos, g))
        temp = np.vstack((temp, neg))

        plt.imshow(temp)
        plt.show()

    def sort(self, mat):
        sim = cosine_similarity(mat)
        output = mat[np.argsort(sim[:, 0]), :]
        return output
        #sorted = mat[mat[:,0].argsort()]
        #return sorted

    def gamma_mat(self, params, skip):
        if self.pca!=None:
            output = np.zeros((params.shape[0], self.length * self.pca))
        else:
            output = np.zeros((params.shape[0], (self.data.shape[1] - 1) * (self.length - skip)))
        for trial in range(params.shape[0]):
            primitive = self.data[params[trial, 0]:params[trial, 1], 1:]
            primitive = np.real(np.fft.fft(primitive)[:self.length])[skip:]
            if self.pca != None:
                pca = PCA(n_components=self.pca)
                primitive = pca.fit_transform(primitive)
            primitive = primitive.flatten("F")
            output[trial, :] = primitive
        return output


    def optimize(self, randomize=False):
        acc = np.zeros(self.resamples)
        for resample in range(self.resamples):
            # Reset train/eval trials
            train_trials, eval_trials = self.split(self.params, randomize=randomize)
            if self.pca == None:
                train_data = np.zeros((train_trials.shape[0], (self.data.shape[1] - 1) * self.length))
                eval_data = np.zeros((eval_trials.shape[0], (self.data.shape[1] - 1) * self.length))
            else:
                train_data = np.zeros((train_trials.shape[0], self.pca * self.length))
                eval_data = np.zeros((eval_trials.shape[0], self.pca * self.length))
                pca = PCA(n_components=self.pca)

            # Configure train/eval data
            for trial in range(train_trials.shape[0]):
                primitive = self.data[train_trials[trial, 0]:train_trials[trial, 1], 1:]
                primitive = np.real(np.fft.fft(primitive)[:self.length])
                if self.pca != None:
                    primitive = pca.fit_transform(primitive)
                    pass
                primitive = primitive.flatten()
                train_data[trial, :] = primitive
            for trial in range(eval_trials.shape[0]):
                primitive = self.data[eval_trials[trial, 0]:eval_trials[trial, 1], 1:]
                primitive = np.real(np.fft.fft(primitive)[:self.length])
                if self.pca != None:
                    primitive = pca.fit_transform(primitive)
                    pass
                primitive = primitive.flatten()
                eval_data[trial, :] = primitive

            # Model
            model = SVC(random_state=0, cache_size=7000, kernel=self.kernel)
            model.fit(train_data, train_trials[:, -1])
            acc[resample] = model.score(eval_data, eval_trials[:,-1])
        return acc

    def split(self, params, randomize=False):
        params = shuffle(params)
        pos_trials = params[params[:, 2] == 1]
        neg_trials = params[params[:, 2] == 0]

        pre_min = min(pos_trials.shape[0], neg_trials.shape[0])
        if pre_min < self.minimum:
            if pre_min == 0:
                print("\033[1m" + "ERROR: There are no trials for at least one of the classes, SVM cannot be performed on this dataset!" + "\033[0m")
            else:
                print("\033[1m" + "WARNING: There are less than", self.minimum, "trials of each class! Training "
                                                                                "trials have been automatically adjusted to", str(pre_min - 1), "of each class" + "\033[0m")
            self.minimum = pre_min - 1
        training_trials = np.concatenate([pos_trials[:self.minimum], neg_trials[:self.minimum]])
        if randomize: np.random.shuffle(training_trials[:,-1])
        eval_trials = np.concatenate([pos_trials[self.minimum:], neg_trials[self.minimum:]])
        self.ratio = training_trials.shape[0] / (training_trials.shape[0] + eval_trials.shape[0])
        self.behavioral = pos_trials.shape[0] / (pos_trials.shape[0] + neg_trials.shape[0])
        return training_trials, eval_trials

class Batcher:
    def __init__(self, data, params, constraints, length, output_classes,
                 output_column=2, start_col=5, end_col=7, folds=100, resamples=10, showDiagnostics=False, visualize=False):
        self.data = data
        self.folds = folds
        self.length = length
        self.constraints = constraints
        self.showDiagnostics = showDiagnostics
        self.cleaned_params = self.clean_params(params, start_col, end_col,
                                                output_column, output_classes, constraints=constraints).astype(int)
        if visualize:
            self.gen_vis()
        else:
            self.evaluate(resamples)

    def clean_params(self, params, start_col, end_col, output_column, output_classes, constraints=None):
        #output style >>> [start_time | end_time | output]
        self.len_skip = 0
        output = np.zeros((1,3))
        for trial in range(np.shape(params)[0]):
            valid_trial = True
            for constraint in constraints: #enforce constraints
                if params[trial,constraint] != constraints[constraint]:
                    valid_trial = False
                    break
            try: #enforce length
                interval = int(params[trial, end_col]) - int(params[trial, start_col])
                if interval < self.length:#enforce length
                    if valid_trial:
                        self.len_skip += 1
                    valid_trial = False
                    continue
            except ValueError:
                valid_trial = False
                continue
            if int(params[trial, end_col]) > self.data.shape[0]:
                valid_trial = False
            if valid_trial:
                output = np.vstack((output, params[trial, [start_col, end_col, output_column]]))
        for trial in range(1,np.shape(output)[0]):
            output[trial, -1] = output_classes[output[trial, -1]]#converts output from an object to a numerical value
        return output[1:,:]

    def gen_vis(self):
        model = Optimizer(data=self.data, params=self.cleaned_params, length=self.length, visualize=True)

    def evaluate(self, resamples):
        print("Starting batch job with", resamples, "resamples")
        model = Optimizer(data=self.data, params=self.cleaned_params, kernel="poly", length=self.length,
                          resamples=resamples)
        self.accuracies = model.acc
        self.control_accuracies = model.cacc
        self.ratio = model.ratio
        self.behavioral = model.behavioral

        for resample in range(resamples):
            if self.showDiagnostics:
                print("Resample", resample, self.accuracies[resample])
            pass
        print("-"*65)
        print("Mean:", np.mean(self.accuracies), "| Stdev:", np.std(self.accuracies), "(Model)")
        print("Mean:", np.mean(self.control_accuracies), "| Stdev:", np.std(self.control_accuracies), "(Control)")
        print("Training trials accounted for", str(100 * self.ratio) + "% of the total trials")
        print("The behavioral ratio was", str(self.behavioral) + "% correct")
        print(str(self.len_skip), "trials were omitted due to insufficient length")
        print("-"*65)

    def standard_iteration(self):
        return np.ones(self.length)

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
                best_model = np.nonzero(log)[0]
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
        #self.get_statistics(best_acc, best_model)
        if self.showDiagnostics: print("Complete with maximum accuracy as " + str(best_acc) +
                                       " using model:" + str(best_model))
        return log, [best_acc, best_stdev]

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

    def parallel_run(self, data_path, param_path):
        data, params = self.single_run(data_path, param_path)
        batch = Batcher(data, params, constraints={1: "INIT", 4: "clean"}, length=10,
                        output_classes={"wrong": 0, "correct": 1}, output_column=3, start_col=6, end_col=7, folds=10,
                        resamples=10, showDiagnostics=False)
        return np.array([np.mean(batch.accuracies), np.std(batch.accuracies)])

    # TODO: make a path/params array and pass to pool.map
    '''
    def full_run(self):
        with multiprocessing.Pool() as pool:
            results = pool.map(self.parallel_run, ???)
        print(results)
    '''
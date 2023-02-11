import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AffinityPropagation, SpectralClustering, KMeans
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class Optimizer:
    def __init__(self, data, params, freqs, constraints, mode="power", lower_freq=2, upper_freq=40):
        self.data = data #Array containing time series data about the total session
        self.params = params #Array containing cleaned trial parameters

    def process_data(self):
        return 0

    def optimize(self):
        return 0

class Batcher:
    def __init__(self, output_column=2):
        print()
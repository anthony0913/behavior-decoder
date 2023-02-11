import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AffinityPropagation, SpectralClustering, KMeans
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class Optimizer:
    def __init__(self, data, params, freqs, output_column, mode="power", lower_freq, upper_freq,):
        self.data = data
        self.params = params

    def process_data(self):
        return 0

    def clean_params(self):
        return 0

    def optimize(self):
        return 0
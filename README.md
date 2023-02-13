# How to use
This script searches for singular value decomposition (SVD) models that best describe a given dataset. In particular, SVD is applied to the component frequencies of multiple time series recordings to describe phenomena that may occur several times within a single recording session, e.g., describing a behavior testing session that contains 150 trials. 

The primary module is found in `optimized/main.py` See details regarding each class and their available methods below.

## Optimizer

### Usage
The optimizer class generates a single model to describe a session `data` with trials specified by `params` with a characteristic band of searched frequencies `freqs`. This model is cross-validated across a specified number of iterations `iterations` with the training and testing trial specification randomized automatically. 

>See **Batcher** for more information on formatting requirements for `data` and `params`

The overall performance of the model is further evaluated on `shuffles` number of shuffled datasets, where the datasets are shuffled and outputs set at increasingly large intervals, to test for bias. Statistics can be retrieved from the model, with the mean accuracy given by `acc_mean` and standard deviation as `acc_stdev` or outputted more reader-friendly with the method `get_statistics()`.

### Implementation

### Example
temp = Optimizer(data, params, freqs, constraints, iterations, shuffles)

## Batcher


## Pooler

# How to use
This script searches for singular value decomposition (SVD) models that best describe a given dataset. In particular, SVD is applied to the component frequencies of multiple time series recordings to describe two-choice phenomena that may occur several times within a single recording session, e.g., predicting the results of a behavior testing session that contains 150 trials from time series data of 200 neuron recordings. 

The primary module is found in `optimized/main.py` See details regarding each class and their available methods below.

## Optimizer

### Usage
The optimizer class generates a single model to describe a session `data` with trials specified by `params` with a characteristic band of searched frequencies `freqs`. This model is cross-validated across a specified number of iterations `iterations` with the training and testing trial specification randomized automatically. 

>See **Batcher** for more information on formatting requirements for `data` and `params`

The overall performance of the model is further evaluated on `shuffles` number of shuffled datasets, where the datasets are shuffled and outputs set at increasingly large intervals, to test for bias. Statistics can be retrieved from the model, with the mean accuracy given by `acc_mean` and standard deviation as `acc_stdev` or outputted more reader-friendly with the method `get_statistics()`.

### Implementation
under construction

### Example
We will initialize a model that makes use of the frequency band consisting of 5, 7 and 10 Hz  across 100 cross-validation iterations. The model will be tested for bias with shuffled trials with the following ratio of 0 outputs to 1 outputs: 0:100 20:80, 40:60, 60:40, 80:20, 100:0. 

`example_model = Optimizer(data, params, freqs=[5,7,10], iterations=100, shuffles=5)`

## Batcher
under construction

time | cell 1 | cell 2 | ... | cell n
constraints, start / end times

## Pooler
to be implemented


outdated (will fix soon)

# How to use
This script searches for singular value decomposition (SVD) models that best describe a given dataset. In particular, SVD is applied to the component frequencies of multiple time series recordings to describe two-choice phenomena that may occur several times within a single recording session, e.g., predicting the results of a behavior testing session that contains 150 trials from time series data of 200 neuron recordings. 

The primary module is found in `main.py` See details regarding each class and their available methods below.

## Optimizer

### Usage
The **Optimizer** class generates a single model to describe a session `data` with trials specified by `params` with a characteristic band of searched frequencies `freqs`. This model is cross-validated across a specified number of iterations `iterations` with the training and testing trial specification randomized automatically. 

>See **Batcher** for more information on formatting requirements for `data` and `params`

The overall performance of the model is further evaluated on `shuffles` number of shuffled datasets, where the datasets are shuffled and outputs set at increasingly large intervals, to test for bias. Statistics can be retrieved from the model, with the mean accuracy given by `acc_mean` and standard deviation as `acc_stdev` or outputted more reader-friendly with the method `get_statistics()`.

### Implementation
under construction

### Example
We will initialize a model that makes use of the frequency band consisting of 5, 7 and 10 Hz  across 100 cross-validation iterations. The model will be tested for bias with shuffled trials with the following ratio of 0 outputs to 1 outputs: 0:100 20:80, 40:60, 60:40, 80:20, 100:0. 

`example_model = Optimizer(data, params, freqs=[5,7,10], iterations=100, shuffles=5)`

## Batcher

### Usage
The **Batcher** class is used to search models made by the **Optimizer** class that best describe the data presented. The following search modes are available: "power", "continuous" and "combination".

 | mode | description | searches | 
 |---|---|---|
 |power| The power set of available frequency bands is searched | $$n!$$
 |continuous| The set of continuous frequency bands is searched | $$n^2$$
 |combination| The restriction of the power set to subsets of length *m* is searched | $$_nC_m$$

### Data Formatting
The data `data` should be prepared in a numpy array with the time series data varying along axis 0 and the recording site (e.g. neurons) varying along axis 1. If pulling from a program like excel, spreadsheet columns should be organized like this. (Note that the first row and/or column can be ignored by changing the options `ignoreFirstRow` and `ignoreFirstColumn`)

`cell 1 | cell 2 | ... | cell n`

The trial parameters `params` should also be prepared in a numpy array with individual trials occupying separate rows. The column structure is flexible however, allowing specification later on (again, turn off the first row with `ignoreFirstRow`). For a given trial, the starting time `start_col` should refer to the column of `params` specifying the row of `data` from which the trial begins. The ending time `end_col` follows similarly. Constraints of which trials to be processed can be specified in the hashmap `constraints`.

### Example
We will initialize a search of the power set of available frequency bands where we have the constraints that trials specified in `params` must have their third (index=2) column specify "left" and that the ninth (index=8) column specifies 3. The start and end columns are 6 and 8 (indicies 5 and 7) respectively. The two output classes that the model can predict, i.e., the entries present in the output column, are "cat" and "dog".

`Batch = Batcher(data, params, constraints={2:"left", 8:3}, length, output_classes={"cat":0, "dog":1}, output_column=2, start_col=5, end_col=7)`

## Pooler
to be implemented


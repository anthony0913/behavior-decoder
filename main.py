from model import *
import matplotlib.pyplot as plt
import os

mainLength = 10

data_paths = []
params_paths = []
print("Current Working Directory:", os.getcwd())
for root, dirs, files in os.walk('.'):
    #print("Searching directory:", root)
    for file in files:
        if file.endswith('.csv') and 'calcium_raw_' in file:
            path = os.path.join(root, file)
            data_paths.append(path)
        if file.endswith('.csv') and 'trial_parameters' in file:
            path = os.path.join(root, file)
            params_paths.append(path)


#Runs model
iterations = len(data_paths)
session_info = np.zeros((len(data_paths),2))

for session in range(iterations):
    pooler = Pooler([data_paths[session]], params_paths=[params_paths[session]])
    batch = Batcher(pooler.data, pooler.params, constraints={1: "INIT", 4: "clean"}, length=10,
                    output_classes={"wrong": 0, "correct": 1},output_column=3, start_col=6, end_col=7, folds=10,
                    resamples=10, showDiagnostics=False)
    session_info[session,0] = np.mean(batch.accuracies)
    session_info[session,1] = np.std(batch.accuracies)

print(session_info)
plt.errorbar(np.arange(len(session_info)), session_info[:, 0], yerr=session_info[:, 1]/10, fmt='o')

# set labels and title
plt.xlabel('Accuracy')
plt.ylabel('Index')
plt.title('Accuracy vs Index')
plt.ylim(0,1)

# show the plot
plt.show()

from model import *
mainLength = 10

#Runs model
pool = Pooler(["datasets/data.csv"], ["datasets/params.csv"])
batch = Batcher(pool.data, pool.params, constraints={1:"INIT", 4:"clean"}, length=10, output_classes={"wrong":0, "correct":1},
                output_column=3, start_col=6, end_col=7, folds=10, resamples=10, showDiagnostics=False)

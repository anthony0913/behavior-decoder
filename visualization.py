import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class Clean:
    def __init__(self, params_file):
        self.params_file = params_file
        self.cleaned_params = None
        self.cleaned_params_1 = None
        self.cleaned_params_0 = None

    def clean_params(self):
        cleaned_params = []
        with open(self.params_file, 'r') as params_csv:
            for row in csv.reader(params_csv):
                if row[1] == "FTP" and row[4] == "clean":
                    cleaned_row = [1 if row[3] == "correct" else 0, int(row[6]), int(row[7])]
                    cleaned_params.append(cleaned_row)
        self.cleaned_params = np.array(cleaned_params, dtype=int)

    def separate_cleaned_params(self):
        if self.cleaned_params is None:
            self.clean_params()
        self.cleaned_params_1 = self.cleaned_params[self.cleaned_params[:, 0] == 1]
        self.cleaned_params_0 = self.cleaned_params[self.cleaned_params[:, 0] == 0]

    def get_cleaned_params(self):
        if self.cleaned_params is None:
            self.clean_params()
        return self.cleaned_params

    def get_cleaned_params_1(self):
        if self.cleaned_params_1 is None:
            self.separate_cleaned_params()
        return self.cleaned_params_1

    def get_cleaned_params_0(self):
        if self.cleaned_params_0 is None:
            self.separate_cleaned_params()
        return self.cleaned_params_0


class Transform:
    def __init__(self, session_file):
        self.session_file = session_file
        self.blocks_1 = None
        self.blocks_0 = None

    def get_blocks(self, cleaned_params):
        with open('data.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            rows = [row for row in reader]
        data = np.array(rows, dtype=float)
        blocks = np.zeros(np.shape(data)[1]-1)
        for cleaned_row in cleaned_params:
            block = data[cleaned_row[1]:cleaned_row[2],1:]
            blocks = np.vstack((blocks, np.fft.rfft(block, axis=0).real))
        blocks = blocks[1:,:]
        #print(np.shape(blocks))
        trimmed_blocks = blocks[1:10, :]
        stacked_blocks = np.stack([trimmed_blocks] * cleaned_params.shape[0], axis=2)
        return stacked_blocks

    def generate_heatmaps(self, cleaned_params_1, cleaned_params_0, output_file,
                          normalize=False, threshold=False, inf=0.4, sup=1):
        blocks_1 = self.get_blocks(cleaned_params_1)
        blocks_0 = self.get_blocks(cleaned_params_0)

        print(np.max(np.std(blocks_0, axis=2)), np.max(np.std(blocks_1, axis=2)))

        if normalize:
            min_val = min(np.min(blocks_0),np.min(blocks_1))
            blocks_0 -= min_val
            blocks_1 -= min_val
            max_val = max(np.max(blocks_0),np.max(blocks_1))
            blocks_0 /= max_val
            blocks_1 /= max_val

        print(np.max(blocks_0), np.max(blocks_1))

        if threshold:
            blocks_0[blocks_0 < inf] = 0
            blocks_0[blocks_0 > sup] = 1

            blocks_1[blocks_1 < inf] = 0
            blocks_1[blocks_1 > sup] = 1

        print(np.max(np.std(blocks_0,axis=2)),np.max(np.std(blocks_1,axis=2)))

        with PdfPages(output_file) as pdf:
            num_blocks_1 = blocks_1.shape[2]
            num_blocks_0 = blocks_0.shape[2]
            num_blocks = max(num_blocks_1, num_blocks_0)
            for i in range(num_blocks):
                print(str(i+1) + "/" + str(num_blocks))
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                if i < num_blocks_1:
                    axs[0].imshow(blocks_1[:, :, i], cmap='hot', interpolation='nearest', aspect=10)
                    axs[0].set_title('Block {}'.format(i))
                    axs[0].set_yticks(np.arange(blocks_1.shape[0]), minor=False)
                if i < num_blocks_0:
                    axs[1].imshow(blocks_0[:, :, i], cmap='hot', interpolation='nearest', aspect=10)
                    axs[1].set_title('Block {}'.format(i))
                    axs[1].set_yticks(np.arange(blocks_0.shape[0]), minor=False)
                pdf.savefig(fig)
                plt.close()


clean = Clean('params.csv')
transform = Transform('data.csv')

cleaned_params_1 = clean.get_cleaned_params_1()
cleaned_params_0 = clean.get_cleaned_params_0()

output_file = 'heatmaps.pdf'
transform.generate_heatmaps(cleaned_params_1, cleaned_params_0, output_file)
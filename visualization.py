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
        blocks = []
        with open(self.session_file, 'r') as session_csv:
            session_reader = csv.reader(session_csv)
            next(session_reader)
            for row in session_reader:
                session_col = int(row[0])
                for cleaned_row in cleaned_params:
                    if session_col >= cleaned_row[1] and session_col <= cleaned_row[2]:
                        block = np.array(row[1:], dtype=float)
                        blocks.append(block)
        blocks_ft = np.fft.rfft(blocks, axis=0).real
        trimmed_blocks = blocks_ft[:10, :]
        stacked_blocks = np.stack([trimmed_blocks] * cleaned_params.shape[0], axis=2)
        return stacked_blocks

    def generate_heatmaps(self, cleaned_params_1, cleaned_params_0, output_file):
        blocks_1 = self.get_blocks(cleaned_params_1)
        blocks_0 = self.get_blocks(cleaned_params_0)

        with PdfPages(output_file) as pdf:
            num_blocks_1 = blocks_1.shape[2]
            num_blocks_0 = blocks_0.shape[2]
            num_blocks = max(num_blocks_1, num_blocks_0)
            for i in range(num_blocks):
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                if i < num_blocks_1:
                    axs[0].imshow(blocks_1[:, :, i], cmap='hot', interpolation='nearest', aspect='auto')
                    axs[0].set_title('Block {}'.format(i))
                    axs[0].set_aspect(10.0)
                if i < num_blocks_0:
                    axs[1].imshow(blocks_0[:, :, i], cmap='hot', interpolation='nearest', aspect='auto')
                    axs[1].set_title('Block {}'.format(i))
                    axs[1].set_aspect(10.0)
                pdf.savefig(fig)
                plt.close()


clean = Clean('params.csv')
transform = Transform('data.csv')

cleaned_params_1 = clean.get_cleaned_params_1()
cleaned_params_0 = clean.get_cleaned_params_0()

output_file = 'heatmaps.pdf'
transform.generate_heatmaps(cleaned_params_1, cleaned_params_0, output_file)
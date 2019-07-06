import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
modes = ['12000','4000']

for mod in modes:
    for i in range(17,20):
        outD = "../output/MY_113.MY_115_model%s/MY_113.MY_115.%d"%(mod,i)
        files = os.listdir(outD)
        for afile in files:
            if afile.endswith("npy.enhanced.npy"):
                full_path = outD + "/" + afile
                matrix = np.load(full_path)
                matrix.shape()
                sns.set()
                ax = sns.heatmap(matrix)
                plt.show()



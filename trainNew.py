from __future__ import print_function
from math import log10
import numpy as np
import utils
import os
import gzip
import trainConvNet

print('loading data...')
highres = np.load(gzip.GzipFile('/storage/home/jxw1315/scratch/trainData/GM12878.chr7-8.obs.KR.10k.txthighres.npy.gz', "r")).astype(np.float32)
lowres = np.load(gzip.GzipFile('/storage/home/jxw1315/scratch/trainData/GM12878.chr7-8.obs.KR.10k.txtlowres.npy.gz', "r")).astype(np.float32)
print('finish data loading, start training...')
trainConvNet.train(lowres,highres)







# Author: Yan Zhang  
# Email: zhangyan.cse (@) gmail.com

import argparse
import sys
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import os
import gzip
import model
from torch.utils import data
import torch
import torch.optim as optim
from torch.autograd import Variable
from time import gmtime, strftime
import sys
import torch.nn as nn
import utils
import math
from datetime import datetime

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def is_symmetric(matrix):
    if False in (matrix==np.transpose(matrix)):
        return False
    else:
        return True

startTime = datetime.now()

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_matrix', type=str, required=True, help='Low resolution matrix sample')
parser.add_argument('--compare_matrix', type=str, required=True, help='real matrix')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_dir', type=str, help='where to save the output files')
#parser.add_argument('--scale_factor', type=float, help='factor by which super resolution needed')
parser.add_argument('--chrN', type=int,required=True, help='chromosome number')
parser.add_argument('--delimiter', type=str,default=",", help='contact matrix delimiter')
#parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)
chrN = opt.chrN
input_file = opt.input_matrix
compare_matrix = opt.compare_matrix
output_dir = opt.output_dir


if not output_dir.endswith("/"):
    output_dir += "/"

if not os.path.exists(output_dir):
    os.makedirs( output_dir, 0755 )



use_gpu = 1 #opt.cuda
if use_gpu and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

use_gpu = 0

conv2d1_filters_numbers = 8
conv2d1_filters_size = 9
conv2d2_filters_numbers = 8
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 5


down_sample_ratio = 16
epochs = 10
HiC_max_value = 100


chrs_length = [195471971,182113224,160039680,156508116,151834684,149736546,145441459,129401213,124595110,130694993,122082543,120129022,120421639,124902244,104043685,98207768,94987271,90702639,61431566]


delimiter = opt.delimiter
expRes = 10000
## need to make resolution adjustable.
length = chrs_length[chrN-1]/expRes

# divide the input matrix into sub-matrixes. 


inputMatrix = utils.readFiles(input_file, length + 1, expRes, delimiter)
print("inputMatrix is symmetric?")
print(is_symmetric(inputMatrix))


compareMatrix = utils.readFiles(compare_matrix, length+1, expRes, delimiter)
print("compareMatrix is symmetric?")
print(is_symmetric(compareMatrix))

low_resolution_samples, index = utils.divide(inputMatrix,chrN)

low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)# why use HiC_max_value, in this way, low_resolution_samples will not change.

batch_size = low_resolution_samples.shape[0] #256
# batch_size=256

print("batch_size:",batch_size)

# Reshape the high-quality Hi-C sample as the target value of the training. 
sample_size = low_resolution_samples.shape[-1]
padding = conv2d1_filters_size + conv2d2_filters_size + conv2d3_filters_size - 3
half_padding = padding / 2
output_length = sample_size - padding
#print(output_length)

print(low_resolution_samples.shape)

lowres_set = data.TensorDataset(torch.from_numpy(low_resolution_samples), torch.from_numpy(np.zeros(low_resolution_samples.shape[0])))
lowres_loader = torch.utils.data.DataLoader(lowres_set, batch_size=batch_size, shuffle=False)

hires_loader = lowres_loader

model = model.Net(40, 28)
model.load_state_dict(torch.load(opt.model))
if use_gpu:
    model = model.cuda()

#_loss = nn.MSELoss()


#running_loss = 0.0
#running_loss_validate = 0.0
#reg_loss = 0.0
for i, v1 in enumerate(lowres_loader):
    _lowRes, _ = v1
    _lowRes = Variable(_lowRes).float()
    if use_gpu:
        _lowRes = _lowRes.cuda()
    y_prediction = model(_lowRes)



y_predict = y_prediction.data.cpu().numpy()
print(y_predict.shape)

# recombine samples

length = int(y_predict.shape[2])
y_predict = np.reshape(y_predict, (y_predict.shape[0], length, length))


length = int(math.ceil(chrs_length[chrN-1]*1.0/expRes))

prediction_1 = np.zeros((length, length))


print('predicted sample: ', y_predict.shape, '; index shape is: ', index.shape)
#print(index)
for i in range(0, y_predict.shape[0]):          
    if (int(index[i][1]) != chrN):
        continue
    #print index[i]
    x = int(index[i][2])
    y = int(index[i][3])
    #print np.count_nonzero(y_predict[i])
    prediction_1[x+6:x+34, y+6:y+34] = y_predict[i]

''' Copy upper triangle to lower triangle by Le Huang
'''
pred_triu = np.triu(prediction_1)
pred_lower = np.transpose(pred_triu)
pred = pred_triu + pred_lower
for i in range(length):
    pred[i,i]=0.0

'''Print to see the result by Le Huang
'''
print("prdecition size:")
print(prediction_1.shape)

print("input Matrix size:")
print(inputMatrix.shape)


inputMatrix_ori = output_dir + os.path.basename(input_file) + '.downsample.npy'
enhanced_npy_path = output_dir + os.path.basename(input_file) +'.enhanced.npy'
enhanced_txt_path = output_dir + os.path.basename(input_file) + '.enhanced.txt'
compare_npy_path = output_dir + os.path.basename(opt.compare_matrix) + '.real.npy'
np.save(enhanced_npy_path, prediction_1)
np.save(inputMatrix_ori, inputMatrix)
np.savetxt(enhanced_txt_path,prediction_1, fmt='%d', delimiter=delimiter)
np.save(compare_npy_path, compareMatrix)




''' Caculate precision of HiCPlus Code by Le Huang
'''

mse_predict = mean_squared_error(compareMatrix, prediction_1)/2.0
mae_predict = mean_absolute_error(compareMatrix, prediction_1)/2.0
print(input_file)
print(input_file + '.enhanced.npy')
print(str(mse_predict))
print(str(str(mae_predict)))
print(datetime.now() - startTime) 

if not os.path.exists("../accuracy.txt"):
    fw = open("../accuracy.txt","w")
    fw.write("#Chr Input_filepath\tChr Enhanced_filepath\tChr Real_filepath\tChr Length\tModel\tDuration\tMean Squared Error(MSE)\tMean Absolute Deviation(MAE)\n")
    fw.close()

with open("../accuracy.txt", "a+") as f:
    f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(os.path.basename(opt.input_matrix), enhanced_npy_path, os.path.basename(compare_matrix), length, os.path.basename(opt.model), str(datetime.now() - startTime), str(mse_predict),str(mae_predict)))


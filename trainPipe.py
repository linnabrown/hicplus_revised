from __future__ import print_function
import argparse as ap
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import model
import argparse
import trainConvNet
import numpy as np

chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]
input_resolution = 10000
#chrN = 21
#scale = 16


parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_file', type=str, required=True, help='input training matrix data')
parser.add_argument('--chrN', type=int, required=True, help='chromosome used to train')
#parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--scale_factor', type=float, required=True, help='factor by which resolution needed')

#parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)

#use_cuda = opt.cuda
#if use_cuda and not torch.cuda.is_available():
#    raise Exception("No GPU found, please run without --cuda")

infile = opt.input_file
chrN = opt.chrN
scale = opt.scale_factor

highres = utils.readFiles(infile, chrs_length[chrN-1]/input_resolution + 1, input_resolution)
highres_sub,index = utils.divide(highres,chrN)
print(highres_sub.shape)
np.save(infile+"highres",highres_sub)

lowres = utils.genDownsample(highres,1/float(scale))
lowres_sub,index = utils.divide(lowres,chrN)
print(lowres_sub.shape)
np.save(infile+"lowres",lowres_sub)
#trainConvNet.train(lowres_sub,highres_sub)





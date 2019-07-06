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

chrs_length = [195471971,182113224,160039680,156508116,151834684,149736546,145441459,129401213,124595110,130694993,122082543,120129022,120421639,124902244,104043685,98207768,94987271,90702639,61431566]
input_resolution = 10000
chrN = 19
scale = 16


parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_file', type=str, required=True, help='input training matrix data')
# parser.add_argument('--chrN', type=int, required=True, help='chromosome used to train')
#parser.add_argument('--output_filename', type=str, help='where to save the output image')
# parser.add_argument('--scale_factor', type=float, required=True, help='factor by which resolution needed')

#parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)

#use_cuda = opt.cuda
#if use_cuda and not torch.cuda.is_available():
#    raise Exception("No GPU found, please run without --cuda")

infile = opt.input_file
# chrN = opt.chrN
# scale = opt.scale_factor

highres = utils.readFiles(infile, chrs_length[chrN-1]/input_resolution + 1, input_resolution)
highres_sub,index = utils.divide(highres,chrN)
print(highres_sub.shape)
np.save(infile+"highres",highres_sub)

lowres = utils.genDownsample(highres,1/float(scale))
lowres_sub,index = utils.divide(lowres,chrN)
print(lowres_sub.shape)
np.save(infile+"lowres",lowres_sub)
#trainConvNet.train(lowres_sub,highres_sub)

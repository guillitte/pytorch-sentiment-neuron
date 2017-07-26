import os
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import models
import argparse
import time
import math


parser = argparse.ArgumentParser(description='load_from_numpy.py')

parser.add_argument('-save_model', default='mlstm-ns.pt',
                    help="""Model filename to save""")
parser.add_argument('-load_model', default='',
                    help="""Model filename to load""")
parser.add_argument('-train', default='data/input.txt',
                    help="""Text filename for training""")
parser.add_argument('-valid', default='data/valid.txt',
                    help="""Text filename for validation""")                    
parser.add_argument('-rnn_type', default='mlstm',
                    help='mlstm, lstm or gru')
parser.add_argument('-layers', type=int, default=1,
                    help='Number of layers in the encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=4096,
                    help='Size of hidden states')
parser.add_argument('-embed_size', type=int, default=128,
                    help='Size of embeddings')
parser.add_argument('-seq_length', type=int, default=20,
                    help="Maximum sequence length")
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-learning_rate', type=float, default=0.001,
                    help="""Starting learning rate.""")
parser.add_argument('-dropout', type=float, default=0.0,
                    help='Dropout probability.')
parser.add_argument('-param_init', type=float, default=0.05,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-clip', type=float, default=5,
                    help="""Clip gradients at this value.""")
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')   
# GPU
parser.add_argument('-cuda', action='store_true',
                    help="Use CUDA")
               

opt = parser.parse_args()   

embed = nn.Embedding(256, opt.embed_size)
rnn = models.StackedLSTM(models.mLSTM, opt.layers, opt.embed_size, opt.rnn_size, 256, opt.dropout)

embed.weight.data = torch.from_numpy(np.load("weights/embd.npy"))
rnn.h2o.weight.data = torch.from_numpy(np.load("weights/w.npy")).t()
rnn.h2o.bias.data = torch.from_numpy(np.load("weights/b.npy"))
rnn.layers[0].wx.weight.data = torch.from_numpy(np.load("weights/wx.npy")).t()
rnn.layers[0].wh.weight.data = torch.from_numpy(np.load("weights/wh.npy")).t()
rnn.layers[0].wh.bias.data = torch.from_numpy(np.load("weights/b0.npy"))
rnn.layers[0].wmx.weight.data = torch.from_numpy(np.load("weights/wmx.npy")).t()
rnn.layers[0].wmh.weight.data = torch.from_numpy(np.load("weights/wmh.npy")).t()
checkpoint = {
    'rnn': rnn,
    'embed': embed,
    'opt': opt,
    'epoch': 0
    }
save_file = opt.save_model
print('Saving to '+ save_file)
torch.save(checkpoint, save_file)

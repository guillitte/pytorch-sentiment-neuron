import os
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import argparse
import models
import math

parser = argparse.ArgumentParser(description='sample.py')

parser.add_argument('-init', default='The meaning of life is ',
                    help="""Initial text """)
parser.add_argument('-load_model', default='',
                    help="""Model filename to load""")                   
parser.add_argument('-seq_length', type=int, default=50,
                    help="""Maximum sequence length""")
parser.add_argument('-temperature', type=float, default=0.4,
                    help="""Temperature for sampling.""")
parser.add_argument('-neuron', type=int, default=0,
                    help="""Neuron to read.""")
parser.add_argument('-overwrite', type=float, default=0,
                    help="""Value used to overwrite the neuron. 0 means don't overwrite.""")
parser.add_argument('-layer', type=int, default=-1,
                    help="""Layer to read. -1 = last layer""")
# GPU
parser.add_argument('-cuda', action='store_true',
                    help="""Use CUDA""")
                    
opt = parser.parse_args()    


def batchify(data, bsz):
    tokens = len(data.encode())
    ids = torch.LongTensor(tokens)
    token = 0
    for char in data.encode():
        ids[token] = char
        token += 1
    nbatch = ids.size(0) // bsz
    ids = ids.narrow(0, 0, nbatch * bsz)
    ids = ids.view(bsz, -1).t().contiguous()
    return ids        


def color(p):
	p = math.tanh(3*p)*.5+.5
	q = 1.-p*1.3
	r = 1.-abs(0.5-p)*1.3+.3*q
	p=1.3*p-.3
	i = int(p*255)	
	j = int(q*255)
	k = int(r*255)
	if j<0:
		j=0	
	if k<0:
		k=0
	if k >255:
		k=255
	if i<0:
		i = 0
	return ('\033[38;2;%d;%d;%dm' % (j, k, i)).encode()



batch_size = 1

checkpoint = torch.load(opt.load_model)
embed = checkpoint['embed']
rnn = checkpoint['rnn']

loss_fn = nn.CrossEntropyLoss() 


text = batchify(opt.init, batch_size)

def make_cuda(state):
    if isinstance(state, tuple):
    	return (state[0].cuda(), state[1].cuda())
    else:
    	return state.cuda()

batch = Variable(text) 
states = rnn.state0(batch_size)
if isinstance(states, tuple):
	hidden, cell = states
else:
	hidden = states
last = hidden.size(0)-1
if opt.layer <= last and opt.layer >= 0:
	last = opt.layer

if opt.cuda:
    batch =batch.cuda()
    states = make_cuda(states)
    embed.cuda()
    rnn.cuda()
    
loss_avg = 0
loss = 0
gen = bytearray()
for t in range(text.size(0)):                           
    emb = embed(batch[t])
    ni = (batch[t]).data[0]
    states, output = rnn(emb, states)
    if isinstance(states, tuple):
        hidden, cell = states
    else:
        hidden = states
    feat = hidden.data[last,0,opt.neuron]
    if ni< 128:
        col = color(feat)
        gen+=(col)
    gen.append(ni)        	

print(opt.init)        	


if opt.temperature == 0:
    topv, topi = output.data.topk(1)
    ni = topi[0][0]
    gen.append(ni)
    inp = Variable(topi[0], volatile=True)
    if opt.cuda:
        inp = inp.cuda()  

    for t in range(opt.seq_length):
        
        emb = embed(inp)
        states, output = rnn(emb, states)        
        topv, topi = output.data.topk(1)
        ni = topi[0][0]
        gen.append(ni)
        inp = Variable(topi[0])
        if opt.cuda:
            inp = inp.cuda() 

else:
    probs = F.softmax(output[0].squeeze().div(opt.temperature)).data.cpu()
    ni = torch.multinomial(probs,1)[0]
    feat = hidden.data[last,0,opt.neuron]
    
    if ni < 128:
    	col = color(feat)
    	gen+=(col)
    	
    gen.append(ni)	
    
    
    inp = Variable(torch.LongTensor([ni]), volatile=True) 
    if opt.cuda:
        inp = inp.cuda()      
    for t in range(opt.seq_length):
        
        emb = embed(inp)
        states, output = rnn(emb, states)
        if isinstance(states, tuple):
        	hidden, cell = states
        else:
        	hidden = states
        feat = hidden.data[last,0,opt.neuron] 
        if isinstance(output, list): 
        	output =output[0]      
        probs = F.softmax(output.squeeze().div(opt.temperature)).data.cpu()
        ni = torch.multinomial(probs,1)[0]
        
        if ni< 128:
        	col = color(feat)
        	gen+=(col)
        gen.append(ni)
        inp = Variable(torch.LongTensor([ni]))
        if opt.cuda:
            inp = inp.cuda() 
        if opt.overwrite != 0: 
        	hidden.data[last,0,opt.neuron] = opt.overwrite

gen+=('\033[0m').encode()

print(gen.decode("utf-8",errors = 'ignore' ))

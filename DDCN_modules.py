import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable, Function
import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore")

def dilate(x, dilation, d_i, init_dilation=1, pad_start=True):

    [n, c, l] = x.size()

    dilation_factor = dilation / init_dilation
    if dilation_factor == 1:
        return x


    new_l = (torch.ceil(l / dilation_factor) * dilation_factor).int()
    if new_l != l:
        l = new_l
        x = constant_pad_1d(x, new_l, dimension=2, pad_start=pad_start)

    l_old = (torch.round(l / dilation_factor)).int()
    n_old = (torch.round(n * dilation_factor)).int()
    l = torch.ceil(l * init_dilation / dilation)
    n = torch.ceil(n * dilation / init_dilation)

    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    
    l = l.cpu().detach().numpy()
    n = n.cpu().detach().numpy() 
    
    [c1, l1, n1] = x.size()
    factor1 = c1 * l1 * n1
    factor2 = c *int(l) * int(n)

    f = factor2 - factor1

    if (f >= 0):
        n_d = n - f/(c*l)
        l_d = l - f/(n*c) 

    
    if(d_i <= 7):
    
        x = x.view(c, int(l), int(n))
    
    elif (d_i > 7 and n != 1):
        
        x = x.view(c, int(l), int(n_d))
          
    elif (d_i > 7 and n == 1):
        
        x = x.view(c, int(l_d), int(n))

    
#############dilated network################
    
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x


class DilatedQueue:
    def __init__(self, max_length, data=None, dilation=1, num_deq=1, num_channels=1, dtype=torch.FloatTensor):
        self.in_pos = 0
        self.out_pos = 0
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dilation = dilation
        self.max_length = max_length
        self.data = data
        self.dtype = dtype
        if data == None:
            self.data = Variable(dtype(num_channels, max_length).zero_())

    def enqueue(self, input):
        self.data[:, self.in_pos] = input
        self.in_pos = (self.in_pos + 1) % self.max_length

    def dequeue(self, num_deq=1, dilation=1):

        start = self.out_pos - ((num_deq - 1) * dilation)
        if start < 0:
            t1 = self.data[:, start::dilation]
            t2 = self.data[:, self.out_pos % dilation:self.out_pos + 1:dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:self.out_pos + 1:dilation]

        self.out_pos = (self.out_pos + 1) % self.max_length
        return t

    def reset(self):
        self.data = Variable(self.dtype(self.num_channels, self.max_length).zero_())
        self.in_pos = 0
        self.out_pos = 0


class ConstantPad1d(Function):
    def __init__(self, target_size, dimension=0, value=0, pad_start=False):
        super(ConstantPad1d, self).__init__()
        self.target_size = target_size
        self.dimension = dimension
        self.value = value
        self.pad_start = pad_start

    def forward(self, input):
        self.num_pad = self.target_size - input.size(self.dimension)
        assert self.num_pad >= 0, 'target size has to be greater than input size'

        self.input_size = input.size()

        size = list(input.size())
        size[self.dimension] = self.target_size
        output = input.new(*tuple(size)).fill_(self.value)
        c_output = output

        if self.pad_start:
            c_output = c_output.narrow(self.dimension, self.num_pad, c_output.size(self.dimension) - self.num_pad)
        else:
            c_output = c_output.narrow(self.dimension, 0, c_output.size(self.dimension) - self.num_pad)

        c_output.copy_(input)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new(*self.input_size).zero_()
        cg_output = grad_output

        # crop grad_output
        if self.pad_start:
            cg_output = cg_output.narrow(self.dimension, self.num_pad, cg_output.size(self.dimension) - self.num_pad)
        else:
            cg_output = cg_output.narrow(self.dimension, 0, cg_output.size(self.dimension) - self.num_pad)

        grad_input.copy_(cg_output)
        return grad_input


def constant_pad_1d(input,
                    target_size,
                    dimension=0,
                    value=0,
                    pad_start=False):
    return ConstantPad1d(target_size, dimension, value, pad_start).forward(input)

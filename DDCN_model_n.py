import os
import os.path
import time
from DDCN_modules import *
import torch.optim as optim
import torch as tc
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class DDCNModel(nn.Module):

    def __init__(self,
                 layers=10,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 end_channels=256,
                 inputF=75,
                 output_length=32,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=False):

        super(DDCNModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.inputF = inputF
        self.kernel_size = kernel_size
        self.dtype = dtype

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = []
        # self.main_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.count = 0
        
        self.offset_layer = torch.nn.Linear(10, 10)
        #######################################################

        self.offset = torch.tensor(np.ones(10), dtype=torch.float32,
                                                      device='cuda', requires_grad=True)
        
        self.offset8 = torch.tensor(1.0, dtype=torch.float32, 
                                                      device='cuda', requires_grad=True)
        
        self.offset9 = torch.tensor(1.0, dtype=torch.float32, 
                                                      device='cuda', requires_grad=True)
        
        self.offset10 = torch.tensor(1.0, dtype=torch.float32, 
                                                      device='cuda', requires_grad=True)
        
        self.w_i = torch.tensor(1.0, dtype=torch.float32, 
                                                      device='cuda', requires_grad=True)
        
        self.w_f8 = torch.tensor(1.0, dtype=torch.float32, 
                                                      device='cuda', requires_grad=True)
        
        self.w_f9 = torch.tensor(1.0, dtype=torch.float32, 
                                                      device='cuda', requires_grad=True)
        
        self.w_f10 = torch.tensor(1.0, dtype=torch.float32, 
                                                      device='cuda', requires_grad=True)
        
        self.offset_init = torch.tensor(0.0, dtype=torch.float32, 
                                                      device='cuda', requires_grad=True)
        self.dila = torch.tensor(256, dtype=torch.float32, 
                                                      device='cuda', requires_grad=True)       

        self.dila_static = torch.tensor(256, dtype=torch.float32, 
                                                      device='cuda', requires_grad=False)
        
        self.init_dila = torch.tensor(128, dtype=torch.float32, 
                                                      device='cuda', requires_grad=True)
        
        self.start_conv = nn.Conv1d(in_channels=self.inputF,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)


        for b in range(blocks):
            additional_scope = kernel_size - 1
            
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated queues for fast generation
                self.dilated_queues.append(DilatedQueue(max_length=(kernel_size - 1) * new_dilation + 1,
                                                        num_channels=residual_channels,
                                                        dilation=new_dilation,
                                                        dtype=dtype))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   bias=bias))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2
                

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=1,
                                  bias=True)

        self.end_conv_offset = nn.Conv1d(in_channels=10,
                                  out_channels=end_channels,
                                  kernel_size=1,
                                  bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=10,
                                    kernel_size=1,
                                    bias=True)

        self.output_length = output_length
        self.receptive_field = receptive_field
        

    def DDCN(self, input, offset, d_loss, dilation_func):
        
        x = self.start_conv(input)

        skip = 0

        for i in range(10):
            
            (dilation, init_dilation) = self.dilations[i]

            if (i == 8):
                self.dila.data  = torch.tensor(dilation, dtype=torch.float32, 
                                          device='cuda', requires_grad=True) + 5*self.offset[7]
            if (i == 9):
                self.dila.data  = torch.tensor(dilation, dtype=torch.float32, 
                                          device='cuda', requires_grad=True) + 5*self.offset[8]
            if (i == 10):
                self.dila.data  = torch.tensor(dilation, dtype=torch.float32, 
                                          device='cuda', requires_grad=True) + 5*self.offset[9]
            else:
                 self.dila.data  = torch.tensor(dilation, dtype=torch.float32, 
                                          device='cuda', requires_grad=True)  
            
            self.dila_static.data  = torch.tensor(dilation, dtype=torch.float32, 
                                      device='cuda', requires_grad=True) 
            
            if (i > 7):
            
                self.init_dila.data  = torch.tensor(init_dilation, dtype=torch.float32, 
                                     device='cuda', requires_grad=True) + self.offset_init
            else:
                self.init_dila.data  = torch.tensor(init_dilation, dtype=torch.float32, 
                                     device='cuda', requires_grad=True)
                

            residual = dilation_func(x, self.dila, self.init_dila, i)

            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = F.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = F.sigmoid(gate)
            x = filter * gate

            s = x
            if x.size(2) != 1:

                 s = dilate(x, 1, i, init_dilation=self.dila)
            s = self.skip_convs[i](s)

            try:
                skip = skip[:, :, -s.size(2):]

            except:
                skip = 0
                
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, (self.kernel_size - 1):]
            
            
            if (i == 8):
                self.offset8.backward()

                self.w_f8 = self.w_f8 - 0.001 * self.offset8.grad

                self.offset8.data = 1 / (1 + tc.exp(-self.offset8.reshape([1]).
                                                    dot(self.w_f8.reshape([1]))))
                self.offset[7] = self.offset[7] + self.offset8

            if (i == 9):
                self.offset9.backward()

                self.w_f9 = self.w_f9 - 0.001 * self.offset9.grad

                self.offset9.data = 1 / (1 + tc.exp(-self.offset9.reshape([1]).
                                                    dot(self.w_f9.reshape([1]))))

                self.offset[8] = self.offset[8] + self.w_f9

            if (i == 10):
                self.offset10.backward()

                self.w_f10 = self.w_f10 - 0.001 * self.offset10.grad


                self.offset10.data = 1 / (1 + tc.exp(-self.offset10.reshape([1]).
                                                     dot(self.w_f10.reshape([1]))))

                self.offset[9] = self.offset[9] + self.w_f10

            self.offset = self.offset_layer(self.offset)


        x = F.relu(skip)

        x = F.relu(self.end_conv_1(x))
        x_d = x.clone()
        offset_features = self.end_conv_offset(self.offset.repeat(x.shape[0]).reshape(x.shape[0], 10, 1))
        x_d = x + offset_features

        x = self.end_conv_2(x)

        return x
        

    def DDCN_dilate(self, input, dilation, init_dilation, i):
        x = dilate(input, dilation, i, init_dilation)
        return x

    def queue_dilate(self, input, dilation, init_dilation, i):
        queue = self.dilated_queues[i]
        queue.enqueue(input.data[0])
        x = queue.dequeue(num_deq=self.kernel_size,
                          dilation=dilation)
        x = x.unsqueeze(0)

        return x


    def forward(self, input, offset, loss):
        x = self.DDCN(input, offset, loss,
                         dilation_func=self.DDCN_dilate)

        return x


    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def cpu(self, type=torch.FloatTensor):
        self.dtype = type
        for q in self.dilated_queues:
            q.dtype = self.dtype
        super().cpu()


def load_latest_model_from(location, use_cuda=True):
    files = [location + "/" + f for f in os.listdir(location)]
    newest_file = max(files, key=os.path.getctime)
    print("load model " + newest_file)

    if use_cuda:
        model = torch.load(newest_file)
    else:
        model = load_to_cpu(newest_file)

    return model


def load_to_cpu(path):
    model = torch.load(path, map_location=lambda storage, loc: storage)
    model.cpu()
    return model

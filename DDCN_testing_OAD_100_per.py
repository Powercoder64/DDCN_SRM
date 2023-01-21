import torch
import torch.optim as optim
import torch.utils.data
from DDCN_modules import *
import numpy as np
import warnings
from copy import deepcopy

warnings.filterwarnings("ignore")


def print_last_loss(opt):
    print("loss: ", opt.losses[-1])


def print_last_validation_result(opt):
    print("validation loss: ", opt.validation_results[-1])



class DDCNTester_100:
    def __init__(self,
                 model,
                 optimizer=optim.Adam,
                 lr=0.001,
                 weight_decay=0,
                 gradient_clipping=None,
                  snapshot_path=None,
                 snapshot_name='snapshot',
                 snapshot_interval=1000,
                 dtype=torch.FloatTensor,
                 ltype=torch.LongTensor):
        self.model = model
        self.dataloader = None
        self.snapshot_path = snapshot_path
        self.snapshot_name = snapshot_name
        self.snapshot_interval = snapshot_interval
        self.dtype = dtype
        self.ltype = ltype
        self.model_path = '.'
        self.dataset_d = '.'
        self.label_d = '.'

    def test(self,
             ):
        outputs = []
        targets = []

        model_path = self.model_path
        total_steps = 0
        self.model.load_state_dict(torch.load(model_path))
        for x, target in self.infinite_batch():
            x = gpu(x).transpose(1, 2)

            output = self.model(x, 0, 0)

            output = output.transpose(1, 2)

            output = output.squeeze()

            output = output.data.cpu().numpy()

            target = target[(target.shape[0] - output.shape[0]): target.shape[0]]

            total_steps += 1

            nn = 20
            #print('video sample number ' + str(total_steps - 1) + ' out of ' + str(nn))
            if (total_steps > nn):
                break

            outputs.append(output)
            targets.append(target)

        outputs = np.array(outputs)
        targets = np.array(targets)

        return outputs, targets

    def infinite_batch(self):
        dataset = self.dataset_d

        labels = self.label_d
        while True:
            for d, l in zip(dataset, labels):
                yield np.array(d).reshape([1, len(d), 75]), np.array(l)


def gpu(data):
    tensor = torch.from_numpy(data).float()

    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor.cuda())
    else:
        return torch.autograd.Variable(tensor)


def gpu_int(data):
    tensor = torch.from_numpy(np.array((data))).int()

    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor.cuda())
    else:
        return torch.autograd.Variable(tensor)
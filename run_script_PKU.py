from DDCN_model_p import *
from DDCN_testing_PKU_100_per import *
from DDCN_testing_PKU_50_per import *
from DDCN_testing_PKU_10_per import *
import numpy as np
from copy import deepcopy
import sklearn.metrics


############  PATHS ############### please fix the paths

dataset_path = '/home/acw6ze/DDCN/data/PKU/data_PKU.npy' # path for data
label_path = '/home/acw6ze/DDCN/data/PKU/labels_PKU.npy' # path for labels
model_path = '/home/acw6ze/DDCN/model/model_params_PKU.pkl' # path for model


############ loading data ##############

dataset_d = np.load(dataset_path, allow_pickle=True)
label_d = np.load(label_path, allow_pickle=True)
###################################################

dtype = torch.FloatTensor
ltype = torch.LongTensor

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor


def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom


##################### Testing on Observation Ratio = 100% ######################

model_100 = DDCNModel(layers=10,
                     blocks=3,
                     dilation_channels=32,
                     residual_channels=32,
                     skip_channels=1024,
                     end_channels=512,
                     output_length=16,
                     dtype=dtype,
                     bias=True)


if use_cuda:
    model_100.cuda()
model_100.receptive_field = 30

tester_100 = DDCNTester_100(model=model_100, # initializing DDCN model
                         lr=0.0001,
                         weight_decay=0.0,
                         snapshot_path='snapshots',
                         snapshot_name='chaconne_model',
                         snapshot_interval=1000,
                         dtype=dtype,
                         ltype=ltype,
                        )
tester_100.model_path = model_path
tester_100.dataset_d = dataset_d
tester_100.label_d = label_d

print('start testing DDCN on Observation Ratio = 100%...')
result_100, label_100 = tester_100.test() # loading DDCN model

result_100 = np.concatenate(result_100)

label_100 = np.concatenate(label_100)

labels_100 = []
results_100 = []

for i in range(0, label_100.shape[0]):
    if (label_100[i] != -1):
        labels_100.append(label_100[i])
        results_100.append(result_100[i])

labels_100 = np.array(labels_100)

results_100 = np.array(results_100)

results_class_100 = []

for i in range(0, len(results_100)):
    results_class_100.append(list(results_100[i]).index(max(results_100[i])))
results_class_100 = np.array(results_class_100)

f1_100 = sklearn.metrics.f1_score(labels_100, results_class_100, average='micro')



##################### Testing on Observation Ratio = 50% ######################

model_50 = DDCNModel(layers=10,
                     blocks=3,
                     dilation_channels=32,
                     residual_channels=32,
                     skip_channels=1024,
                     end_channels=512,
                     output_length=16,
                     dtype=dtype,
                     bias=True)


if use_cuda:
    model_50.cuda()
model_50.receptive_field = 30

tester_50 = DDCNTester_50(model=model_100,
                         lr=0.0001,
                         weight_decay=0.0,
                         snapshot_path='snapshots',
                         snapshot_name='chaconne_model',
                         snapshot_interval=1000,
                         dtype=dtype,
                         ltype=ltype,
                        )
tester_50.model_path = model_path
tester_50.dataset_d = dataset_d
tester_50.label_d = label_d

print('start testing DDCN on Observation Ratio = 50%...')

result_50, label_50 = tester_50.test()

result_50 = np.concatenate(result_50)

label_50= np.concatenate(label_50)

labels_50 = []
results_50 = []

for i in range(0, label_50.shape[0]):
    if (label_50[i] != -1):
        labels_50.append(label_50[i])
        results_50.append(result_50[i])

labels_50 = np.array(labels_50)

results_50 = np.array(results_50)

results_class_50 = []

for i in range(0, len(results_50)):
    results_class_50.append(list(results_50[i]).index(max(results_50[i])))
results_class_50 = np.array(results_class_50)

f1_50 = sklearn.metrics.f1_score(labels_50, results_class_50, average='micro')


##################### Testing on Observation Ratio = 10% ######################

model_10 = DDCNModel(layers=10,
                     blocks=3,
                     dilation_channels=32,
                     residual_channels=32,
                     skip_channels=1024,
                     end_channels=512,
                     output_length=16,
                     dtype=dtype,
                     bias=True)


if use_cuda:
    model_10.cuda()
model_10.receptive_field = 30

tester_10 = DDCNTester_10(model=model_10,
                         lr=0.0001,
                         weight_decay=0.0,
                         snapshot_path='snapshots',
                         snapshot_name='chaconne_model',
                         snapshot_interval=1000,
                         dtype=dtype,
                         ltype=ltype,
                        )
tester_10.model_path = model_path
tester_10.dataset_d = dataset_d
tester_10.label_d = label_d

print('start testing DDCN on Observation Ratio = 10%...')
result_10, label_10 = tester_10.test()

result_10 = np.concatenate(result_10)

label_10 = np.concatenate(label_10)

labels_10 = []
results_10 = []

for i in range(0, label_10.shape[0]):
    if (label_10[i] != -1):
        labels_10.append(label_10[i])
        results_10.append(result_10[i])

labels_10 = np.array(labels_10)

results_10 = np.array(results_10)

results_class_10 = []

for i in range(0, len(results_10)):
    results_class_10.append(list(results_10[i]).index(max(results_10[i])))
results_class_10 = np.array(results_class_10)

f1_10 = sklearn.metrics.f1_score(labels_10, results_class_10, average='micro')



################ printing the results #################################

print( 'F1 score for DDCN on OR 100%  = ' + str(round(f1_100*100, 2)) + '%')
print( 'F1 score for DDCN on OR 50%  = ' + str(round(f1_50*100, 2)) + '%')
print( 'F1 score for DDCN on OR 10%  = ' + str(round(f1_10*100, 2)) + '%')

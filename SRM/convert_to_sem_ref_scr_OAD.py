from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
import numpy as np
import random
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.datasets.synthetic import generate_sequential
import torch

array = np.load('off_sem_ref_attr_OAD', allow_pickle=True) # add the path for offline semantic reference attributes from the previous step
objects = np.load('sem_lst_OAD.npy', allow_pickle=True) # add the path for the provided list of semantics

rating = []
user = []
item = []

for i in range(0, len(array)):
    # print (i)
    for j in range(0, len(array[i]) - 1):
        if (array[i].ndim != 1):
            # rating.append(array[i][:, 1][1:len(array[i])][j])
            rating.append(float(array[i][:, 1][1:len(array[i])][j]) * float(array[i][:, 2][1:len(array[i])][j]) *
                          float(array[i][:, 3][1:len(array[i])][j]))  ##[:, 2] is the dimesion, movment

for i in range(0, len(array)):
    # print (i)
    for j in range(0, len(array[i]) - 1):
        if (array[i].ndim != 1):
            user.append(array[i][:, 6][1:len(array[i])][j])

for i in range(0, len(array)):
    # print (i)
    for j in range(0, len(array[i]) - 1):
        if (array[i].ndim != 1):
            item.append(np.where(array[i][:, 0][1:len(array[i])][j] == objects)[0][0])


num_class = 10
dataset = get_movielens_dataset(variant='100K')


item_L =[]
rating_L=[]
user_L = []
for i in range(0, len(item)):
    if (user[i] != -1):
        item_L.append(item[i])
        user_L.append(user[i])
        rating_L.append(rating[i])
item = np.array(item_L)      
rating = np.array(rating_L) 
user = np.array(user_L) 
        
        
dataset.item_ids = item
dataset.ratings = rating
dataset.user_ids = user
dataset.num_items = len(objects)
dataset.num_users = num_class


#model = ImplicitFactorizationModel(n_iter=250, loss='bpr')


RANDOM_SEED = 42
LATENT_DIM = 32
NUM_EPOCHS = 200
BATCH_SIZE = 256
L2 = 1e-6
LEARNING_RATE = 1e-3


model = ImplicitFactorizationModel(loss='bpr',
                                                      embedding_dim=LATENT_DIM,
                                                      n_iter=NUM_EPOCHS,
                                                      learning_rate=LEARNING_RATE,
                                                      batch_size=BATCH_SIZE,
                                                      l2=L2,
                                                      random_state=np.random.RandomState(RANDOM_SEED))


# model = ExplicitFactorizationModel(loss='regression',
#                                                      embedding_dim=LATENT_DIM,
#                                                      n_iter=NUM_EPOCHS,
#                                                      learning_rate=LEARNING_RATE,
#                                                      batch_size=BATCH_SIZE,
#                                                      l2=L2,
#                                                      random_state=np.random.RandomState(RANDOM_SEED))


#=============================================================================
# model = ExplicitFactorizationModel(loss='regression',
#                                     embedding_dim=128,  # latent dimensionality
#                                     n_iter=200,  # number of epochs of training
#                                     batch_size=1024,  # minibatch size
#                                     l2=1e-9,  # strength of L2 regularization
#                                     learning_rate=1e-3,
#                                     use_cuda=torch.cuda.is_available())
#=============================================================================

dataset = dataset
model.fit(dataset)

predict = []

for i in range(0, num_class):
    predict.append(model.predict(i))

predict = np.array(predict)
labels = []

for i in range(0, num_class):
    ind=predict[i].argsort()[-10:][::-1]
    labels.append(objects[ind])
    

labels = np.array(labels)

np.save('sem_ref_scr_OAD', predict)


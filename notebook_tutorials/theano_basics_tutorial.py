
# coding: utf-8

# 
# # Linear Regression in Theano

# https://www.youtube.com/watch?v=S75EdAcXHKk

# In[70]:

from __future__ import division

get_ipython().magic(u'matplotlib inline')

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import mpld3
mpld3.enable_notebook()


# In[71]:

import theano
from theano import tensor


# ## Load MNIST Data

# In[72]:

from os import path
data_path = '/home/andershuss/data/theano_data/mnist/train/'
features = np.load(path.join(data_path, 'features.npy'))
targets = np.load(path.join(data_path, 'targets.npy'))


# In[73]:

features.shape


# In[74]:

targets.shape


# In[75]:

n_samples = targets.shape[0]
n_classes = len(np.unique(targets))
n_features = 28 * 28


# ### Flatten and normalize X

# In[76]:

training_X = features.reshape(n_samples, -1) / 255.0  # normalize!


# ### One-hot-encode y

# In[77]:

targets[:5]


# In[78]:

training_y = targets.flatten()
training_y_one_hot = np.zeros((targets.shape[0], 10))
training_y_one_hot[np.arange(n_samples), targets.flatten()] = 1.0
training_y_one_hot[:5]


# In[79]:

def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)


# In[80]:

def init_param(shape):
    std = 0.01
    param = theano.shared(
        floatX(
            np.random.randn(*shape) * std
        )
    )
    return param


# In[81]:

def model(X, W):
    y_prob_hat = tensor.nnet.softmax(
        tensor.dot(X, W)
    )
    return y_prob_hat


# In[82]:

X = tensor.fmatrix('X')
y_one_hot = tensor.fmatrix('y')


# In[83]:

W = init_param(
    shape=(
        training_X.shape[1],
        10
    )
)  # shape = (28 x 28, 10) = (n_features, n_classes) 


# In[84]:

y_prob_hat = model(X, W)
y_hat = tensor.argmax(y_prob_hat, axis=1)


# In[85]:

cost = tensor.mean(
    tensor.nnet.categorical_crossentropy(
        y_prob_hat,
        y_one_hot
    )
)


# In[86]:

gradient = tensor.grad(cost, wrt=W)


# In[87]:

learning_rate = 0.05
updates = [[W, W - gradient * learning_rate]]


# In[88]:

train_one_batch = theano.function(
    inputs=[X, y_one_hot],
    outputs=cost,
    updates=updates,
    allow_input_downcast=True
)


# In[89]:

def train_one_epoch(batch_size=100):
    cost_acc = 0.0
    n_processed_batches = 0
    for start, end in zip(
        range(0, n_samples, batch_size),
        range(batch_size, n_samples, batch_size)
    ):
        batch_cost = train_one_batch(
            training_X[start:end, :],
            training_y_one_hot[start:end, :]
        )
        cost_acc += batch_cost
        n_processed_batches += 1
    
    epoch_mean_cost = cost_acc / n_processed_batches
    
    return epoch_mean_cost
    


# In[90]:

n_epochs = 100
epoch_costs = []
for i in range(n_epochs):
    epoch_cost = train_one_epoch()
    epoch_costs.append(
        epoch_cost
    )
    print(epoch_cost)


# In[91]:

plt.plot(epoch_costs)


# In[92]:

y_hat_evaluate = theano.function(inputs=[X], outputs=y_hat, allow_input_downcast=True)


# In[93]:

y_pred = y_hat_evaluate(training_X)


# In[94]:

missclass = (training_y != y_pred).sum() / n_samples


# In[95]:

missclass


# In[ ]:




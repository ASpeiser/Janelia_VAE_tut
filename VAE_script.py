# -*- coding: utf-8 -*-
"""
Created on Sun Nov 01 11:59:20 2015

@author: User
"""

import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import functions
from lasagne.updates import get_or_compute_grads
from lasagne.updates import adagrad
from collections import OrderedDict
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import time
from scipy.stats import norm
import pickle
import os.path

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
    
def init_weights(shape, scale = 0.01, name = None):
    return theano.shared(floatX(np.random.randn(*shape) * scale), name = name)
    
def init_biases(shape, scale = 0.01, name = None):
    return theano.shared(floatX(np.random.randn(shape,1)* scale), name = name, broadcastable=(False, True))
    
def model(x, W1, W2, W3, W4, W5, b1, b2, b3, b4, b5, batch_size):

    eps = theano.shared(floatX(np.random.randn(dimZ, batch_size)))
    
    h_enc =         T.tanh(T.dot(W1,x) + b1)
    mu_enc =        T.dot(W2,h_enc) + b2
    log_sig_enc =   0.5*(T.dot(W3,h_enc) + b3)
    z =             mu_enc + T.exp(log_sig_enc)*eps
    
    prior = 0.5 *   T.sum(1 + 2*log_sig_enc - mu_enc**2 - T.exp(2*log_sig_enc)) 
    
    h_dec =         T.tanh(T.dot(W4,z) + b4)
    y =             T.nnet.sigmoid(T.dot(W5,h_dec) + b5)
    
    logpxz =        T.sum(x*T.log(y) + (1 - x)*T.log(1 - y))
    
    bound = (prior + logpxz)/batch_size
    
    return bound
        
def latent_rec(z, W4, W5, b4, b5):

    h_dec =         T.tanh(T.dot(W4,z) + b4)
    y =             T.nnet.sigmoid(T.dot(W5,h_dec) + b5)
    
    return y

def adada(loss_or_grads, params, factor, learning_rate=1.0, epsilon=1e-6):

    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2
        updates[accu] = accu_new
        
        if param.name.count('W') == 1:
            updates[param] = param - (learning_rate * (grad + 0.5*factor*param) / T.sqrt(accu_new + epsilon))
        else:
            updates[param] = param - (learning_rate * (grad / T.sqrt(accu_new + epsilon)))

    return updates

HU_enc = 500
HU_dec = 500
dimZ   = 3
b_size = 100
epochs = 4000


#datasets = functions.load_data_np('/home/artur/Dropbox/BONN/datasets/mnist.pkl')
datasets = functions.load_data_np(os.path.abspath('../../datasets/mnist.pkl'))
train_set_x, _ = datasets[0]
valid_set_x, _ = datasets[1]

N = train_set_x.shape[0]

X = T.fmatrix()
Z = T.fmatrix()

W1 = init_weights((HU_enc, 28*28), name = 'W1')
b1 = init_biases((HU_enc), name = 'b1')

W2 = init_weights((dimZ, HU_enc), name = 'W2')
b2 = init_biases((dimZ), name = 'b2')

W3 = init_weights((dimZ, HU_enc), name = 'W3')
b3 = init_biases((dimZ), name = 'b3')

W4 = init_weights((HU_dec, dimZ), name = 'W4')
b4 = init_biases((HU_dec), name = 'b4')

W5 = init_weights((28*28, HU_dec), name = 'W5')
b5 = init_biases((28*28), name = 'b5')

#f = file('obj.save', 'rb')
#params_loaded = pickle.load(f)
#f.close()
#
#W1, W2, W3, W4, W5, b1, b2, b3, b4, b5 = params_loaded

cost = model(X, W1, W2, W3, W4, W5, b1, b2, b3, b4, b5, b_size)
params = [W1, W2, W3, W4, W5, b1, b2, b3, b4, b5]
reconstruction = latent_rec(Z, W4, W5, b4, b5)

updates = adada(-cost , params, b_size/N ,learning_rate = 0.01)

train = theano.function(inputs=[X], updates=updates, allow_input_downcast=True)
valid = theano.function(inputs=[X], outputs= cost, allow_input_downcast=True)
latent_space = theano.function(inputs=[Z], outputs = reconstruction, allow_input_downcast=True)

bound_train = np.empty(epochs)
bound_valid = np.empty(epochs)
times       = np.empty(epochs)

for i in range(epochs):
    
    t0 = time.time()    
    print(i)    
    
    bound_t = 0
    bound_v = 0
    for start, end in zip(range(0, len(train_set_x), b_size), range(b_size, len(train_set_x), b_size)):
        train(train_set_x[start:end].T)
        
    for start, end in zip(range(0, len(train_set_x), b_size), range(b_size, len(train_set_x), b_size)):
        bound_t += valid(train_set_x[start:end].T)   
        
    for start, end in zip(range(0, len(valid_set_x), b_size), range(b_size, len(valid_set_x), b_size)):
        bound_v += valid(valid_set_x[start:end].T)          

    bound_train[i] = bound_t/(len(train_set_x)/b_size)    
    bound_valid[i] = bound_v/(len(valid_set_x)/b_size) 
    times[i]          = time.time()-t0
    
    print(bound_valid[i])
    print(times[i])
        
        
plt.plot(N*np.arange(len(bound_train[:1000])), bound_train[:1000], linewidth=3, label="train")
plt.plot(N*np.arange(len(bound_valid[:1000])), bound_valid[:1000], linewidth=3, label="valid")
plt.grid()
plt.legend()
plt.xlabel("samples")
plt.ylabel("loss")
plt.xscale("log")
plt.xlim(1e5, N*len(bound_train[:1000]))
plt.ylim(-150,-100)
plt.show()

#%%

#f = file('/home/artur/Dropbox/BONN/saved_models/mnist.pkl', 'wb')
#pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
#f.close()

#f = file('/home/artur/Dropbox/BONN/saved_models/mnist.pkl', 'rb')
#params_loaded = pickle.load(f)
#f.close()
#
#W1, W2, W3, W4, W5, b1, b2, b3, b4, b5 = params_loaded

#%%

plt.figure(frameon=False)

gs1 = gridspec.GridSpec(19, 19)
gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 

count = 0

for i in np.arange(0.05,1.,0.05):
    for j in np.arange(0.05,1.,0.05):
        plt.subplot(gs1[count])
        z = np.array([norm.ppf(i),norm.ppf(j)])
        z=z.reshape(z.shape[0],1) 
        shared_z = theano.shared(np.asarray(z,dtype=theano.config.floatX),borrow=True)
        rec = np.reshape(latent_space(z),[28,28])
        plt.imshow(rec)
        plt.set_cmap('Greys')
        plt.axis('off')
        count += 1

#count = 0
#
#for i in np.arange(100):
#    count += 1
#    plt.subplot(10,10,count)
#    z = norm.ppf(np.random.rand(dimZ))
#    z=z.reshape(z.shape[0],1) 
#    shared_z = theano.shared(np.asarray(z,dtype=theano.config.floatX),borrow=True)
#    rec = np.reshape(latent_space(z),[28,28])
#    plt.imshow(rec)
#    plt.set_cmap('Greys')
#    plt.axis('off')        
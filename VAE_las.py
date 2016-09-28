# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:44:57 2016

@author: artur
"""

from matplotlib import pyplot as plt
import theano.tensor as T
import numpy as np
import os.path
import scipy.io
import matplotlib.gridspec as gridspec
import pickle
from scipy import signal
import lasagne
from lasagne.updates import adagrad
from lasagne import layers
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import objective
from lasagne.layers import get_all_params

n_hidden = 500
n_latent = 2
batch_size = 100

def lower_bound(layers, batch_size, *args, **kwargs):

    prior = 0.5*T.sum(1 + 2*log_sig_enc - mu_enc**2 - T.exp(2*log_sig_enc))
    logpxz = T.sum(x * T.log(y + (1 - x)*T.log(1 - y)))

    return (prior + logpxz) / batch_size

class GaussTrickLayer(lasagne.layers.Layer):

    eps = theano.shared(floatX(np.random.randn(self.input_shape)))    

    def get_output_for(self, input, **kwargs):
        return input[0] + T.exp(input[1])*eps

net1 = NeuralNet(
    layers=[('x', layers.InputLayer),
            ('h_enc', layers.DenseLayer),
            ('mu_enc', layers.DenseLayer),
            ('log_sig_enc', layers.DenseLayer),
            ('z', layers.GaussTrickLayer),
            ('h_dec', layers.DenseLayer),
            ('y', layers.DenseLayer)
            ],
    
    x_shape=(batch_size, 1, inp_size, inp_size),   #Batchsi

    h_enc_num_units = n_hidden,
    h_enc_incoming = 'x',
    h_enc_nonlinearity = lasagne.nonlinearities.tanh,  

    mu_enc_num_units = n_latent,
    mu_enc_nonlinearity = lasagne.nonlinearities.linear,
    mu_enc_incoming = 'h_enc',   
    
    log_sig_enc_num_units = n_latent,
    log_sig_nonlinearity = lasagne.nonlinearities.linear,
    log_sig_enc_incoming = 'h_enc',    
    
    z_num_units = n_latent,
    z_incomings = ['mu_enc','log_sig_enc'],
    
    h_dec_num_units = n_hidden,
    h_dec_nonlinearity = lasagne.nonlinearities.tanh,
    h_incoming = 'z',
    
    y_num_units = inp_size**2,
    y_incoming = 'h_dec',
    y_nonlinearity = lasagne.nonlinearity.sigmoid,
    
    update = adagrad,
    update_learning_rate = 0.01,
    objective = lower_bound,
    objective_layers = ['log'],

    objective_lambda11 = omega1,
    objective_lambda12 = omega2,
    regression = True,
    max_epochs = 300,
    verbose = 1,
    )

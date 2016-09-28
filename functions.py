# -*- coding: utf-8 -*-
"""
Created on Sun Nov 01 12:35:46 2015

@author: User
"""

import theano
import theano.tensor as T
import numpy
import pickle
import sys
import timeit

def load_data_shared(dataset):  
     
    print('... loading data') 
    f = open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding = 'latin1')
    f.close()
    
    def shared_dataset(data_xy, borrow=True):
        
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')
        
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    
    return rval      
    
def load_data_np(dataset):  
     
    print('... loading data') 
    f = open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding = 'latin1')
    f.close()
    
    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    
    return rval      
    
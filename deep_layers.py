# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 23:34:10 2017

@author: PeaceSea
"""

import numpy as np

#%% Helper functions
def sigmoid(Z):
    g = 1.0/(1.0 + np.exp(-Z))
    return g
def relu(Z):
    return Z * (Z > 0)
#%% Affine
def affine_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def affine_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

#%% Sigmoid
def sigmoid_forward(A_prev, W, b):
    Z, cache = affine_forward(A_prev, W, b)
    A = sigmoid(Z)
    sigmoid_cache = (cache, Z)
    return A, sigmoid_cache # A, A_prev, W, b, Z
def sigmoid_backward(dA, sigmoid_cache):
    affine_cache, Z = sigmoid_cache
    dZ = dA * sigmoid(Z) * (1-sigmoid(Z))
    dA_prev, dW, db = affine_backward(dZ, affine_cache)
    return dA_prev, dW, db

#%% Relu
def relu_forward(A_prev, W, b):
    Z, cache = affine_forward(A_prev, W, b)
    A = Z * (Z > 0)
    relu_cache = (cache, Z)
    return A, relu_cache # A, A_prev, W, b, Z
def relu_backward(dA, relu_cache):
    affine_cache, Z = relu_cache
    dZ = dA * (Z > 0)
    dA_prev, dW, db = affine_backward(dZ, affine_cache)
    return dA_prev, dW, db

#%% layer with input activation function
def layer_forward(A_prev, W, b, activation = 'relu'):
    Z, cache = affine_forward(A_prev, W, b)
    if activation == 'relu':
        A = Z *(Z > 0)
    elif activation == 'sigmoid':
        A = sigmoid(Z)
    cache = (cache, Z)
    
    return A, cache # A, A_prev, W, b, Z

def layer_backward(dA, cache, activation):
    affine_cache, Z = cache
    if activation == 'relu':
        dZ = dA *(Z > 0)
    elif activation == 'sigmoid':
        dZ = dA * sigmoid(Z) * (1-sigmoid(Z))
    dA_prev, dW, db = affine_backward(dZ, affine_cache)
    
    return dA_prev, dW, db

#%% Inititalize deep layers
def init_deep_layers(layer_dims):
    params = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return params

#%% Deep model calculation flow general
def deep_model_forward(X, params, activations, keep_probs):
    A = X
    caches = []
    L = len(params)//2
    
    for l in range(0, L):
        A_prev = A
        A, cache = layer_forward(A_prev, params['W' + str(l+1)], params['b' + str(l+1)], activations[l])        
        D = None
        if keep_probs[l] < 1.0: # Drop out
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < 1.0)
            A = np.multiply(A,D)
            A = A/keep_probs[l]

        cache = (cache, D)
        caches.append(cache)
    return A, caches

def compute_cost(AL, Y, params, lamda = 0.0):
    m = Y.shape[1]
    cost = -1/m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))
    cost = np.squeeze(cost)
    
    if lamda  != 0: # Regularization
        L = len(params)//2
        L2 = 0
        for l in range(1, L+1):
            L2 += np.sum(np.square(params['W' + str(l)]))    
        cost += lamda/2/m * L2
    
    return cost

def deep_model_backward(AL, Y, caches, activations, params, keep_probs, lamda = 0.0):
    # [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID
    grads = {}
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    L = len(caches)
    m = Y.shape[1]
    
    dA = dAL
    for l in reversed(range(L)):
        cache, D = caches[l]
        dA, grads['dW' + str(l+1)], grads['db' + str(l+1)] = layer_backward(dA, cache, activations[l])
        if lamda !=0: # Regularization
            grads['dW' + str(l+1)] += lamda/m * params['W' + str(l+1)]
        elif keep_probs[l] < 1.0: # Drop out
            dA = dA * D
            dA /= keep_probs[l]
            
    return grads

#%% Update params
def update_params_with_SGD(grads, params, learning_rate):
    L = len(params)//2
    for l in range(1, L+1):        
        params['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        params['b' + str(l)] -= learning_rate * grads['db' + str(l)]
    return params

# Momentum
def init_momentum(params):
    v = {}
    L = len(params)//2
    for l in range(1, L+1):
        v['dW' + str(l)] = np.zeros(params['W' + str(l)].shape)
        v['db' + str(l)] = np.zeros(params['b' + str(l)].shape)
    return v

def update_params_with_momentum(grads, params, learning_rate, v, beta = 0.9):
    L = len(params)//2
    for l in range(1, L+1):
        v['dW' + str(l)] = beta * v['dW' + str(l)] + (1-beta) * grads['dW' + str(l)]
        v['db' + str(l)] = beta * v['db' + str(l)] + (1-beta) * grads['db' + str(l)]
        params['W' + str(l)] -= learning_rate * v['dW' + str(l)]
        params['b' + str(l)] -= learning_rate * v['db' + str(l)]
    return v, params

# Adam
def init_adam(params):
    v = {}
    s = {}
    L = len(params)//2
    for l in range(1, L+1):
        v['dW' + str(l)] = np.zeros(params['W' + str(l)].shape)
        v['db' + str(l)] = np.zeros(params['b' + str(l)].shape)
        s['dW' + str(l)] = np.zeros(params['W' + str(l)].shape)
        s['db' + str(l)] = np.zeros(params['b' + str(l)].shape)
    
    return v, s

def update_params_with_adam(grads, params, learning_rate, v, s, t, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8):
    v_corrected = {}
    s_corrected = {}
    L = len(params)//2
    for l in range(1, L+1):
        v['dW' + str(l)] = beta1 * v['dW' + str(l)] + (1-beta1) * grads['dW' + str(l)]
        v['db' + str(l)] = beta1 * v['db' + str(l)] + (1-beta1) * grads['db' + str(l)]
        v_corrected['dW' + str(l)] = v['dW' + str(l)] / (1-np.power(beta1,t))
        v_corrected['db' + str(l)] = v['db' + str(l)] / (1-np.power(beta1,t))
        s['dW' + str(l)] = beta2 * s['dW' + str(l)] + (1-beta2) * grads['dW' + str(l)]**2
        s['db' + str(l)] = beta2 * s['db' + str(l)] + (1-beta2) * grads['db' + str(l)]**2
        s_corrected['dW' + str(l)] = s['dW' + str(l)] / (1-np.power(beta2,t))
        s_corrected['db' + str(l)] = s['db' + str(l)] / (1-np.power(beta2,t))
        
        params['W' + str(l)] -= learning_rate * v_corrected['dW' + str(l)]/np.sqrt(s_corrected['dW' + str(l)] + epsilon)
        params['b' + str(l)] -= learning_rate * v_corrected['db' + str(l)]/np.sqrt(s_corrected['db' + str(l)] + epsilon)
    
    return v, s, params

#%% Build model and update params

# Random minibatch
def random_mini_batches(X, Y, batch_size):
    m = Y.shape[1]
    permutation = np.random.permutation(m)
    X_shuffle = X[:, permutation]
    Y_shuffle = Y[:, permutation]
    num_batches = int(np.floor(m/batch_size))
    
    mini_batches = []
    for i in range(num_batches):
        X_batch = X_shuffle[:,i*batch_size: (i+1)*batch_size]
        Y_batch = Y_shuffle[:,i*batch_size: (i+1)*batch_size]        
        mini_batches.append((X_batch, Y_batch))
    if m % batch_size !=0:
        X_batch = X_shuffle[:, num_batches*batch_size: m]
        Y_batch = Y_shuffle[:, num_batches*batch_size: m]
        mini_batches.append((X_batch, Y_batch))
    
    return mini_batches

# Update
def deep_model(X, Y, layer_dims, activations, epochs, batch_size, lamda, keep_probs, learning_rate, method = 'sgd', beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8):  
    np.random.seed(0)
    params = init_deep_layers(layer_dims)
    costs = []
    t = 0
    
    if method == 'sgd':
        pass
    elif method == 'momentum':
        v = init_momentum(params)
    elif method == 'adam':
        v, s = init_adam(params)        

    for i in range(epochs):
        mini_batches = random_mini_batches(X, Y, batch_size)
        
        for mini_batch in mini_batches:
            X_batch, Y_batch = mini_batch
            
            AL, caches = deep_model_forward(X_batch, params, activations, keep_probs)
            cost = compute_cost(AL, Y_batch, params, lamda)
            grads = deep_model_backward(AL, Y_batch, caches, activations, params, keep_probs, lamda)
            
            if method == 'sgd':
                params = update_params_with_SGD(grads, params, learning_rate)
            elif method == 'momentum':
                v, params = update_params_with_momentum(grads, params, learning_rate, v, beta1)
            elif method == 'adam':
                t += 1
                v, s, params = update_params_with_adam(grads, params, learning_rate, v, s, t)
        if i % 1 == 0:
            print('Cost {:6.2f}'.format(cost))
            costs.append(cost)
            
    return costs, params


    
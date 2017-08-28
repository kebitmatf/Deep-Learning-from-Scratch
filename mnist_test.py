# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 12:09:53 2017

@author: PeaceSea
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import deep_layers as dl

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST', one_hot = True)

X_train = data.train.images.T
Y_train = data.train.labels.T
X_test = data.test.images.T
Y_test = data.test.images.T

nx = X_train.shape[0]
ny = Y_train.shape[0]

layer_dims = [nx, 300, 100, ny]
activations = ['relu', 'relu', 'sigmoid']
keep_probs = [1, 1, 1]
learning_rate = 0.001
lamda = 0.001
epochs = 10
batch_size = 64

start_time = time.time()
costs, params = dl.deep_model(X_train, Y_train, layer_dims, activations, epochs, batch_size, lamda, keep_probs, learning_rate, 'adam', 0.9, 0.99, 1e-8)
end_time = time.time()

print(end_time - start_time)

plt.plot(costs)
plt.show()

#%% Check accuracy
y_train_true_cls = np.argmax(data.train.labels, axis = 1)
y_test_true_cls = np.argmax(data.test.labels, axis = 1)

AL, _ = dl.deep_model_forward(X_train, params, activations, keep_probs)
y_train_pred_cls = np.argmax(AL, axis = 0)
correct = (y_train_pred_cls == y_train_true_cls)
accuracy_train = np.sum(correct)/len(y_train_pred_cls)

AL, _ = dl.deep_model_forward(X_test, params, activations, keep_probs)
y_test_pred_cls = np.argmax(AL, axis = 0)
correct = (y_test_pred_cls == y_test_true_cls)
accuracy_test = np.sum(correct)/len(y_test_pred_cls)

print('Train accuracy {0:2.3%}, Test accuracy {1:2.3%}'.format(accuracy_train, accuracy_test))

    
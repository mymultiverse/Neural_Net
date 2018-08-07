import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from nn import NNet
import os

print(os.listdir("input"))
import pandas as pd

train = pd.read_csv("input/train.csv",sep=',')
test  = pd.read_csv("input/test.csv",sep=',')

train_x = train.values[:,1:]
train_y = train.values[:,0]
# Write to the log:
print("Training set {}has".format(train_x.shape))
print("Test set has {} columns".format(train_y.shape))



X = train_x
Y = train_y
layer_dims=[784,15,10] # desired structure of NN with nodes in each layer

lr = 0.03 # learning rate
iterations = 20000
nn = NNet(X, Y, layer_dims, lr, iterations) 
para = nn.train()
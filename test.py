import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from nn import NNet
import pandas as pd
from argparse import ArgumentParser

# For MNIST data
# train = pd.read_csv("input/train.csv",sep=',')
# test  = pd.read_csv("input/test.csv",sep=',')

# train_x = train.values[:,1:]
# train_y = train.values[:,0]

argparser.add_argument('-itr', '--iteration', type=int, default=20000, dest='itr', help='number of iterations for learning process')
 args = argparser.parse_args()

# IRIS DATA
iris = datasets.load_iris()
# shuffle for generalized training as data arrange in catagory order
iris.data , iris.target = shuffle(iris.data,iris.target)

l = 120 # data split
train_x = iris.data[:l,:]
train_y = iris.target[:l]

test_x = iris.data[l:,:]
test_y = iris.target[l:]

X = train_x   # make sure input data dim. (no. of example, number of features)
Y = train_y # Target output

inD = train_x.shape[1]
outD = np.max(Y)+1

layer_dims=[ind,8,6,outd] # desired structure of NN with nodes in each layer

lr = 0.03 # learning rate
iterations = args.itr
nn = NNet(X, Y, layer_dims, lr, iterations) 
para = nn.train()
test_y_pred = nn.predict(test_x, para)
print ("Test Accuracy:{}%".format(nn.accuracy(test_y_pred,test_y)))

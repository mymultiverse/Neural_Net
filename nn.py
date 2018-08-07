import numpy as np 
from matplotlib import pyplot as plt

class NNet(object):
    def __init__(self, X,Y,layer_dims, lr, iterations, dp = 1, out_act="sigmoid"):
        super(NNet,self).__init__()
        self.X = X.T
        self.Y = Y
        self.layer_dims = layer_dims 
        self.learning_rate = lr
        self.iterations = iterations
        self.keep_prob = dp
        self.out_act = out_act  

    def one_hot(self):
        # conversion into one hot encoding   
        Y = np.zeros((self.Y.shape[0],self.layer_dims[-1]))
        Y[np.arange(self.Y.shape[0]),self.Y]=1 
        return  Y.T


    def init_paras(self):
        L = len(self.layer_dims)  # number of layers in the network
        parameters = {}          
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1])*0.01
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))        
            assert(parameters['W' + str(l)].shape == (self.layer_dims[l], self.layer_dims[l - 1]))
            assert(parameters['b' + str(l)].shape == (self.layer_dims[l], 1))
 
        return parameters
    

    def sigmoid(self,z):
        s = 1 / (1 + np.exp(-z))
        cache = s    
        return s, cache # storing requried variables in cache as required for back propagation

    def softmax(self,z):
        s = np.exp(z)/np.sum(np.exp(z),axis=0)
        cache = s
        return s, cache

    # back functions are derivatives of respective activation
    def sig_back(self,dA, activation_cache):
        return dA*activation_cache*(1 - activation_cache)

    def soft_back(self,dA, activation_cache):

        ac = activation_cache
        dz = np.empty(ac.shape)

        for a in range(ac.shape[1]):
            s = ac[:,a].reshape(-1,1)
            ds = np.diagflat(s) - np.dot(s, s.T)
            dz[:,a] = np.matmul(ds,dA[:,a]) 
        
        assert(dz.shape == (dA.shape[0], dA.shape[1]))
        return dz

    def lin_forward(self,A, W, b,D):
       
        Z = np.dot(W, A) + b
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b, D)
        
        return Z, cache

    def apply_dp_out(self,A_prev,D):
        if (np.array_equal(self.X,A_prev)!=False):
            D = np.random.rand(A_prev.shape[0], A_prev.shape[1])     
            D = D < self.keep_prob                            
            A_prev = A_prev * D                                     
            A_prev = A_prev / self.keep_prob 

        return A_prev, D

        # applying activation after linear operation
    def lin_act_forward(self,A_prev,W,b,activation):
        D=0
        if activation == "sigmoid":
            if self.keep_prob<1:
                A_prev, D = self.apply_dp_out(A_prev,D)
                

            Z, linear_cache = self.lin_forward(A_prev, W, b,D)
            A, activation_cache = self.sigmoid(Z)
        
        elif activation == "softmax":
            if self.keep_prob<1:
                A_prev, D = self.apply_dp_out(A_prev,D)

            Z, linear_cache = self.lin_forward(A_prev, W, b,D)
            A, activation_cache = self.softmax(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_prop(self,in_X,parameters):
      
        caches = []
        A = in_X # input layer
        L = len(parameters) // 2             
        
        # hidden layers
        for l in range(1, L):
            A_prev = A 
            A, cache = self.lin_act_forward(A_prev,
                parameters['W' + str(l)],parameters['b' + str(l)],activation='sigmoid')

            caches.append(cache)
            
        W = parameters['W' + str(L)]
        b = parameters['b' + str(L)]

        # output layer
        if self.out_act == "softmax":
            AL, cache = self.lin_act_forward(A,W,b,activation='softmax')
        else:
            AL, cache = self.lin_act_forward(A,W,b,activation='sigmoid')

        caches.append(cache)
        
        assert(AL.shape == (self.one_hot().shape[0], in_X.shape[1]))
                
        return AL, caches

    def compute_cost(self,AL):

        Y = self.one_hot()
        m = Y.shape[1] # total number of training examples

        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL))) 
        # cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        return cost


    def linear_backward(self,dZ, cache):

        A_prev, W, b, D = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, cache[0].T) / m
        db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
        dA_prev = np.dot(cache[1].T, dZ)

        # backward dropout
        if self.keep_prob<1:
            dA_prev = dA_prev*D
            dA_prev = dA_prev/self.keep_prob

        db = np.reshape(db,b.shape)
     
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        
        return dA_prev, dW, db



    def back_prop(self, AL, caches):
       
        grads = {}
        L = len(caches) 
        m = AL.shape[1]
        Y = self.one_hot()

        # Derivative of loss  
        dAL = - np.divide(Y, AL) 

        # dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
       
        current_cache = caches[-1]
        
        #output layer back prop.
        if self.out_act=="softmax":
            grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_backward(self.soft_back(dAL, 
                                                                                            current_cache[1]), 
                                                                                           current_cache[0])
        else:

            grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_backward(self.sig_back(dAL, 
                                                                                            current_cache[1]), 
                                                                                           current_cache[0])

        # hidden layer back prop.    
        for l in reversed(range(L-1)):
            
            current_cache = caches[l]
            dA = grads["dA" + str(l+2)]
            dA_prev_temp, dW_temp, db_temp = self.linear_backward(self.sig_back(dA, current_cache[1]), current_cache[0])
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads


    def update_parameters(self, parameters,grads):
        L = len(parameters) // 2 

        for l in range(L):
            
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - self.learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - self.learning_rate * grads["db" + str(l + 1)]
            
        return parameters


    def train(self, print_cost=True): 
        costs = []                         
        
        parameters = self.init_paras()
        inx = self.X

        for i in range(0, self.iterations):
            AL, caches = self.forward_prop(inx,parameters)
            cost = self.compute_cost(AL)
            grads = self.back_prop(AL, caches)
            parameters = self.update_parameters(parameters,grads)
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" % (i, cost))
            if i % 100 == 0:
                costs.append(cost)

        train_y_pred = self.predict(inx.T, parameters) 
        
        print ("    Training accuracy:{}%" .format(self.accuracy(train_y_pred,self.Y)))

        print("    Close the Current plot to see Next Result")

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.grid()
        plt.show()

        

        return parameters


    def predict(self,t_X,parameters):
        self.keep_prob = 1 # Dropout only used during training
        t_X = t_X.T
        Y_pred, _ = self.forward_prop(t_X,parameters)
        Y_pred = np.argmax(Y_pred.T, axis=1)

        return Y_pred

    def accuracy(self,Y_pred,True_y):
        acc = Y_pred == True_y 
        acc = acc.astype(np.float)
        acc = (np.sum(acc)/True_y.size)*100
        return acc
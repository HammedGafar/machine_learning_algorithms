# Place your EWU ID and Name here. 
#1008912 Gafar Hammed

### Delete every `pass` statement below and add in your own code. 



# Implementation of the forwardfeed neural network using stachastic gradient descent via backpropagation
# Support parallel/batch mode: process every (mini)batch as a whole in one forward-feed/backtracking round trip. 



import numpy as np
import math
import math_util 
import nn_layer 


class NeuralNetwork:
    
    def __init__(self):
        self.layers = []     # the list of L+1 layers, including the input layer. 
        self.L = -1          # Number of layers, excluding the input layer. 
                             # Initting it as -1 is to exclude the input layer in L. 
        self.start = None
        self.mini_batch_end_location = None
        self.error_list = []
    
    
    def add_layer(self, d = 1, act = 'tanh'):
        ''' The newly added layer is always added AFTER all existing layers.
            The firstly added layer is the input layer.
            The most recently added layer is the output layer. 
            
            d: the number of nodes, excluding the bias node, which will always be added by the program. 
            act: the choice of activation function. The input layer will never use an activation function even if it is given. 
            
            So far, the set of supported activation functions are (new functions can be easily added into `math_util.py`): 
            - 'tanh': the tanh function
            - 'logis': the logistic function
            - 'iden': the identity function
            - 'relu': the ReLU function
        '''
        layer = nn_layer.NeuralLayer(d=d, act=act)
        self.layers.append(layer)

                
        self.L = self.L + 1

    

    def _init_weights(self):
        ''' Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)], 
            where d is the number of nonbias node of the layer
        '''

        for i in range(1, len(self.layers)):
            curr_layer = self.layers[i]
            d = curr_layer.d
            prev_layer = self.layers[i-1]
            weight_n = prev_layer.d + 1
            weight_d = curr_layer.d
            lower_range = -1/np.sqrt(d)
            upper_range = 1/np.sqrt(d)
            matrix = np.random.uniform(lower_range, upper_range, (weight_n, weight_d))
            
            
            curr_layer.W = matrix

    
        
    def fit(self, X, Y, eta = 0.01, iterations = 1000, SGD = True, mini_batch_size = 1):
        ''' Find the fitting weight matrices for every hidden layer and the output layer. Save them in the layers.
          
            X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
            Y: n x k vector of lables, where k >= 1 is the number of classes in the multi-class classification
            eta: the learning rate used in gradient descent
            iterations: the maximum iterations used in gradient descent
            SGD: True - use SGD; False: use batch GD
            mini_batch_size: the size of each mini batch size, if SGD is True.  
        '''
        self._init_weights()  # initialize the edge weights matrices with random numbers.

        
        
        # I will leave you to decide how you want to organize the rest of the code, but below is what I used and recommend. Decompose them into private components/functions. 

        ## prep the data: add bias column; randomly shuffle data training set.
        n,d = X.shape
        permutation = np.random.permutation(n)
        X = X[permutation]
        Y = Y[permutation] 
        X = np.insert(X, 0, 1, axis=1)
        
        if SGD == False:
            mini_batch_size = n
            
        
        itr = n/mini_batch_size
        itr = math.ceil(itr)
        
        self.start = 0
        self.mini_batch_end_location = mini_batch_size
        
        for  iteration in range(iterations):
            
            for miniBatch_itr in range(itr):
                
                X_mini, y_mini = self.mini_batch (X, Y, mini_batch_size)
                n_mini,d_mini = X_mini.shape
                
                
                #Forward Feeding
                self.feedForward_fit(X_mini)

                #error = self.error(X_mini, y_mini)
                #self.error_list.append(error)
                
                
                
                #Backward propagation
                for i in range(len(self.layers)-1, 0, -1):
                    
                    if i == len(self.layers)-1:
                        index = len(self.layers)-1
                        curr_layer = self.layers[index]
                        prev_layer = self.layers[index-1]
                        pred = ((curr_layer.X[:,1:]))
                        #pred_network = np.argmax(pred, axis=1).reshape(-1, 1)
                        #pred_actual = np.argmax(y_mini, axis=1).reshape(-1, 1)
                        curr_layer.Delta = (2*(pred - y_mini)) * (curr_layer.act_de(curr_layer.S))
                        curr_layer.G = np.einsum('ij,ik->jk', prev_layer.X, curr_layer.Delta) * 1/n_mini
                        
                    else:
                        curr_layer= self.layers[i]
                        next_layer = self.layers[i+1]
                        prev_layer = self.layers[i-1]
                        
                        curr_layerS = curr_layer.S
                        
                        next_layerDelta = next_layer.Delta
                        nextLayer_W_minusBiaseRow = (next_layer.W[1:,:])
                        
                        curr_layer.Delta = curr_layer.act_de(curr_layerS) * (next_layerDelta @ nextLayer_W_minusBiaseRow.T)
                        curr_layer.G = np.einsum('ij,ik->jk', prev_layer.X, curr_layer.Delta) * 1/n_mini
                        
                for i in range(1, len(self.layers)):
                    curr_layer = self.layers[i]
                    curr_layer.W = curr_layer.W - eta *(curr_layer.G)
            
                
                    
                    
                    
            
            
            

        ## for every iteration:
        #### get a minibatch and use it for:
        ######### forward feeding
        ######### calculate the error of this batch if you want to track/observe the error trend for viewing purpose.
        ######### back propagation to calculate the gradients of all the weights
        ######### use the gradients to update all the weight matrices. 

  
    
    
    
    def predict(self, X):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column.
            
            return: n x 1 matrix, n is the number of samples, every row is the predicted class id.
            
            Note that the return of this function is NOT the sames as the return of the `NN_Predict` method
            in the lecture slides. In fact, every element in the vector returned by this function is the column
            index of the largest number of each row in the matrix returned by the `NN_Predict` method in the 
            lecture slides.
         '''
         
         #X = np.insert(X, 0, 1, axis=1)

        self.feedForward_predict(X)
        
        length = len(self.layers)
        
        output = self.layers[length-1].X
        
        
        output = output[:,1:]
        
        # Return indices of maximum values along each row
        return np.argmax(output, axis=1).reshape(-1, 1)
    
    
    def error(self, X, Y):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column. 
               n is the number of samples. 
               d is the number of (non-bias) features of each sample. 
            Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample, 
               where only one entry is 1 and the rest are all 0. 
               Y[i,j]=1 indicates the ith sample belongs to class j.
               k is the number of classes. 
            
            return: the percentage of misclassfied samples
        '''
        n,d = X.shape
        
        prediction = self.predict(X)
        
        Y = np.argmax(Y, axis=1).reshape(-1, 1)
        
        
        error_count = np.sum(prediction != Y)
        
        percentage_error = (error_count/n) * 100
        
        return percentage_error
        
    
    def feedForward_fit (self, X):
        
        #X = np.insert(X, 0, 1, axis=1)
        
        #First layer output
        self.layers[0].X = X
        
        for i in range(1, len(self.layers)):
            curr_layer = self.layers[i]
            prev_layer = self.layers[i-1]
            curr_layer.S = prev_layer.X @ curr_layer.W
            act = curr_layer.act(curr_layer.S)
            

            curr_layer.X = np.insert(act, 0, 1, axis=1)
                
                
    def feedForward_predict (self, X):
        
        X = np.insert(X, 0, 1, axis=1)
        
        #First layer output
        self.layers[0].X = X
        
        for i in range(1, len(self.layers)):
            curr_layer = self.layers[i]
            prev_layer = self.layers[i-1]
            #curr_layer.S = prev_layer.X @ curr_layer.W
            act = curr_layer.act(prev_layer.X @ curr_layer.W)
            
            curr_layer.X = np.insert(act, 0, 1, axis=1)
            

                
    
    def mini_batch (self, X, y, miniBatchSize):
        n_main, d_main = X.shape
        d_main
        
        if (self.start >= n_main):
            self.start = 0
            self.mini_batch_end_location = miniBatchSize
            
        X_mini = X[self.start:self.mini_batch_end_location, :]
        y_mini = y[self.start:self.mini_batch_end_location, :]

        
        self.start = self.mini_batch_end_location
        self.mini_batch_end_location = self.mini_batch_end_location + miniBatchSize
        
        return X_mini, y_mini

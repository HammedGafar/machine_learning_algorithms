########## >>>>>> Put your full name and 6-digit EWU ID here. 

# Implementation of the logistic regression with L2 regularization and supports stachastic gradient descent




import numpy as np
import pandas as pd
import math
import sys
sys.path.append("..")

from itertools import product

from code_misc.utils import MyUtils



class LogisticRegression:
    def __init__(self):
        self.w = None
        self.degree = 1
        
        self.start = None
        self.mini_batch_end_location = None
        
        

        

    def fit(self, X, y, lam = 0, eta = 0.01, iterations = 1000, SGD = False, mini_batch_size = 1, degree = 1):
        ''' Save the passed-in degree of the Z space in `self.degree`. 
            Compute the fitting weight vector and save it `in self.w`. 
         
            Parameters: 
                X: n x d matrix of samples; every sample has d features, excluding the bias feature. 
                y: n x 1 vector of lables. Every label is +1 or -1. 
                lam: the L2 parameter for regularization
                eta: the learning rate used in gradient descent
                iterations: the number of iterations used in GD/SGD. Each iteration is one epoch if batch GD is used. 
                SGD: True - use SGD; False: use batch GD
                mini_batch_size: the size of each mini batch size, if SGD is True.  
                degree: the degree of the Z space
        '''

        #self.degree = degree
        #self.mini_batch_location = mini_batch_size
        
        X = MyUtils.z_transform(X, degree = self.degree)
        X = np.insert(X, 0, 1, axis=1)
        n,d = X.shape
        
        initial_w  = np.zeros(d).reshape(-1,1)
        self.w = initial_w
        
        
        if SGD is False: 
            n,d = X.shape
            full_batch_regularizer = (1) - ((2 * lam * eta)/n)
            for i in range(iterations): 
                s = y * (X@self.w)
                vectorized_sigmoid = LogisticRegression._v_sigmoid(-s)
                vectorized_sigmoid = vectorized_sigmoid
                self.w = full_batch_regularizer * self.w + (eta/n) * (X.T @ (y * vectorized_sigmoid))
                
            self.w = self.w.reshape(-1, 1)
  
        else:
                # Create a permutation of indices
            permutation = np.random.permutation(n)
            X = X[permutation]
            y = y[permutation]
            
            n_main, d_main = X.shape
            
            self.start = 0
            self.mini_batch_end_location = mini_batch_size

            for i in range(iterations):
                X_mini, y_mini = self.mini_batch (X, y, mini_batch_size)
                
                print(f"start {self.start}")
                print(f"mini_batch_location {self.mini_batch_end_location}")
                 
                n_mini,d_mini = X_mini.shape
                print(f"X_mini rows {n_mini}")
                
                lam_mini = lam * (n_mini / n_main)
                
                miniBatch_batch_regularizer = (1) - ((2 * lam_mini * eta)/n_mini)
                
                s = y_mini * (X_mini@self.w)
                
                vectorized_sigmoid = LogisticRegression._v_sigmoid(-s)
                self.w = miniBatch_batch_regularizer * self.w + (eta/n_mini) * (X_mini.T @ (y_mini * vectorized_sigmoid))
                

    
    def predict(self, X):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
            return: 
                n x 1 matrix: each row is the probability of each sample being positive. 
        '''
    
        X = MyUtils.z_transform(X, degree = self.degree)
        X = np.insert(X, 0, 1, axis=1)
        # remove the pass statement and fill in the code. 
        
        s = X @ self.w
        vectorized_sigmoid = LogisticRegression._v_sigmoid(s)
        
    
            
        return vectorized_sigmoid
        
    
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
                y: n x 1 matrix; each row is a labels of +1 or -1.
            return:
                The number of misclassified samples. 
                Every sample whose sigmoid value > 0.5 is given a +1 label; otherwise, a -1 label.
        '''
        
        X = MyUtils.z_transform(X, degree = self.degree)
        X = np.insert(X, 0, 1, axis=1)
        
        signals = X @ self.w
        
        # Define bins for categorization
        bins = [0]

        # Categorize the data
        categories = np.digitize(signals, bins)

        mapped_categories = np.where(categories == 1, 1, -1)
        
        not_equal_elements = mapped_categories != y
        errors = np.sum(not_equal_elements)
        
        return errors


    def _v_sigmoid(s):
        '''
            vectorized sigmoid function
            
            s: n x 1 matrix. Each element is real number represents a signal. 
            return: n x 1 matrix. Each element is the sigmoid function value of the corresponding signal. 
        '''

        vectorizeSigmoid = np.vectorize(LogisticRegression._sigmoid)
        
        return vectorizeSigmoid(s)
                 

        
    def _sigmoid(s):
        ''' s: a real number
            return: the sigmoid function value of the input signal s
        '''

        return 1 / ( 1 + np.exp(-s))
    
    def SGD_test_hyperparameters(self, X_train, y_train, X_test, y_test):
        
        self.SGD_Degree = []
        self.SGD_Lambda = []
        self.SGD_LearningRate = []
        self.SGD_Iteration = []
        self.SGD_mini_batch_size = []
        self.SGD_sample_error = []
        self.SGD_Validation_Error = []
        
        SGD_Degree = [1, 2, 3, 4,]
        SGD_Lambda =  [10, 1, 0.1, 0.01, 0.001]
        SGD_LearningRate = [0.1, 0.05, 0.01, 0.001]
        SGD_Iteration = [500, 1000, 10000]
        SGD_mini_batch_size = [5, 10, 20, 30 , 40]

        combinations = list(product(SGD_Degree, SGD_Lambda, SGD_LearningRate, SGD_Iteration, SGD_mini_batch_size))
        
        self.append(combinations)

        for SGD_Degree, SGD_Lambda, SGD_LearningRate, SGD_Iteration, SGD_mini_batch_size in combinations:
            
            self.fit(X_train, y_train, lam = SGD_Lambda, eta = SGD_LearningRate, iterations = SGD_Iteration, SGD = True, mini_batch_size = SGD_mini_batch_size, degree = SGD_Degree)
            
            sampleError = self.error(X_train, y_train)
            self.SGD_sample_error.append(sampleError)
            
            validationError = self.error(X_test, y_test)
            self.SGD_Validation_Error.append(validationError)
    
        df = pd.DataFrame({"SGD_degree": self.SGD_Degree, "SGD_LearningRate" : self.SGD_LearningRate, "SGD_Iteration" : self.SGD_Iteration,
                        "SGD_lambda": self.SGD_Lambda,"SGD_Minibatch_Size": self.SGD_mini_batch_size, "SGD_sampleError": self.SGD_sample_error, "SGD_ValidationError": self.SGD_Validation_Error})
        df.to_csv("SGD_LogisticRegression.csv")
        
    
    def mini_batch (self, X, y, miniBatchSize):
        n_main, d_main = X.shape
        
        if (self.start >= n_main):
            self.start = 0
            self.mini_batch_end_location = miniBatchSize
            
        X_mini = X[self.start:self.mini_batch_end_location, :]
        y_mini = y[self.start:self.mini_batch_end_location, :]
        
        self.start = self.mini_batch_end_location
        self.mini_batch_end_location = self.mini_batch_end_location + miniBatchSize
        
        return X_mini, y_mini
    
    def append(self, Combinations):
        
        for Degree, Lambda, LearningRate, Iteration, mini_batch_size in Combinations:
            self.SGD_Degree.append(Degree)
            self.SGD_Lambda.append(Lambda)
            self.SGD_LearningRate.append(LearningRate)
            self.SGD_Iteration.append(Iteration)
            self.SGD_mini_batch_size.append(mini_batch_size)
        
        
        


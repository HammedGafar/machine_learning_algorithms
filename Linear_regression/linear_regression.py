########## >>>>>> Put your full name and 6-digit EWU ID here. 


# Implementation of the linear regression with L2 regularization.
# It supports the closed-form method and the gradient-desecent based method. 



import numpy as np
import math
import sys
sys.path.append("..")
from itertools import product
import pandas as pd

from misc.utils import MyUtils


class LinearRegression:
    def __init__(self):
        self.w = None   # The (d+1) x 1 numpy array weight matrix
        self.degree = 1
        self.lam = 0
        self.epochs = 0
        self.eta = 0
        
        self.MSE = []
        self.MSE_main = 0
        self.MSE_Train = 0
        self.MSE_Validate = 0
        
        self.Cf_degree =  []
        self.CF_lambda = []
        self.CF_inSampleError = []
        self.CF_ValidationError = []
        
        self.gd_degree =  []
        self.gd_lambda = []
        self.gd_learningRate = []
        self.gd_epochs = []
        self.gd_inSampleError = []
        self.gd_ValidationError = []
        
        
        
    def fit(self, X, y, CF = True, lam = 0, eta = 0.01, epochs = 1000, degree = 1):
        ''' Find the fitting weight vector and save it in self.w. 
            
            parameters: 
                X: n x d matrix of samples, n samples, each has d features, excluding the bias feature
                y: n x 1 matrix of lables
                CF: True - use the closed-form method. False - use the gradient descent based method
                lam: the ridge regression parameter for regularization
                eta: the learning rate used in gradient descent
                epochs: the maximum epochs used in gradient descent
                degree: the degree of the Z-space
        '''
        self.degree = degree
        self.lam = lam
        self.epochs = epochs
        self.eta = eta
        
        X = MyUtils.z_transform(X, degree = self.degree)
        
        if CF:
            self._fit_cf(X, y, lam)
        else: 
            self._fit_gd(X, y, lam, eta, epochs)
 


            
    def _fit_cf(self, X, y, lam = 0):
        ''' Compute the weight vector using the clsoed-form method.
            Save the result in self.w
        
            X: n x d matrix, n samples, each has d features, excluding the bias feature
            y: n x 1 matrix of labels. Each element is the label of each sample. 
        '''

        
        X = np.insert(X, 0, 1, axis=1)
        X_transpose = X.T
        n,d = X.shape
        I = np.eye(d)

        
        
        #self.w = np.linalg.pinv(X_transpose @ X) @ (X_transpose @ y)
        self.w = (np.linalg.pinv((X_transpose @ X) + (lam * I))) @ (X_transpose) @ y
        self.w = self.w.reshape(-1, 1)
        


    
    
    def _fit_gd(self, X, y, lam = 0, eta = 0.01, epochs = 1000):
        ''' Compute the weight vector using the gradient desecent based method.
            Save the result in self.w

            X: n x d matrix, n samples, each has d features, excluding the bias feature
            y: n x 1 matrix of labels. Each element is the label of each sample. 
        '''

        X = np.insert(X, 0, 1, axis=1)
        n,d = X.shape

        initial_w  = np.zeros(d).reshape(-1,1)
        
        I = np.eye(d)
        x = (((2 * eta) / n) * (( X.T @ X) + (lam * I) ) )

        a =   I - x
        b = ((2 * eta) / n) * (X.T @ y)


        self.w = initial_w
        for i in range(epochs):
            self.w = ( (a @ self.w) + b ).reshape(-1,1)

            prediction = X@self.w
            grad_error = (prediction - y)
            grad_error_squared = pow(grad_error,2)
            mse = np.sum(grad_error_squared)
            self.MSE.append(mse)
            
            #if mse < best_mse:
                #best_w = self.w
            
        #self.w = best_w.reshape(-1,1)
        
        return self.w
        
        

        ## Delete the `pass` statement below.
        
        ## Enter your code here that implements the gradient descent based method
        ## for linear regression 

        pass


    
    def predict(self, X):
        ''' parameter:
                X: n x d matrix, the n samples, each has d features, excluding the bias feature
            return:
                n x 1 matrix, each matrix element is the regression value of each sample
        '''


        X = MyUtils.z_transform(X, degree = self.degree)
        X = np.insert(X, 0, 1, axis=1)
        
        prediction = X @ self.w
        
        return prediction
        
        
        ## Delete the `pass` statement below.
        
        ## Enter your code here that produces the label vector for the given samples saved
        ## in the matrix X. Make sure your predication is calculated at the same Z
        ## space where you trained your model. Make sure X has been normalized via the same
        ## normalization used by the training process. 

        #pass
        

    def error(self, X, y):
        ''' parameters:
                X: n x d matrix of future samples
                y: n x 1 matrix of labels
            return: 
                the MSE for this test set (X,y) using the trained model
        '''

        
        n,d = X.shape
        
        prediction = self.predict(X)
        error = (prediction - y) 
        error_squared = pow(error,2) 
        
        mse = error_squared/n
        MSE_total = np.sum(mse)
        self.MSE_main = MSE_total
        

        
        return self.MSE_main
        
        
        

        ## Delete the `pass` statement below.
        
        ## Enter your code here that calculates the MSE between your predicted
        ## label vector and the given label vector y, for the sample set saved in matraix X
        ## Make sure your predication is calculated at the same Z space where you trained your model. 
        ## Make sure X has been normalized via the same normalization used by the training process. 

        #pass
    
    def error_train(self, X, y):
        ''' parameters:
                X: n x d matrix of future samples
                y: n x 1 matrix of labels
            return: 
                the MSE for this test set (X,y) using the trained model
        '''
        
        n,d = X.shape
        
        prediction = self.predict(X)
        error = (prediction - y) 
        error_squared = pow(error,2) 
        
        mse = error_squared/n
        MSE_total = np.sum(mse)    
        self.MSE_Train = MSE_total
        

        return self.MSE_Train
        
        
        

        ## Delete the `pass` statement below.
        
        ## Enter your code here that calculates the MSE between your predicted
        ## label vector and the given label vector y, for the sample set saved in matraix X
        ## Make sure your predication is calculated at the same Z space where you trained your model. 
        ## Make sure X has been normalized via the same normalization used by the training process. 

        #pass
        
        
    def errorValidate(self, X, y):
        ''' parameters:
                X: n x d matrix of future samples
                y: n x 1 matrix of labels
            return: 
                the MSE for this test set (X,y) using the trained model
        '''
        
        n,d = X.shape
        
        prediction = self.predict(X)
        error = (prediction - y) 
        error_squared = pow(error,2) 
        
        mse = error_squared/n
        #mse_list = list(mse)
        MSE_total = np.sum(mse)
        self.MSE_Validate = MSE_total
        
        #for i in mse_list:
           # self.MSE.append(i)

        return self.MSE_Validate
    
    def append_cf(self):
        self.Cf_degree.append(self.degree)
        self.CF_lambda.append(self.lam)
        self.CF_inSampleError.append(self.MSE_Train)
        self.CF_ValidationError.append(self.MSE_Validate)
        
    def append_gd(self):
        self.gd_degree.append(self.degree)
        self.gd_lambda.append(self.lam)
        self.gd_learningRate.append(self.eta)
        self.gd_epochs.append(self.epochs)
        self.gd_inSampleError.append(self.MSE_Train)
        self.gd_ValidationError.append(self.MSE_Validate)
        

    def cf_test_hyperparameters(self, X_train, y_train, X_test, y_test):
        cf_lam = [10, 1, 0.1, 0.01, 0.001]
        cf_degree = [1, 2, 3, 4, 5]

        combinations = list(product(cf_lam, cf_degree))

        lr = self.__init__()
        for cf_lam, cf_degree in combinations:
            self.w = None
            self.fit(X_train, y_train, CF = True, lam = cf_lam, eta = 0.01, epochs = 1000, degree = cf_degree)
            self.error_train(X_train, y_train)
            self.errorValidate(X_test, y_test)
            self.append_cf()
    
        df = pd.DataFrame({"cf_degree": self.Cf_degree,"cf_lambda": self.CF_lambda,"cf_sampleError": self.CF_inSampleError, "cf_ValidationError": self.CF_ValidationError})
        df.to_csv("ClosedForm_LinearRegression.csv")
        
    
    def gd_test_hyperparameters(self, X_train, y_train, X_test, y_test):
        gd_lam = [10, 1, 0.1, 0.01, 0.001]
        gd_degree = [1, 2, 3, 4]
        learning_rate = [0.01, 0.001]
        epochs = [1000]

        combinations = list(product(gd_lam, gd_degree, learning_rate, epochs))

        #lr = self()
        for gd_lam, gd_degree, learning_rate, epoch in combinations:
            self.w = None
            self.fit(X_train, y_train, CF = False, lam = gd_lam, eta = learning_rate, epochs = epoch, degree = gd_degree)
            self.error_train(X_train, y_train)
            self.errorValidate(X_test, y_test)
            self.append_gd()
    
        df = pd.DataFrame({"gd_degree": self.gd_degree, "gd_LearningRate" : self.gd_learningRate, "gd_epochs" : self.gd_epochs, "gd_lambda": self.gd_lambda,"gd_sampleError": self.gd_inSampleError, "gd_ValidationError": self.gd_ValidationError})
        df.to_csv("GradientDescent_LinearRegression.csv")



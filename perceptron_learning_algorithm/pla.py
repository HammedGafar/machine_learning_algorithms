
# Implementation of the perceptron learning algorithm. Support the pocket version for linearly unseparatable data. 
# Authro: Bojian Xu, bojianxu@ewu.edu

#Important observation: 
#    - The PLA can increase or decrease $w[0]$ by 1 per update, so if there is a big difference between $w^*[0]$ and the #initial value of $w[0]$, the PLA is likely to take a long time before it halts. However, the theoretical bound $O((L/d)^2)$ #step of course still holds, where $L = \max\{\lVert x\rVert\}$ and $d$ is the margine size.
#    - This can solved by always have feature values within [0,1], because by doing so, the $x_0=1$ becomes relatively larger (or one can also say $x_0$ becomes fairly as important as other feathers), which makes the changes to $w[0]$ much faster. This is partially why nueral network requires all feature value to be [0,1] --- the so-called data normalization process!!!

# Another reason for normalizing the feature into [0,1] is: no matter which Z space the samples are tranformed to, the Z-space sample features will still be in the [0,1] range. 

import numpy as np

#import sys
#sys.path.append("..")

from utils import MyUtils



class PLA:
    def __init__(self, degree=1):
        self.w = None
        self.degree = degree
        
    def fit(self, X, y, pocket = True, epochs = 100):
        ''' find the classifer weight vector and save it in self.w
            X: n x d matrix, i.e., the bias feature is not included. 
            It is assumed that X is already normalized after data preprocessing. 
            y: n x 1 vector of {+1, -1}
            degree: the degree of the Z space
            return self.w
        '''
        
        if(self.degree > 1):
            X = MyUtils.z_transform(X, degree=self.degree)

            
        ### BEGIN YOUR SOLUTION
        d = X.shape[1]
        n = X.shape[0]
        X = np.insert(X, 0, 1, axis = 1)
        self.w = np.zeros((1,d +1))
        
        best_w = self.w
        original_w = []
        
        while (not np.array_equal(original_w, self.w) and epochs > 0):
            original_w = self.w
            epochs -= 1
            
            for i in range(n):
                prev_mc = np.sum( (np.sign(X @ self.w.T)).reshape(-1,1) != y)
                
                if ((np.sign(X[i,:] @ self.w.T)).reshape(-1,1) != y[i]):
                    self.w = self.w + (y[i]*X[i])
                    curr_mc = np.sum( (np.sign(X @ self.w.T).reshape(-1, 1)) != (y))
                    
                    if curr_mc < prev_mc:
                        best_w = self.w
                        
            self.w = best_w
            

        ### END YOUR SOLUTION
        self.w = self.w.reshape(-1, 1)   
                          
        return self.w
    
                          


    def predict(self, X):
        ''' x: n x d matrix, i.e., the bias feature is not included.
            return: n x 1 vector, the labels of samples in X
        '''
        if(self.degree > 1):
            X = MyUtils.z_transform(X, degree = self.degree)

    
        ### BEGIN YOUR SOLUTION
        X = np.insert(X, 0, 1, axis = 1)
        
        return np.sign(X @ self.w)
        ### END YOUR SOLUTION

        
        


    def error(self, X, y):
        ''' X: n x d matrix, i.e., the bias feature is not included. 
            y: n x 1 vector
            return the number of misclassifed elements in X using self.w
        '''
        
        if(self.degree > 1):
            X = MyUtils.z_transform(X, degree = self.degree)

        ### BEGIN YOUR SOLUTION
        X = np.insert(X, 0, 1, axis =1)
        y_pred = np.sign(X @ self.w)
        y_pred = y_pred.reshape(-1,1)
        count = np.sum(y_pred != y)
        return count
        ### END YOUR SOLUTION
            


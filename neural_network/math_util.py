# Place your EWU ID and name here
#1008912 GAFAR HAMMED

## delete the `pass` statement in every function below and add in your own code. 


import numpy as np



# Various math functions, including a collection of activation functions used in NN.




class MyMath:

    def tanh(x):
        ''' tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh of the corresponding element in array x
        '''
        
        vectorized_tanh = np.vectorize(MyMath.tanh_unvectorize) 
        
        return vectorized_tanh(x)
        
        
        

    
    def tanh_de(x):
        ''' Derivative of the tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh derivative of the corresponding element in array x
        '''

        # tanh'(x) = 1.0 - (tanh(x))^2.0

        vectorized_tanh_de = np.vectorize(MyMath.tanh_de_unvectorize)
        
        return vectorized_tanh_de(x)

    
    def logis(x):
        ''' Logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic of 
                    the corresponding element in array x
        '''

        # You can compute logis via tanh as: logix(x) = (tanh(x/2.0) + 1.0) / 2.0
        
        vectorized_logis = np.vectorize(MyMath.logis_unvectorize)
        
        return vectorized_logis(x)

    
    def logis_de(x):
        ''' Derivative of the logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic derivative of 
                    the corresponding element in array x
        '''
        #logis'(x) = logis(x) * (1.0-logis(x)). 
        
        vectorized_logis_de = np.vectorize(MyMath.logis_de_unvectorize)
        
        return vectorized_logis_de(x)

    
    def iden(x):
        ''' Identity function
            Support vectorized operation
            
            x: an array type of real numbers
            return: the numpy array where every element is the same as
                    the corresponding element in array x
        '''
        return np.array(x)

    
    def iden_de(x):
        ''' The derivative of the identity function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of all ones of the same shape of x.
        '''
        return np.ones_like(x)
        

    def relu(x):
        ''' The ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of the same shape of x, where every element is the max of: zero vs. the corresponding element in x.
        '''
        vectorized_relu = np.vectorize(MyMath.relu_unvectorize)
        
        return vectorized_relu(x)

    
    def relu_de(x):
        ''' The derivative of the ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of the same shape of x, where every element is 1 if the correponding x element is positive; 0, otherwise. 
        '''
        vectorized_relu_de = np.vectorize(MyMath.relu_de_unvectorize)
        
        return vectorized_relu_de(x)
    
    
    def tanh_unvectorize(x):
        ''' tanh function. 
        '''
        return np.tanh(x)

    
    def tanh_de_unvectorize(x):
        ''' Derivative of the tanh function. 
        '''

        return 1.0 - (MyMath.tanh_unvectorize(x))**2.0

    
    def logis_unvectorize(x):
        ''' Logistic function. 
        '''

        return (MyMath.tanh_unvectorize(x/2.0) + 1.0) / 2.0
        
        

    
    def logis_de_unvectorize(x):
        ''' Derivative of the logistic function. 
        '''
        #logis'(x) = logis(x) * (1.0-logis(x)). 
        
        return MyMath.logis_unvectorize(x) * (1.0 - MyMath.logis_unvectorize(x))



    def relu_unvectorize(x):
        
        return 0 if x <= 0 else x

    
    def relu_de_unvectorize(x):
       
        return 1 if x > 0 else 0


    
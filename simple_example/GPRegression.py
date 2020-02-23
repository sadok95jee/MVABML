# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import scipy.linalg

# This file is attached with a jupyter notebook "GPRegression_Handcoded.ipynb"
# which deal with a simple case in order to illustrate the GP regression.
# It also contains details about the formulas we used to define our models


class GP:
    
    """ Defines the Gaussian Process which will be used for the regression.
        Specifies the kernel and its hyperparameters.
    """
    
    def __init__(self , theta , kernel):
        
        self.theta = theta
        self.kernel_type = kernel
        
    def prior_kernel(self , x):
        if self.kernel_type == "Gaussian":
            return (self.theta[0]**2)*np.exp(-0.5*(x/self.theta[1])**2)
        if self.kernel_type == "Matern":
            return (self.theta[0]**2)*np.exp(-x/self.theta[1])
        
    #NB: We only consider here a simplified version of the Matern kernel
    # with p=0 and v = 0.5 
    #(see. https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)
        
class GPRegression:
    
    """ Defines our model of regression for a given training set (X_train , Y_train)
    
        methods: "Full" (Exact GP regression) , "PP" (Sparse GP regression) 
                , "SPGP" (Sparse GP regression)
        kernels: "Gaussian" , "Matern"
        
        For the formulas of the posterior mean, the posterior covariance or the
        marginal log-likelihood we refered to:
            - http://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf (eq. (1) , (3))
            - http://www.gatsby.ucl.ac.uk/~snelson/SPGP_up.pdf (eq. (8))
    """
    
    
    def __init__(self, X_train , Y_train , method , kernel):
        self.X_train, self.Y_train = X_train.copy(), Y_train.copy()
        self.method = method
        self.kernel_type = kernel
        self.sample_size = X_train.shape[0]
        self.dimension = X_train.shape[1]

    def reset(self , theta , sigma , X_induced = None):
        self.theta = theta
        self.sigma = sigma
        self.gp = GP(theta , self.kernel_type)       
        self.Knn = None        
        if not(self.method == "Full"):
            self.X_induced = X_induced
            self.Kmn = None
            self.Km = None
            self.Km_cholesky = None
            self.Qm_cholesky = None
            self.Lambda = None
            
    def get_method(self):
        return self.method
    
    def get_X_train(self):
        return self.X_train
    
    def get_Y_train(self):
        return self.Y_train

    def get_X_induced(self):
        return self.X_induced      

    def K_matrix(self , X , Y):
        return self.gp.prior_kernel(cdist(X , Y))
            
    def Nystrom(self):
        return np.dot(self.Kmn.T , np.dot(np.linalg.inv(self.Km) , self.Kmn ))
    
    def update(self):  
        
        """ Computes all the matrices needed to compute the posterior mean,
            the posterior covariance and the marginal log-likelihood (except 
            the matrix Qm_inv that we deal with in the next fuction).
            
            For the formulas see the articles above (the names of the matrices
            should match)
        """
        if self.method == "Full":
            self.Knn = self.K_matrix(self.X_train , self.X_train) 
        else:
            self.Km = self.K_matrix(self.X_induced , self.X_induced) + 10**(-8)*np.identity(self.X_induced.shape[0])
            self.Km_cholesky = np.linalg.cholesky(self.Km)
            self.Kmn = self.K_matrix(self.X_induced , self.X_train)            
            if self.method == "SPGP":
                self.Knn = self.K_matrix(self.X_train , self.X_train) 
                self.Lambda = np.diag(np.diag(self.Knn - self.Nystrom()))                   
        return 
    
    def update_Qm_cholesky(self):
        
        """ Computes the Cholesky decomposition of the matrix Qm that we only 
            need for the posterior mean and the posterior variance.
            
            We want to avoid unnecessary computations during the optimization
            phase.
        """
        
        sigma = self.sigma
        n = self.sample_size       
        if self.method == 'PP':
            temp = self.Km + (1/sigma**2)*np.dot(self.Kmn , self.Kmn.T)
        elif self.method =='SPGP':
            temp = self.Km + np.dot(self.Kmn , np.dot(np.linalg.inv(self.Lambda + sigma**2*np.identity(n)) , self.Kmn.T)) 
        self.Qm_cholesky = np.linalg.cholesky(temp)
    
    def cholesky_cov(self):
        
        """ Computes the Cholesky decomposition of the exact covariance matrix
        ("Full) or its approximate version ("PP" , "SPGP")
        """
        
        if self.method == 'PP':
            cov = self.Nystrom()
        elif self.method == 'SPGP':
            cov = self.Lambda + self.Nystrom()
        elif self.method == 'Full':
            cov =  self.Knn
        result = cov + (self.sigma**2)*np.identity(self.sample_size)
        return np.linalg.cholesky(result)
    
    def marginal_log_likelihood(self):
        
        """ Computes the marginal log-likelihood p(Y_train | X_train , theta) 
            with the exacct covariance matrix ("Full") or its approximate
            version ("PP" , "SPGP")
        """
        
        n = self.sample_size
        L = self.cholesky_cov()
        temp = scipy.linalg.cho_solve((L , True) , self.Y_train)
        return -0.5*(np.vdot(temp.T, temp) + 2*np.sum(np.log(np.diag(L)))
             + n*np.log(2*np.pi)
             )
    
    def posterior_mean(self , X):
        
        """ Computes the posterior mean of the Gaussian process for
            new observations X.
        """
        
        sigma = self.sigma
        n = self.sample_size
        if self.method == "Full":
            Kx = self.K_matrix(X , self.X_train)
            L = self.cholesky_cov()
            temp = scipy.linalg.cho_solve((L , True) , self.Y_train)
        elif self.method == "PP":
            Kx = self.K_matrix(X , self.X_induced)
            temp = (1/sigma**2)*(scipy.linalg.cho_solve((self.Qm_cholesky , True) 
                                                        , np.dot(self.Kmn , self.Y_train))
                                 )
        elif self.method == "SPGP":
            Kx = self.K_matrix(X , self.X_induced)
            tempbis = np.linalg.inv(self.Lambda + sigma**2*np.identity(n))
            temp = scipy.linalg.cho_solve((self.Qm_cholesky , True) ,
                                             np.dot(self.Kmn , 
                                                    np.dot(tempbis , self.Y_train)
                                                    )
                                           )
        return np.dot(Kx , temp)
    
    def posterior_cov(self , X):
        
        """ Computes the posterior covariance matrix of the Gaussian process
            for new observations X.
        """
        
        Kxx = self.K_matrix(X, X)
        if self.method == "PP" or self.method == "SPGP":
            Kxm = self.K_matrix(X, self.X_induced)
            temp = ( scipy.linalg.cho_solve((self.Km_cholesky , True) , Kxm.T) 
                    - scipy.linalg.cho_solve((self.Qm_cholesky , True) , Kxm.T) ) 
            return Kxx - np.dot(Kxm , temp) + self.sigma**2
        else:
            Kxn = self.K_matrix(X, self.X_train)
            temp = np.linalg.solve(self.cholesky_cov() , Kxn.T)
            return Kxx - np.dot(temp.T, temp) + self.sigma**2
 

    
def optimize_hyperparameters(GPR , theta0 , sigma0 , X_induced0 = None):
    
    """ Optimizes the hyperparameters of the model using a scipy method
        (Quasi - Newton algorithm).
        
        The objective function reset the GPR model for new values of the
        hyperparameters and then computes - marginal log-likelihood.
        
        theta0 , sigma0 , X_induced0 are given in order to initialize the
        minimize method.
        
        NB: Currently we encounter difficulties when it comes to optimize the
            variance sigma. So, we consider for the moment a fixed sigma.
            Uncomment and delete the each line below to recover the full 
            optimization.
    """
    if GPR.get_method() == "Full":
        def objective(x):
            GPR.reset(x[:-1] , x[-1])
            GPR.update()
            return  - GPR.marginal_log_likelihood()
        init = np.hstack((theta0 , sigma0))
        result = minimize(objective , init , method='BFGS',  options={'gtol': 1e-6, 'disp': True})
        print("theta* = " , result.x[:-1] , "sigma* = " , result.x[-1])
        return result.x[:-1] , result.x[-1]
    else:
        def objective(x):
            GPR.reset(x[:2] , x[2] , x[3:].reshape(-1 , 1))
            GPR.update()
            return - GPR.marginal_log_likelihood()
        init = np.hstack((theta0 , sigma0 ,X_induced0))
        result = minimize(objective , init , method='BFGS',  options={'gtol': 1e-6, 'disp': True})
        print("theta* =" , result.x[:2] , "sigma* = " , result.x[2])
        return result.x[:2] , result.x[2] , result.x[3:].reshape(-1 , 1)
        
        


def Predict(GPR , X_test , theta0 , sigma0 , m = None):
    
    """ Main function:        
        Optimizes the hyperparameters of our regression model and then, given
        a set of new observations X_test, computes the posterior mean and the
        posterior standard deviation (with the optimal hyperparameters).
        
        In the cases "PP" or "SPGP" also returns the initial set of induced
        points for plotting purposes.
    """
    
    method = GPR.get_method()
    if method == "Full":
        theta , sigma = optimize_hyperparameters(GPR , theta0 , sigma0)
        PM = GPR.posterior_mean(X_test)
        std = np.sqrt(np.diag(GPR.posterior_cov(X_test))).reshape(PM.shape)
        return PM , std
    else:
        X_train = GPR.get_X_train()
        X_induced0 = X_train[np.random.choice(X_train.shape[0],size = m) , :].reshape(-1)
        theta , sigma , X_induced = optimize_hyperparameters(GPR , theta0 , sigma0  ,X_induced0)
        GPR.update_Qm_cholesky()
        PM = GPR.posterior_mean(X_test)
        std = np.sqrt(np.diag(GPR.posterior_cov(X_test))).reshape(PM.shape)
        return PM , std , X_induced0
    

def plot_regression(GPR , X_test , PM , std , X_induced0 = None):
    
    """ Plots the results (posterior mean and posterior standard deviation)
        in a the same way as the Figure 1 of Snelson et al. or the Figure 1
        of Titsias.
    """  
    
    method = GPR.get_method()
    
    plt.figure(figsize=(10 , 6))
    plt.scatter(GPR.get_X_train() , GPR.get_Y_train() , marker = 'x' ,label = 'Training data')
    plt.plot(X_test , PM , 'b' , label = 'mean prediction')
    plt.plot(X_test , PM -std , '--' ,  color='red' , label = 'Standard deviation')
    plt.plot(X_test , PM + std , '--' , color='red')   
    if (method == 'PP') or (method == 'SPGP'):
        plt.scatter(X_induced0 , (max(PM + std) + 0.5)*np.ones(X_induced0.shape[0]) 
                             , marker='+' , color='k' , label = 'Initial induced points' )
        plt.scatter(GPR.get_X_induced() , (min(PM - std) - 0.5)*np.ones(X_induced0.shape[0]) 
                                    , marker='+' , color='r' ,  label = 'Induced points' )
    plt.legend()
    plt.title(method + " Gaussian Process Regression")
    return


        
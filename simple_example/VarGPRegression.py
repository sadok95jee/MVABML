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
# which deal with a simple case in order to illustrate the  variational GP regression.
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
        
class VarGPRegression:
    
    """ Defines our model of regression for a given training set (X_train , Y_train)
    
        For the formulas of the posterior mean, the posterior covariance or the
        variational lower bound, we refered to:
            - http://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf 
    """
    
    def __init__(self , X_train , Y_train , kernel):
        self.X_train, self.Y_train = X_train.copy(), Y_train.copy()
        self.kernel_type = kernel
        self.sample_size = X_train.shape[0]    
        self.selection = []
        self.indices = [*range(self.sample_size)]
        
    def reset_prior(self , theta , sigma):
        self.theta = theta
        self.sigma = sigma
        self.gp = GP(theta , self.kernel_type)
        self.Knn = self.K_matrix(self.X_train , self.X_train)  
        
        self.X_induced = None
        self.index_induced = None     
        self.Km = None
        self.Km_cholesky = None
        self.Kmn = None
        self.A = None
        self.mu = None
        
    def update_induced(self , indexes):
        
        """ Computes all the matrices that will be needed to obtain the variationnal 
            lower bound.
            
            indexes:  the subset of training points which will be used as 
                      induced points for our regression             
        """
        self.index_induced = indexes
        self.X_induced = self.X_train[self.index_induced , :].copy()
        
        self.Km = self.K_matrix(self.X_induced , self.X_induced) + 10**(-8)*np.identity(self.X_induced.shape[0])
        self.Km_cholesky = np.linalg.cholesky(self.Km)
        self.Kmn = self.K_matrix(self.X_induced , self.X_train)
        
    def update_var_dist(self):
        
        """ Computes the mean mu and the covariance A of the variational distribution
            phi (see the article of Titisias).
            This distribution is needed to compute the posterior mean and covariance
        """
        
        Sigma_mat = self.Km + (1/self.sigma**2)*np.dot(self.Kmn , self.Kmn.T)
        L = np.linalg.cholesky(Sigma_mat)
        self.mu = (1/self.sigma**2)*np.dot(self.Km , 
                            scipy.linalg.cho_solve((L, True) , np.dot(self.Kmn , self.Y_train)))
        self.A = np.dot(self.Km , scipy.linalg.cho_solve((L , True) , self.Km))
                

    def K_matrix(self , X , Y):
        return self.gp.prior_kernel(cdist(X , Y))
    
            
    def covariance_cholesky(self):
        
        """ Computes the Cholesky decomposition of the approximate covariance matrix
            given by the Nyström formula.
        """
        
        Qnn = np.dot(self.Kmn.T , 
                     scipy.linalg.cho_solve((self.Km_cholesky , True) , self.Kmn))
        return np.linalg.cholesky(Qnn + (self.sigma**2)*np.identity(self.sample_size))
    
    def trace_term(self):
        
        """ Computes the regularization trace term that is added to the marginal
            log-likelihood in Titsias 2009.
            We only use the training points that are not induced points (see Titsias
            section 3.1)
        """
        
        X = np.delete(self.X_train , self.index_induced , axis=0)
        Kpp = self.K_matrix(X , X)   #p = (n - m)
        Kpm = self.K_matrix(X, self.X_induced)
        return np.trace(Kpp - np.dot(Kpm , 
                        scipy.linalg.cho_solve((self.Km_cholesky , True) , Kpm.T)))
    
    def variationnal_bound(self):
        
        """ Computes the variational lower bound (FV in Titsias) that will be used
            in the greedy selection algorithm (select induced points and optimize
            hyperparameters)
        """
        
        n = self.sample_size
        L = self.covariance_cholesky()
        temp = scipy.linalg.cho_solve((L , True) , self.Y_train)
        return -0.5*(np.vdot(self.Y_train, temp) + 2*np.sum(np.log(np.diag(L)))
             + n*np.log(2*np.pi)
             ) - (1/2*self.sigma**2)*self.trace_term()
    
    
    def posterior_mean(self , X):
        
        """ Computes the posterior mean of the Gaussian process for
            new observations X.
        """
        
        Kxm = self.K_matrix(X , self.X_induced)
        return np.dot(Kxm ,
                      scipy.linalg.cho_solve((self.Km_cholesky , True) , self.mu))
        
    def posterior_cov(self , X):
        
        """ Computes the posterior covariance matrix of the Gaussian process
            for new observations X.
        """
        
        Kxx = self.K_matrix(X, X)
        Kxm = self.K_matrix(X, self.X_induced)
        temp = scipy.linalg.cho_solve((self.Km_cholesky , True) , Kxm.T)
        tempbis = scipy.linalg.cho_solve((self.Km_cholesky , True) , self.A)
        return Kxx - np.dot(Kxm , temp) + np.dot(Kxm , np.dot(tempbis , temp)) + self.sigma**2


  
class Greedy(VarGPRegression):
    
    """ Given a variational GP regression model, defines the optimization
        strategy for its hyperparameters and induced points (see Titsias section 3.1).
        
        This strategy can be viewed as an EM algorithm:
            - E step: select a new induced point (among the the training points
              that are not part of the inducing set)
            - M step: optimize the hyperparameters thanks to the variational lower bound                     
    """
    
    def E_step(self , J): 
        
        """ Given a subset of indexes J (which defines a subset of training points)
            chooses the one which gives the highest variational lower bound.
            
            Updates the regression model adding this point to the final selection 
            of induced points (self.selection) and removing it from the list of 
            training points that we can choose a new induced point from (self.indices)
        """
        
        result = J[0]
        self.update_induced(self.selection + [J[0]])
        temp = self.variationnal_bound()
        for j in J[1:]:
            self.update_induced(self.selection + [j])
            if temp < self.variationnal_bound():
                result = j
        self.selection.append(result)
        self.indices.remove(result)
     
    def M_step(self , theta0 , sigma0):
        
        """ Optimizes the hyperparameters of the model using a scipy method
        (Quasi - Newton algorithm).
        
        The objective function reset the model for new values of the hyperparameters 
        and then computes -variational lower bound.
        
        theta0 and sigma0 are given in order to initialize the minimize method.
        """
        
        def objective(x):
            self.reset_prior(x[:-1] , x[-1])
            self.update_induced(self.selection)
            return -self.variationnal_bound()
        init = np.hstack((theta0 , sigma0))
        result = minimize(objective , init , method='BFGS',  options={'gtol': 1e-5, 'disp': True})
        print("theta* = " , result.x[:-1] , "sigma* = " , result.x[-1])
        return result.x[:-1] , result.x[-1] , - result.fun
    
    def Greedy_selection(self , nbr_induced_points , theta0 , sigma0 , size):
        
        """ EM algorithm (see Titsias section 3.1) with "nbr_induced_points" steps
            
            At each step randomly select a subset of the training points that are 
            not part of the inducing set of size = size.
            
            Returns the value of the variational lower bound FV at each step.
        """
        
        FV = []
        self.reset_prior(theta0 , sigma0)
        for m in range(nbr_induced_points):
            J = np.random.choice(self.indices , size = size)
            self.E_step(J)
            theta , sigma , f = self.M_step(theta0 , sigma0)
            FV.append(f)
        return FV


def predict(Greedy , X_test , theta0 , sigma0 , m , size):
    
    """ Main function:        
        Optimizes the hyperparameters of our regression model and select m 
        induced points via a greedy strategy. 
        Then, given a set of new observations X_test, computes the posterior 
        mean and the posterior standard deviation (with the optimal hyperparameters).
        
        m: number of induced points
        size: size of the random subset of training points in which a new induced point
              is chosen at each step
        theta0 , sigma0: initial hyperparameters for the minimize method at each step
    """
    
    ML = Greedy.Greedy_selection(m , theta0 , sigma0 , size)
    Greedy.update_var_dist()
    PM = Greedy.posterior_mean(X_test)
    std = np.sqrt(np.diag(Greedy.posterior_cov(X_test))).reshape(PM.shape)
    return PM , std , ML

def plot_regression(Greedy , X_test , PM , std , ML):
    
    """ Plots the results (posterior mean and posterior standard deviation)
        in a the same way as the Figure 1 of Titsias.
        
        Plots the lower variational bound w.r.t the number of induced points.
    """  
    
    plt.figure(figsize=(20 , 7))
    plt.subplot(1 , 2 , 1)
    plt.scatter(getattr(Greedy , "X_train") , getattr(Greedy , "Y_train") , marker = 'x' ,label = 'Training data')
    plt.plot(X_test , PM , 'b' , label = 'mean prediction')
    plt.plot(X_test , PM -2*std , '--' ,  color='red' , label = 'Standard deviation')
    plt.plot(X_test , PM + 2*std , '--' , color='red')  
    plt.scatter(getattr(Greedy , "X_induced") , (min(PM - std) - 0.5)*np.ones(getattr(Greedy , "X_induced").shape[0]) 
                                    , marker='+' , color='r' ,  label = 'Induced points' )
    plt.legend(loc = 'lower right')
    plt.title("Variational Gaussian Process Regression")
    
    plt.subplot(1 , 2 , 2)
    plt.plot(range(1 , len(ML)+1) , ML)
    plt.scatter(range(1 , len(ML)+1) , ML)
    plt.title("Variational lower bound")
    plt.xlabel("number of induced points")
    return 
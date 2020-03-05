import torch
from torch import optim
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.kernels.white_noise_kernel import WhiteNoiseKernel
from gpytorch import mlls
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


class MyGPModel(ExactGP):
    
    def __init__(self, train_inputs, train_targets, likelihood):
        super().__init__(train_inputs, train_targets, likelihood)
        if train_inputs.ndim == 2:
            dims = train_inputs.shape[1]
        else:
            dims = 1
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=dims))
        # self.covar_module = ScaleKernel(MaternKernel(ard_num_dims=dims, nu=2.5))
    
    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return MultivariateNormal(mean, cov)

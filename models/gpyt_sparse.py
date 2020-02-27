"""Use the InducingPointKernel to fit a sparse GP following Titsias (2009)."""
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from models.NystromKernel import NystromKernel


class TitsiasSparseGP(ExactGP):
    """The variational posterior is supposed to be an actual, exact GP but with a different kernel structure."""

    def __init__(self, train_inputs, train_targets, inducing_points, likelihood):
        super().__init__(train_inputs, train_targets, likelihood)
        
        self.mean_module = gpytorch.means.ZeroMean()
        self.base_cov_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(self.base_cov_module, inducing_points, likelihood)
    
    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return MultivariateNormal(mean, cov)


class NystromSparseGP(ExactGP):

    def __init__(self, train_inputs, train_targets, inducing_points, likelihood):
        super().__init__(train_inputs, train_targets, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()
        self.base_cov_module = ScaleKernel(RBFKernel())
        self.covar_module = NystromKernel(
            self.base_cov_module, inducing_points, likelihood)

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return MultivariateNormal(mean, cov)


if __name__ == "__main__":
    # model  = TitsiasSparseGP()
    pass

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class VarsparseGPModel(ApproximateGP):
    """
    Implementation of Hensman's [2013] variational Gaussian Process strategy
    using stochastic variational inference.
    
    See: https://arxiv.org/pdf/1309.6835.pdf
    """
    def __init__(self, inducing_points):
        if (inducing_points.ndim ==2):
            dims = inducing_points.shape[1]
        else:
            dims = 1
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(VarsparseGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = dims))
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

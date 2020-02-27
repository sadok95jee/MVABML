import torch
import gpytorch
import math
import copy
from gpytorch.kernels import Kernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.lazy import DiagLazyTensor, MatmulLazyTensor, PsdSumLazyTensor, RootLazyTensor, delazify



class NystromKernel(Kernel):
    """
    Inducing point kernel for the Nystrom approximation of GPs.
    
    Code lifted from GPyTorch's own `InducingPointKernel` which uses Titsias' (2009)
    added loss term.
    """
    
    def __init__(self, base_kernel, inducing_points, likelihood, active_dims=None):
        super(NystromKernel, self).__init__(active_dims=active_dims)
        self.base_kernel = base_kernel
        self.likelihood = likelihood

        if inducing_points.ndimension() == 1:
            inducing_points = inducing_points.unsqueeze(-1)

        self.register_parameter(name="inducing_points",
                                parameter=torch.nn.Parameter(inducing_points))

    def train(self, mode=True):
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        return super(NystromKernel, self).train(mode)

    @property
    def _inducing_mat(self):
        if not self.training and hasattr(self, "_cached_kernel_mat"):
            return self._cached_kernel_mat
        else:
            res = delazify(self.base_kernel(
                self.inducing_points, self.inducing_points))
            if not self.training:
                self._cached_kernel_mat = res
            return res

    @property
    def _inducing_inv_root(self):
        if not self.training and hasattr(self, "_cached_kernel_inv_root"):
            return self._cached_kernel_inv_root
        else:
            chol = psd_safe_cholesky(self._inducing_mat, upper=True)
            eye = torch.eye(chol.size(-1), device=chol.device,
                            dtype=chol.dtype)
            inv_root = torch.triangular_solve(eye, chol)[0]

            res = inv_root
            if not self.training:
                self._cached_kernel_inv_root = res
            return res

    def _get_covariance(self, x1, x2):
        k_ux1 = delazify(self.base_kernel(x1, self.inducing_points))
        if torch.equal(x1, x2):
            covar = RootLazyTensor(k_ux1.matmul(self._inducing_inv_root))

            # Diagonal correction for predictive posterior
            correction = (self.base_kernel(x1, x2, diag=True) -
                          covar.diag()).clamp(0, math.inf)
            covar = PsdSumLazyTensor(covar, DiagLazyTensor(correction))
        else:
            k_ux2 = delazify(self.base_kernel(x2, self.inducing_points))
            covar = MatmulLazyTensor(
                k_ux1.matmul(self._inducing_inv_root), k_ux2.matmul(
                    self._inducing_inv_root).transpose(-1, -2)
            )

        return covar

    def _covar_diag(self, inputs):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)

        # Get diagonal of covar
        covar_diag = delazify(self.base_kernel(inputs, diag=True))
        return DiagLazyTensor(covar_diag)

    def forward(self, x1, x2, diag=False, **kwargs):
        covar = self._get_covariance(x1, x2)

        if self.training:
            if not torch.equal(x1, x2):
                raise RuntimeError("x1 should equal x2 in training mode")
        if diag:
            return covar.diag()
        else:
            return covar

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def __deepcopy__(self, memo):
        replace_inv_root = False
        replace_kernel_mat = False

        if hasattr(self, "_cached_kernel_inv_root"):
            replace_inv_root = True
            kernel_inv_root = self._cached_kernel_inv_root
        if hasattr(self, "_cached_kernel_mat"):
            replace_kernel_mat = True
            kernel_mat = self._cached_kernel_mat

        cp = self.__class__(
            base_kernel=copy.deepcopy(self.base_kernel),
            inducing_points=copy.deepcopy(self.inducing_points),
            likelihood=self.likelihood,
            active_dims=self.active_dims,
        )

        if replace_inv_root:
            cp._cached_kernel_inv_root = kernel_inv_root

        if replace_kernel_mat:
            cp._cached_kernel_mat = kernel_mat

        return cp



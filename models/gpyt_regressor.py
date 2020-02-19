import torch
from torch import optim
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP
from gpytorch import mlls
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


class MyGPModel(ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.MaternKernel()
    
    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return MultivariateNormal(mean, cov)


if __name__ == "__main__":
    # Model likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    X_all, y_all = load_boston(return_X_y=True)
    X_all = torch.from_numpy(X_all).float()
    y_all = torch.from_numpy(y_all).float()
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=.2)
    
    print("No. of covariates:", X_train.shape[1])
    print("Train size:", X_train.shape[0])
    print("Valid size:", X_val.shape[0])

    model = MyGPModel(X_train, y_train, likelihood)
    model.train()
    likelihood.train()
    
    # Step 2: define marginal log likelihood
    marg_ll = mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    epochs = 80
    optimizer = optim.Adam(model.parameters(), lr=1.4)
    
    import tqdm
    eprange = tqdm.tqdm(range(epochs))
    for i in eprange:
        optimizer.zero_grad()
        output = model(X_train)
        loss = -marg_ll(output, y_train)
        loss.backward()
        
        eprange.write("Loss {:.4g}".format(loss.data))
        optimizer.step()
    
    
    model.eval()
    likelihood.eval()
    
    def make_prediction(model, x_pred):
        """Return predictive distribution at points `x_pred`."""
        return likelihood(model(x_pred))

    with torch.no_grad():
        y_pred_dist = make_prediction(model, X_val)
        y_pred_mean = y_pred_dist.mean
    print(y_val)
    print(y_pred_mean)

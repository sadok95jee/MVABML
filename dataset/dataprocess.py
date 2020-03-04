import numpy as np 
from sklearn.decomposition import PCA

X = np.genfromtxt("sdss.csv" , delimiter=",")
Y = X[:,-1]
X = X[:,:-1]
X[:,5:] = np.log(X[:,5:])
X_projected = PCA(n_components=10).fit_transform(X)

np.savetxt("process_sdss.csv" , np.hstack((X_projected,Y[: , None])) , delimiter=",")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

def gaussianProcess(data_only,loss_data,other_half_data):
    kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    gp.fit(data_only, loss_data)
    y_mean, y_cov = gp.predict(other_half_data[:,:-1],return_cov=True)
    return y_mean,y_cov
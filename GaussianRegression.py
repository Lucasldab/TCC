from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import numpy as np

def gaussianProcess(data_only,loss_data,other_half_data,smallest_loss_local):
    kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    gp.fit(data_only, loss_data)

    surrogate_values = []

    for X in range (len(other_half_data)):
        y_mean, y_cov = gp.predict(other_half_data[X:X+1,:-1], return_cov=True)
        dist = norm(loc=0.0,scale=1.0)
        sigma_tilde = (y_cov[0][0])**(1/2)
        u = (loss_data[smallest_loss_local] - y_mean[0])/sigma_tilde
        ei = sigma_tilde*((u * dist.cdf(u))+dist.pdf(u))
        surrogate_values.append(-ei)
    return np.array(surrogate_values)

def gaussianPSO(best_value,end_particles_position):
    kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

    surrogate_values = []

    for X in range (len(end_particles_position)):
        y_mean, y_cov = gp.predict(end_particles_position[X:X+1,:], return_cov=True)
        dist = norm(loc=0.0,scale=1.0)
        sigma_tilde = (y_cov[0][0])**(1/2)
        u = (best_value - y_mean[0])/sigma_tilde
        ei = sigma_tilde*((u * dist.cdf(u))+dist.pdf(u))
        surrogate_values.append(-ei)
    return np.array(surrogate_values)
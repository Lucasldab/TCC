from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import numpy as np


class GaussianRegression:
    def __init__(self,loss_data = [],other_half_data = [],smallest_loss_local = [],surrogate_values = [], data_only = [], best_value= 0,end_particles_position= 0):
        self.surrogate_values = surrogate_values
        self.loss_data = loss_data
        self.other_half_data = other_half_data
        self.smallest_loss_local = smallest_loss_local
        self.data_only = data_only
        self.best_value = best_value
        self.end_particles_position = end_particles_position
        #kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
        #self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
        #self.gp.fit(self.data_only, self.loss_data)
        #self.mn = np.min(self.loss_data)

    def expectedImprovement(self, loss_data = None,other_half_data = None,smallest_loss_local = None,best_value = None,end_particles_position = [],y_mean = None,y_cov = None,mn = 0):
        if y_mean is not None:
            dist = norm(loc=0.0,scale=1.0)
            sigma_tilde = (y_cov)**(1/2)
            u = (mn - y_mean)/sigma_tilde
            ei = sigma_tilde*((u * dist.cdf(u))+dist.pdf(u))
            return -ei
        elif best_value is not None:
            for X in range (len(end_particles_position)):
                y_mean, y_cov = self.gp.predict(end_particles_position[X:X+1,:], return_cov=True)
                dist = norm(loc=0.0,scale=1.0)
                sigma_tilde = (y_cov[0][0])**(1/2)
                u = (best_value - y_mean[0])/sigma_tilde
                ei = sigma_tilde*((u * dist.cdf(u))+dist.pdf(u))
                self.surrogate_values.append(-ei)
        else:
            for X in range (len(other_half_data)):
                y_mean, y_cov = self.gp.predict(other_half_data[X:X+1,:-1], return_cov=True)
                dist = norm(loc=0.0,scale=1.0)
                sigma_tilde = (y_cov[0][0])**(1/2)
                u = (loss_data[smallest_loss_local] - y_mean[0])/sigma_tilde
                ei = sigma_tilde*((u * dist.cdf(u))+dist.pdf(u))
                self.surrogate_values.append(-ei)
        return np.array(self.surrogate_values)

    def gaussianProcess(self,data_only,loss_data,other_half_data,smallest_loss_local,best_value = False,end_particles_position = False):
        if best_value == False:
            self.gp.fit(data_only, loss_data)
            return self.expectedImprovement(loss_data,other_half_data,smallest_loss_local)
        else:
            return self.expectedImprovement(best_value,end_particles_position)
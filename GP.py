from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import numpy as np

class GP():

    def __init__(self, X, y) -> None:
        self. kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
        self.X=X
        self.y=y
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, normalize_y=True)
        self.fitted_values = self.gp.fit(X, y)
        self.mn = np.min(y)
    
    def refit(self,new_X, new_y):
        try:
            self.fitted_values = self.gp.fit(new_X, new_y)
            self.mn = np.min(new_y)
            return "Success"
        except:
            return "Failure"
        
    def expected_improvement(self, x:np.ndarray) -> np.float16:
        y_mean, y_cov = self.gp.predict(x, return_cov=True)
        dist = norm(loc=0.0,scale=1.0)
        sigma_tilde = (y_cov)**(1/2)
        u = (self.mn - y_mean)/sigma_tilde
        ei = sigma_tilde*((u * dist.cdf(u))+dist.pdf(u))
        return -ei
from scipy.stats import norm
import numpy as np

def expected_improvement(y_mean,y_cov,loss_data,smallest_loss_local,other_half_data):

    for X in range (len(other_half_data)):
        y_mean, y_cov = gp.predict(other_half_data[X:X+1,:-1], return_cov=True)
        dist = norm(loc=0.0,scale=1.0)
        sigma_tilde = (y_cov[0][0])**(1/2)
        u = (loss_data[smallest_loss_local] - y_mean[0])/sigma_tilde
        ei = sigma_tilde*((u * dist.cdf(u))+dist.pdf(u))
        surrogate_values.append(-ei)
    return surrogate_values
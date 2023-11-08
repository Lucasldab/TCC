from scipy.stats import norm


def expected_improvement(y_mean,y_cov,loss_data):
    dist = norm(loc=0.0,scale=1.0)
    sigma_tilde = (y_cov)**(1/2)
    u = (loss_data[min_loc] - y_mean[X])/sigma_tilde
    ei = sigma_tilde*((u * dist.cdf(u))+dist.pdf(u))
    surrogate_values = -ei

return surrogate_values
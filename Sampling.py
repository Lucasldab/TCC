from skopt.sampler import Lhs
import numpy as np
from skopt.space import Space

def sampling_space(n_samples,minLim,maxLim):
    n_samples = n_samples
    space = Space([(minLim, maxLim)])




lhs = Lhs(lhs_type='classic', criterion=None)
x = lhs.generate(space.dimenensions, n_samples)
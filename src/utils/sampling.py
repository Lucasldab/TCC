from skopt.sampler import Lhs
from skopt.sampler import Grid
from skopt.sampler import Sobol
from skopt.sampler import Hammersly
from skopt.sampler import Halton
import numpy as np
from skopt.space import Space
from scipy.spatial.distance import pdist

def lhs_sampling(n_samples,minLim,maxLim,round=True):
    space = Space([(minLim, maxLim)])
    lhs = Lhs(lhs_type='classic', criterion=None)
    samplings = lhs.generate(space.dimensions, n_samples)
    if (round==False):
        samplings = np.array(samplings).flatten()
    else:
        samplings = np.array(samplings).flatten().astype(int)
    return samplings

def grid_sampling(n_samples,minLim,maxLim,round=True):
    space = Space([(minLim, maxLim)])
    grid = Grid(border="include", use_full_layout=False)
    samplings = grid.generate(space.dimensions, n_samples)
    if (round==False):
        samplings = np.array(samplings).flatten()
    else:
        samplings = np.array(samplings).flatten().astype(int)
    return samplings

def random_sampling(n_samples,minLim,maxLim,round=True):
    space = Space([(minLim, maxLim)])
    samplings = space.rvs(n_samples)
    if (round==False):
        samplings = np.array(samplings).flatten()
    else:
        samplings = np.array(samplings).flatten().astype(int)
    return samplings

def halton_sampling(n_samples,minLim,maxLim,round=True):
    space = Space([(minLim, maxLim)])
    halton = Halton()
    samplings = halton.generate(space.dimensions, n_samples)
    if (round==False):
        samplings = np.array(samplings).flatten()
    else:
        samplings = np.array(samplings).flatten().astype(int)
    return samplings

def sobol_sampling(n_samples,minLim,maxLim,round=True):
    space = Space([(minLim, maxLim)])
    sobol = Sobol()
    samplings = sobol.generate(space.dimensions, n_samples)
    if (round==False):
        samplings = np.array(samplings).flatten()
    else:
        samplings = np.array(samplings).flatten().astype(int)
    return samplings

def hammersly_sampling(n_samples,minLim,maxLim,round=True):
    space = Space([(minLim, maxLim)])
    hammersly = Hammersly()
    samplings = hammersly.generate(space.dimensions, n_samples)
    if (round==False):
        samplings = np.array(samplings).flatten()
    else:
        samplings = np.array(samplings).flatten().astype(int)
    return samplings
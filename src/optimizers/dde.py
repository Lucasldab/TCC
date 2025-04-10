import numpy as np
import GaussianRegression
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class DiscreteDifferentialEvolution:
    def __init__(self, X, y, population,population_size, dimension, bounds, discrete_mask, crossover_rate=0.9, scaling_factor=0.8, max_generations=1000, tol=1e-6):
        self.population = population
        self.population_size = population_size
        self.dimension = dimension
        self.bounds = bounds
        self.discrete_mask = discrete_mask
        self.h = 100
        self.crossover_rate = crossover_rate
        self.scaling_factor = scaling_factor
        self.max_generations = max_generations
        self.tol = tol
        self.kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, normalize_y=True)
        self.gp.fit(X, y)
        self.mn= np.min(y)
        self.GaussianRegression = GaussianRegression.GaussianRegression(other_half_data= X,
                                            loss_data=y,smallest_loss_local=self.mn)

    def mutate(self, population, target_index):
        indices = list(range(len(population)))
        indices.remove(target_index)
        selected = np.random.choice(indices, size=3, replace=False)
        a, b, c = population[selected]
        mutant = np.clip(a + self.scaling_factor * (b - c), self.bounds[0], self.bounds[1])
        return mutant

    def forward_transformation(self, target):
        transf = -1 + ((target * self.h * 5) / (1000 - 1))
        return np.where(self.discrete_mask, transf, target)

    def backward_transformation(self, target):
        transf =  np.round(((1+ target)*(1000-1))/(5 * self.h))
        return np.where(self.discrete_mask, transf, target)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.crossover_rate
        trial = np.where(crossover_mask, mutant, target)
        return trial
    
    def optimize(self):
        population = self.population
        #fitness evaluation step
        y_mean, y_cov = self.gp.predict(population, return_cov=True)
        y_cov = np.diag(y_cov)
        fitness = np.asarray([self.GaussianRegression.expectedImprovement(y_mean=ym,y_cov=yc,mn=self.mn) for ym,yc in zip(y_mean,y_cov)])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
    
        for generation in range(1, self.max_generations + 1):
            if np.any(np.abs(best_fitness) < self.tol):
                break
    
            new_population = np.zeros_like(population)
            temp_population = []
    
            for i in range(self.population_size):
                #forward transformation here
                fr = self.forward_transformation(population[i])
                mutant = self.mutate(population, i)
                trial = self.crossover(fr, mutant)
                #backward transformation here
                trial = self.backward_transformation(trial)
                temp_population.append(trial)
    
            #evaluation and substitution all at once
            temp_population = np.asarray(temp_population)
            y_mean_t, y_cov_t = self.gp.predict(temp_population, return_cov=True)
            y_cov_t = np.diag(y_cov_t)
            fitness_t = np.asarray([self.GaussianRegression.expectedImprovement(y_mean=ym,y_cov=yc,mn=self.mn) for ym,yc in zip(y_mean_t,y_cov_t)])
            
            mask = fitness_t <= fitness
            population[mask] = temp_population[mask]
            fitness[mask] = fitness_t[mask]
            
            #update best candidate and fitness
            best_id = np.argmin(fitness)
            best_solution = population[best_id]
            best_fitness = fitness[best_id]
            
        return best_solution, best_fitness
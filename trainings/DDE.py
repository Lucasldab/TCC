import numpy as np
#import GaussianRegression

class DiscreteDifferentialEvolution:
    def __init__(self, objective_function,population,population_size, dimension, bounds, discrete_mask, crossover_rate=0.9, scaling_factor=0.8, max_generations=1000, tol=1e-6, data_only=0,loss_data=0,other_half_data=0,smallest_loss_local=0):
        self.objective_function = objective_function
        #print('Objective Function: ',objective_function)
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
        fitness = self.objective_function ###Talvez esteja errado
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for generation in range(1, self.max_generations + 1):
            if np.abs(best_fitness) < self.tol:
                break

            new_population = np.zeros_like(population)
            #forward aqui

            for i in range(self.population_size): ########### Aqui
                mutant = self.mutate(population, i)
                trial = self.crossover(population[i], mutant)

                trial_fitness = self.objective_function(trial) ### Trial precisa ser adaptado
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                else:
                    new_population[i] = population[i]

            
            population = new_population
            #backward aqui
        return best_solution, best_fitness

# Example usage
def sphere_function(x):
    return np.sum(x**2)

# Define the objective function
def objective_function(x):
    return sphere_function(x)

# Create an instance of DiscreteDifferentialEvolution and optimize
#dde = DiscreteDifferentialEvolution(objective_function, population, population_size, dimension, bounds, discrete_mask)
#best_solution, best_fitness = dde.optimize()
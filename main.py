import dataTreatment
import GaussianRegression
import pandas as pd
import SurrogateFunction
import ParticleSwarmOptimization
import numpy as np

#Data treatment
#data = pd.read_csv('trainings/training_CNN_results_v3.csv')
clean_data = pd.read_csv('trainings/training_CNN_results_v3-1.csv')
half_data,other_half_data = dataTreatment.divide_samplings(clean_data)
loss_data,data_only,smallest_loss_local = dataTreatment.data_from_loss(half_data)

num_particles = 100

#Gaussian Process and Acquisition Function
surrogate_values = GaussianRegression.gaussianProcess(data_only,loss_data,other_half_data,smallest_loss_local)

fitness_values = surrogate_values[:num_particles]
particles_position = other_half_data[:num_particles,:-1]

print(surrogate_values)

# Example usage
max_iterations = 30
inertia_weight = 0.5
cognitive_coeff = 1.5
social_coeff = 1.5

# Run PSO algorithm with fitness values
for iterations in range(max_iterations):
    best_position, best_value, end_particles_position = ParticleSwarmOptimization.pso_with_fitness(fitness_values, particles_position, inertia_weight, cognitive_coeff, social_coeff)
    surrogate_values = GaussianRegression.gaussianPSO(best_value,end_particles_position)
    print(iterations+1)

print("Best Position:", best_position)
print("Best Value:", best_value)

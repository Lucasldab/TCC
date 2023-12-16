import dataTreatment
import gaussianRegression
import pandas as pd
import surrogateFunction
import particleSwarmOptimization
import numpy as np
import random
import os
import csv

class PSO_Generalized:
    def __init__(self, num_particles, dimensions, max_iter, fitness_values,particles_position):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.fitness_function = fitness_values
        
        # Definindo limites
        self.MinPosition_limit = [1.0, 16.0, 1.0, 0.0001, 16.0]
        self.MaxPosition_limit = [9.0, 64.0, 32.0, 0.01, 128.0]
        self.velocity=[]
        for dim in self.MaxPosition_limit:
            self.velocity.append(0.1 * (2 * dim))
        self.Maxvelocity_limit = self.velocity  # 10% do intervalo total
        self.MinVelocity_limit = 0.1

        # Inicializa posições e velocidades
        self.positions = particles_position
        self.velocities = np.random.uniform(self.MinVelocity_limit, self.Maxvelocity_limit, (num_particles, dimensions))

        # Inicializa a melhor posição local e global
        self.local_best_positions = np.copy(self.positions)
        self.local_best_fitnesses = self.fitness_function.copy()
        self.global_best_position = self.positions[np.argmin(self.local_best_fitnesses)]
        self.global_best_fitness = min(self.local_best_fitnesses)

    def apply_boundary_conditions(self):
        # Aplica as condições de fronteira às posições e velocidades
        for dim in range(self.dimensions):
            np.clip(self.positions[:, dim], self.MinPosition_limit[dim], self.MaxPosition_limit[dim], out=self.positions[:, dim])
            np.clip(self.velocities, self.MinVelocity_limit, self.Maxvelocity_limit, out=self.velocities)
        #np.clip(self.positions, self.MinPosition_limit, self.MaxPosition_limit, out=self.positions)
        #np.clip(self.velocities, self.MinVelocity_limit, self.Maxvelocity_limit, out=self.velocities)

    def optimize(self):
        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                fitness = self.fitness_function[i]
                # Atualiza a melhor posição local da partícula
                if fitness < self.local_best_fitnesses[i]:
                    self.local_best_fitnesses[i] = fitness
                    self.local_best_positions[i] = self.positions[i]

                # Atualiza a melhor posição global
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.positions[i]

            # Atualiza a velocidade e a posição de cada partícula
            r1, r2 = np.random.rand(self.num_particles, self.dimensions), np.random.rand(self.num_particles, self.dimensions)
            cognitive_velocities = 2.05 * r1 * (self.local_best_positions - self.positions)
            social_velocities = 2.05 * r2 * (self.global_best_position - self.positions)
            random_number = random.random()
            inertia_term = 0.5 + random_number / 2 
            self.velocities = inertia_term + self.velocities + cognitive_velocities + social_velocities
            self.positions += self.velocities

            # Aplica condições de fronteira
            self.apply_boundary_conditions()

            self.fitness_function = gaussianRegression.gaussianPSO(self.global_best_fitness,self.positions)

        return self.global_best_position, self.global_best_fitness


num_particles = int(input("How many particles for the PSO?"))
samplingMethod = input("Whitch sampling method?")
trainingFile = 'trainings/Fully_Connected_'+ samplingMethod+'/GRPSO_'+str(num_particles)+'particles.csv'

if not os.path.exists(trainingFile):
    with open(trainingFile, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name",  "Hidden_Layer1", "Hidden_Layer2", "Learning_Rate", "Batch_Size", "Best_Value"])
        file.close()

for datasetNumber in range(1,21):

    print('Training number: ',datasetNumber)

    data = pd.read_csv('trainings/Fully_Connected_'+ samplingMethod +'/training_'+ str(datasetNumber) +'.csv')
    clean_data = dataTreatment.clean_data(data)
    half_data,other_half_data = dataTreatment.divide_samplings(clean_data)
    loss_data,data_only,smallest_loss_local = dataTreatment.data_from_loss(half_data)

    half_data

    break

    #Gaussian Process and Acquisition Function
    print('Gaussian Regression Interpolation')
    surrogate_values = gaussianRegression.gaussianProcess(data_only,loss_data,other_half_data,smallest_loss_local)
    fitness_values = surrogate_values[:num_particles]
    particles_position = other_half_data[:num_particles,:-1]
    print("Starting PSO")

    # Exemplo de uso do PSO generalizado com a SMBO
    pso_generalized = PSO_Generalized(num_particles=num_particles, dimensions=5, max_iter=30,fitness_values=fitness_values,particles_position=particles_position)
    best_position, best_fitness = pso_generalized.optimize()
    print('Best position: ',best_position)
    print("Best Value:", best_fitness)

    with open(trainingFile, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([dataTreatment.decimalToName(int(round(best_position[0]))), int(round(best_position[1])), int(round(best_position[2])), best_position[3], int(round(best_position[4])), best_fitness])
        file.close()
    print("Saved")
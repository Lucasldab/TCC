import dataTreatment
import GaussianRegression
import pandas as pd
import ParticleSwarmOptimization
import numpy as np
import os
import csv
import DDE as DDE


num_particles = 100
samplingMethod = 'LHS'
samplesNumber = 20
trainingFile = 'trainings/CNN_'+ samplingMethod+'/GRPSO_'+str(num_particles)+'particles.csv'

if not os.path.exists(trainingFile):
    with open(trainingFile, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Convoluted_Layers1", "Convoluted_Filters1", "Convoluted_Layers2", "Convoluted_Filters2", "Hidden_Layer1", "Hidden_Layer2", "Learning_Rate", "Batch_Size", "Best_Value"])
        file.close()


for training in range(1, samplesNumber+1):

    print('Training number: ',training)

    #Data treatment
    data = pd.read_csv('trainings/CNN_'+ samplingMethod +'/training_'+ str(training) +'.csv')
    clean_data = dataTreatment.clean_data(data)
    half_data,other_half_data = dataTreatment.divide_samplings(clean_data)
    loss_data,data_only,smallest_loss_local = dataTreatment.data_from_loss(half_data)

    #Gaussian Process and Acquisition Function
    print('Gaussian Regression Interpolation')
    gr = GaussianRegression.GaussianRegression(data_only = data_only,loss_data = loss_data,other_half_data = other_half_data,smallest_loss_local = smallest_loss_local)

    surrogate_values = gr.gaussianProcess(data_only,loss_data,other_half_data,smallest_loss_local)

    fitness_values = surrogate_values[:num_particles]
    particles_position = other_half_data[:num_particles,:-1]

    print("Discrete Differential Evolution Optimization:")

    dde = DDE.DiscreteDifferentialEvolution(X=data_only,
                                            y=loss_data,
                                            population_size=num_particles,
                                            population=particles_position,
                                            dimension=9,
                                            bounds=(0, 1000),
                                            discrete_mask=np.array([True, False, False, False, False, False, False, False, False]),
                                            max_generations=1000)


    best_solution, best_fitness = dde.optimize()
    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)

    break

    #print(fitness_values)
    # Example usage
    max_iterations = 50
    #inertia_weight = 0.4
    #cognitive_coeff = 0.5
    #social_coeff = 0.5

    # Run PSO algorithm with fitness values
    print("Starting PSO with Gaussian Regression")
    for iterations in range(max_iterations):
        best_position, best_value, end_particles_position,cognitive_coeff,social_coeff = particleSwarmOptimization.pso_with_fitness(fitness_values, particles_position,cognitive_coeff=cognitive_coeff,social_coeff=social_coeff)#, inertia_weight, cognitive_coeff, social_coeff)
        surrogate_values = gaussianRegression.gaussianPSO(best_value,end_particles_position)
        if (iterations + 1) % 10 == 0:
            print(f"Processing iteration {iterations + 1}")

    print("Best Position without treatment:", best_position)
    print("Best Value:", best_value)

    

    with open(trainingFile, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([best_position[0], best_position[1], best_position[2], best_position[3], best_position[4], best_position[5], best_position[6], best_position[7], best_position[8], best_value])
                file.close()
    print("Saved")
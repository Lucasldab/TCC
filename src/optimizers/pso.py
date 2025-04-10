import numpy as np
import random

def linear_decrease(initial_value, current_iteration, max_iterations, max_value=0.0):
    slope = (max_value - initial_value) / max_iterations
    new_value = initial_value + slope * current_iteration
    return max(new_value, 0.0)



def pso_with_fitness(fitness_values, particles_position, inertia_weight= 0.729844, cognitive_coeff=2.05, social_coeff=2.05):
    min_bounds = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1])  # Define minimum bounds for each dimension
    max_bounds = np.array([0.9, 32.0, 3.0, 32.0, 3.0, 64.0, 32.0, 0.01, 128.0])  # Define maximum bounds for each dimension
    #Vmax_bounds = np.array([0.9, 400, 400, 400, 400, 400, 400, 400, 400])

    Vmin_bounds = []
    Vmax_bounds = []

   # Vmax_bounds = np.array([0.9, 32.0, 3.0, 32.0, 3.0, 64.0, 32.0, 0.01, 128.0])
    for element in Vmin_bounds:
        Vmin_bounds.append(abs(max_bounds[element] - min_bounds[element]) * random.random(0.001,0.02))
        Vmax_bounds.append(abs(max_bounds[element] - min_bounds[element]) * random.random(0.8,1.0)/2)

    num_particles, num_dimensions = particles_position.shape

    particles_velocity = []

    for i in range(num_particles):
        for j in range(num_dimensions):
            particles_velocity[i]

    # Initialize personal best
    personal_best_position = particles_position.copy()
    personal_best_value = fitness_values.copy()

    # Initialize global best
    global_best_index = np.argmin(personal_best_value)
    global_best_position = personal_best_position[global_best_index]
    global_best_value = personal_best_value[global_best_index]

    # Main PSO loop
    for i in range(num_particles):


        # Update personal best
        if fitness_values[i] < personal_best_value[i]:
            personal_best_value[i] = fitness_values[i]
            personal_best_position[i] = particles_position[i]

        # Update global best
        if fitness_values[i] < global_best_value:
            global_best_value = fitness_values[i]
            global_best_position = particles_position[i]

        # Update particle position
        random_number = random.random()
        inertia_term = 0.5 + random_number / 2 

        # Calculate new positio
        new_position = particles_position[i] + inertia_term + cognitive_term + social_term

        # Apply bounds constraints for each dimension separately
        for dim in range(num_dimensions):
            new_position[dim] = np.clip(new_position[dim], min_bounds[dim], max_bounds[dim])

        particles_position[i] = new_position

    return global_best_position, global_best_value,particles_position,cognitive_term,social_term #,inertia_term
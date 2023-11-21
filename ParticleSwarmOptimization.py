import numpy as np

def pso_with_fitness(fitness_values, particles_position, inertia_weight, cognitive_coeff, social_coeff):
    min_bounds = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])  # Define minimum bounds for each dimension
    max_bounds = np.array([0.9, 32, 3, 32, 3, 64, 32, 0.01, 128])  # Define maximum bounds for each dimension


    num_particles, num_dimensions = particles_position.shape

    # Initialize personal best
    personal_best_position = particles_position.copy()
    personal_best_value = fitness_values.copy()#np.inf * np.ones(num_particles)

    # Initialize global best
    global_best_index = np.argmin(fitness_values)
    global_best_position = particles_position[global_best_index]
    global_best_value = fitness_values[global_best_index]

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
        inertia_term = inertia_weight * np.random.rand() * (personal_best_position[i] - particles_position[i])
        cognitive_term = cognitive_coeff #* np.random.rand() * (personal_best_position[i] - particles_position[i])
        social_term = social_coeff #* np.random.rand() * (global_best_position - particles_position[i])

        # Calculate new positio
        new_position = particles_position[i] + inertia_term + cognitive_term + social_term

        # Apply bounds constraints for each dimension separately
        for dim in range(num_dimensions):
            new_position[dim] = np.clip(new_position[dim], min_bounds[dim], max_bounds[dim])

        particles_position[i] = new_position

    return global_best_position, global_best_value,particles_position,inertia_term
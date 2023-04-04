import numpy as np

def fitness_function(individual):
    x = np.linspace(0, 2*np.pi, 100)
    y = individual[0] * np.sin(individual[1]*x + individual[2]) + individual[3]
    error = np.mean(np.abs(y - np.sin(x)))
    fitness = 1 / (1 + error)
    return fitness

def genetic_algorithm(population_size, num_generations, mutation_rate):
    population = np.random.rand(population_size, 4) * 2 - 1

    for generation in range(num_generations):
        fitness_scores = np.array([fitness_function(individual) for individual in population])

        parent_indices = np.random.choice(population_size, size=population_size, p=fitness_scores/fitness_scores.sum(), replace=True)
        parents = population[parent_indices]

        offspring = np.zeros_like(parents)
        for i in range(population_size):
            parent1 = parents[i]
            parent2 = parents[np.random.randint(0, population_size)]
            crossover_point = np.random.randint(0, 4)
            offspring[i, :crossover_point] = parent1[:crossover_point]
            offspring[i, crossover_point:] = parent2[crossover_point:]

        mask = np.random.rand(population_size, 4) < mutation_rate
        offspring += mask * np.random.rand(population_size, 4) * 2 - 1

        population = offspring

        print(f"Generation {generation+1}: Best fitness = {np.max(fitness_scores)}")

    best_individual_index = np.argmax([fitness_function(individual) for individual in population])
    best_individual = population[best_individual_index]
    return best_individual

best_individual = genetic_algorithm(population_size=100, num_generations=100, mutation_rate=0.1)

print("Best individual:", best_individual)

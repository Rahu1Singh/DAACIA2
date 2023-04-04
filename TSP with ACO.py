import numpy as np

# Define the distance matrix
distances = np.array([[ 0, 10, 50, 30, 70, 10, 20, 40],
                      [10,  0, 40, 60, 50, 50, 20, 30],
                      [50, 40,  0, 70, 20, 10, 40, 60],
                      [30, 60, 70,  0, 60, 60, 20, 40],
                      [70, 50, 20, 60,  0, 50, 20, 30],
                      [10, 50, 10, 60, 50,  0, 30, 60],
                      [20, 20, 40, 20, 20, 30,  0, 50],
                      [40, 30, 60, 40, 30, 60, 50,  0]])

# Define the number of ants, iterations, and pheromone evaporation rate
num_ants = 10
num_iterations = 15
evaporation_rate = 0.5

# Define the initial pheromone level
pheromone = np.ones_like(distances) / len(distances)

# Define the ACO function
def ant_colony_optimization(num_ants, num_iterations, evaporation_rate, pheromone, distances):
    best_path = None
    best_distance = np.inf

    # Iterate through iterations
    for iteration in range(num_iterations):
        # Initialize ant paths and distances
        ant_paths = np.zeros((num_ants, len(distances)), dtype=np.int32)
        ant_distances = np.zeros(num_ants)

        # Iterate through ants
        for ant in range(num_ants):
            # Choose starting city
            current_city = np.random.randint(len(distances))
            visited_cities = [current_city]

            # Choose next cities until all cities are visited
            for _ in range(len(distances)-1):
                # Calculate probabilities of choosing each unvisited city
                unvisited_cities = np.setdiff1d(np.arange(len(distances)), visited_cities)
                pheromone_values = pheromone[current_city, unvisited_cities]
                heuristic_values = 1 / distances[current_city, unvisited_cities]
                probabilities = pheromone_values * heuristic_values
                probabilities /= probabilities.sum()

                # Choose next city
                next_city = np.random.choice(unvisited_cities, p=probabilities)
                visited_cities.append(next_city)
                ant_paths[ant, len(visited_cities)-1] = next_city
                ant_distances[ant] += distances[current_city, next_city]
                current_city = next_city

            # Add distance back to starting city
            ant_distances[ant] += distances[current_city, ant_paths[ant, 0]]

        # Update pheromone levels
        pheromone *= evaporation_rate
        for ant in range(num_ants):
            for i in range(len(distances)-1):
                j = ant_paths[ant, i]
                k = ant_paths[ant, i+1]
                pheromone[j, k] += 1 / ant_distances[ant]

        # Update best path and distance
        if np.min(ant_distances) < best_distance:
            best_distance = np.min(ant_distances)
            best_path = ant_paths[np.argmin(ant_distances)]

        # Print best distance for each iteration
        print(f"Iteration {iteration+1}: Best distance = {best_distance}")

    # Return best path and distance
    return best_path, best_distance

# Run the ACO function
best_path, best_distance = ant_colony_optimization(num_ants, num_iterations, evaporation_rate, pheromone, distances)

# Print the best path and distance
print("Best path:", best_path)
print("Best distance:", best_distance)

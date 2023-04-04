import random
import math

# define the cities as a list of tuples (x, y) coordinates
cities = [(60, 200), (180, 200), (80, 180), (140, 180), (20, 160),
          (100, 160), (200, 160), (140, 140), (40, 120), (100, 120),
          (180, 100), (60, 80), (120, 80), (180, 60), (20, 40),
          (100, 40), (200, 40), (20, 20), (60, 20), (160, 20)]

# define the population size, number of generations, and mutation probability
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_PROBABILITY = 0.05

# define a function to calculate the distance between two cities
def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# define a function to calculate the fitness of a path (i.e., the total distance)
def fitness(path):
    total_distance = sum(distance(path[i], path[i+1]) for i in range(len(path)-1))
    total_distance += distance(path[-1], path[0]) # add the distance from the last city back to the first city
    return 1 / total_distance # fitness is inversely proportional to the total distance

# define a function to generate an initial population of paths
def generate_population():
    population = []
    for i in range(POPULATION_SIZE):
        path = cities.copy()
        random.shuffle(path)
        population.append(path)
    return population

# define a function to perform selection by tournament
def selection(population):
    tournament_size = 5
    selected = []
    for i in range(len(population)):
        competitors = random.sample(population, tournament_size)
        winner = max(competitors, key=fitness)
        selected.append(winner)
    return selected

# define a function to perform crossover by partially mapped crossover (PMX)
def crossover(parent1, parent2):
    child = parent1.copy()
    start, end = sorted(random.sample(range(len(parent1)), 2))
    for i in range(start, end+1):
        value = parent2[i]
        index = child.index(value)
        child[i], child[index] = child[index], child[i]
    return child

# define a function to perform mutation by swapping two cities
def mutation(path):
    if random.random() < MUTATION_PROBABILITY:
        i, j = random.sample(range(len(path)), 2)
        path[i], path[j] = path[j], path[i]
    return path

# define the main genetic algorithm function
def genetic_algorithm():
    population = generate_population()
    for generation in range(NUM_GENERATIONS):
        selected = selection(population)
        offspring = []
        for i in range(POPULATION_SIZE):
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            child = mutation(child)
            offspring.append(child)
        population = offspring
    best_path = max(population, key=fitness)
    return best_path

# run the genetic algorithm and print the best path
best_path = genetic_algorithm()
print(best_path)

import random

def create_individual(individual_size):
    return [random.uniform(-1, 1) for _ in range(individual_size)]

def generate_population(individual_size, population_size):
    return [create_individual(individual_size) for _ in range(population_size)]

def mutate(individual, mutation_strength=0.1):
    idx = random.randint(0, len(individual) - 1)
    individual[idx] += random.gauss(0, mutation_strength)
    individual[idx] = max(-1, min(1, individual[idx]))

def reproduce(parent1, parent2):
    crossover = random.randint(0, len(parent1) - 1)
    return parent1[:crossover] + parent2[crossover:]

def select_parent(population, fitnesses, tournament_size=2):
    indices = random.sample(range(len(population)), k=tournament_size)
    best_idx = max(indices, key=lambda idx: fitnesses[idx])
    return population[best_idx]

def genetic_algorithm(individual_size, population_size, fitness_function, target_fitness, generations, elite_rate=0.2, mutation_rate=0.05):
    population = generate_population(individual_size, population_size)
    best_individual = None
    best_fitness = float('-inf')

    for generation in range(generations):
        # Calcula fitness só uma vez por indivíduo
        fitnesses = [fitness_function(ind) for ind in population]

        # Elitismo
        num_elites = int(len(population) * elite_rate)
        elite_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)[:num_elites]
        elites = [population[i] for i in elite_indices]

        # Novo best
        gen_best_idx = elite_indices[0]
        if fitnesses[gen_best_idx] > best_fitness:
            best_individual = population[gen_best_idx]
            best_fitness = fitnesses[gen_best_idx]

        if best_fitness >= target_fitness:
            break

        new_population = elites.copy()
        while len(new_population) < len(population):
            parent1 = select_parent(population, fitnesses)
            parent2 = select_parent(population, fitnesses)
            child = reproduce(parent1, parent2)
            if random.random() < mutation_rate:
                mutate(child)
            new_population.append(child)

        population = new_population
        print(f'Generation {generation} | Best fitness: {best_fitness}')

    return best_individual, best_fitness
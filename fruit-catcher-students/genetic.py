import random

def create_individual(individual_size):
    return [random.uniform(-1, 1) for _ in range(individual_size)]

def generate_population(individual_size, population_size):
    return [create_individual(individual_size) for _ in range(population_size)]

#Classes copiadas do notebook deve ser necessário alterar

def print_individual(individual):
    for r in reversed(range(len(individual))):
        print(' '.join('Q' if c == r else '.' for c in individual))

def fitness(individual): #Ta no contexto das n queens problem !!!!!!!!!!!!! ALTERAR
    n = len(individual)
    f = n * (n - 1) // 2
    for i in range(n):
        for j in range(i+1, n):
            if individual[i] == individual[j] or abs(individual[i] - individual[j]) == abs(i - j):
                f -= 1
    return f

def select_parent(population, tournament_size=2):
    tournament = random.sample(population, k=tournament_size)
    return max(tournament, key=fitness)

def reproduce(parent1, parent2):
    crossover = random.randint(0, len(parent1) - 1)
    return parent1[:crossover] + parent2[crossover:]

def mutate(individual):
    individual[random.randint(0, len(individual) - 1)] = random.randint(0, len(individual) - 1)

def genetic_algorithm(individual_size, population_size, fitness_function, target_fitness, generations, elite_rate=0.2, mutation_rate=0.05):
    population = generate_population(individual_size, population_size)
    best_individual = None
    
    for generation in range(generations):
        if fitness(best_individual) >= target_fitness:
            break

        elites = sorted(population, key=fitness, reverse=True)[ : int(len(population)*elite_rate) ]
        new_population = elites
        while len(new_population) < len(population):
            parent1 = select_parent(population)
            parent2 = select_parent(population)
            child = reproduce(parent1, parent2)

            if random.random() < mutation_rate:
                mutate(child)
            new_population.append(child)

        population = new_population
        best_individual = max(population, key=fitness)
        print(f'Generation {generation}')
        print_individual(best_individual)
        print(f'Fitness: {fitness(best_individual)}')

    return best_individual # This is expected to be a pair (individual, fitness)
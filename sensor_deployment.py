############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Strength Pareto Evolutionary Algorithm 2

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-SPEA-2, File: Python-MH-SPEA-2.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-SPEA-2>

############################################################################

# Required Libraries
import numpy  as np
import math
import matplotlib.pyplot as plt
import random
import os

# Function 1
def func_1():
    return

# Function 2
def func_2():
    return

# Function: Initialize Variables
def initial_population(population_size = 5, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2]):
    population = np.zeros((population_size, len(min_values) + len(list_of_functions)))
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
             population[i,j] = random.uniform(min_values[j], max_values[j])      
        for k in range (1, len(list_of_functions) + 1):
            population[i,-k] = list_of_functions[-k](list(population[i,0:population.shape[1]-len(list_of_functions)]))
    return population
    
# Function: Dominance
def dominance_function(solution_1, solution_2, number_of_functions = 2):
    count = 0
    dominance = True
    for k in range (1, number_of_functions + 1):
        if (solution_1[-k] <= solution_2[-k]):
            count = count + 1
    if (count == number_of_functions):
        dominance = True
    else:
        dominance = False       
    return dominance

# Function: Non-Dominated Solutions
def find_non_dominated_solutions(population, number_of_functions=2):
    non_dominated_indices = []
    
    for i in range(population.shape[0]):
        is_dominated = False
        for j in range(population.shape[0]):
            if i != j:
                # Check if solution j dominates solution i
                if dominance_function(population[j,:], population[i,:], number_of_functions):
                    is_dominated = True
                    break
        
        if not is_dominated:
            non_dominated_indices.append(i)
    
    return non_dominated_indices

#Truncate archive to desired size while preserving diversity
def truncation_operator(archive, archive_size, number_of_functions=2):
    if archive.shape[0] <= archive_size:
        return archive
    
    current_archive = np.copy(archive)
    
    while current_archive.shape[0] > archive_size:
        distances = np.zeros((current_archive.shape[0], current_archive.shape[0]))
        
        # Calculate distances between all pairs in objective space
        for i in range(current_archive.shape[0]):
            for j in range(current_archive.shape[0]):
                if i != j:
                    obj_i = current_archive[i, -number_of_functions:]
                    obj_j = current_archive[j, -number_of_functions:]
                    distances[i,j] = euclidean_distance(obj_i, obj_j)
                else:
                    distances[i,j] = float('inf')  # Distance to self is infinite
        
        # Find minimum distances for each solution
        min_distances = np.min(distances, axis=1)
        
        # Remove solution with smallest minimum distance
        remove_idx = np.argmin(min_distances)
        current_archive = np.delete(current_archive, remove_idx, axis=0)
    
    return current_archive

# Fill archive with non-dominated solutions and best dominated ones if needed
def fill_archive(population, archive_size, number_of_functions=2):
    # Find non-dominated solutions
    non_dominated_indices = find_non_dominated_solutions(population, number_of_functions)
    
    if len(non_dominated_indices) == 0:
        # If no non-dominated solutions, return best solutions by fitness
        raw_fitness = raw_fitness_function(population, number_of_functions)
        fitness = fitness_calculation(population, raw_fitness, number_of_functions)
        sorted_pop, _ = sort_population_by_fitness(population, fitness)
        return sorted_pop[:archive_size]
    
    # Get non-dominated solutions
    non_dominated = population[non_dominated_indices]
    
    if non_dominated.shape[0] >= archive_size:
        # Too many non-dominated solutions, use truncation
        return truncation_operator(non_dominated, archive_size, number_of_functions)
    
    # Add dominated solutions to fill archive
    archive = np.copy(non_dominated)
    remaining_capacity = archive_size - archive.shape[0]
    
    if remaining_capacity > 0:
        # Get dominated solutions
        all_indices = set(range(population.shape[0]))
        dominated_indices = list(all_indices - set(non_dominated_indices))
        
        if len(dominated_indices) > 0:
            dominated_solutions = population[dominated_indices]
            
            # Sort dominated solutions by fitness
            dom_raw_fitness = raw_fitness_function(dominated_solutions, number_of_functions)
            dom_fitness = fitness_calculation(dominated_solutions, dom_raw_fitness, number_of_functions)
            sorted_dominated, _ = sort_population_by_fitness(dominated_solutions, dom_fitness)
            
            # Add best dominated solutions
            num_to_add = min(remaining_capacity, sorted_dominated.shape[0])
            archive = np.vstack([archive, sorted_dominated[:num_to_add]])
    
    return archive

# Improved breeding function using binary tournament selection
def breeding_improved(population, fitness, min_values=[-5,-5], max_values=[5,5], mu=1, list_of_functions=[func_1, func_2]):
    offspring = np.copy(population)
    b_offspring = 0
    
    for i in range(0, offspring.shape[0]):
        # Use binary tournament instead of roulette wheel
        parent_1 = binary_tournament_selection(population, fitness)
        parent_2 = binary_tournament_selection(population, fitness)
        
        # Ensure different parents
        while parent_1 == parent_2:
            parent_2 = binary_tournament_selection(population, fitness)
        
        # Rest of crossover remains the same
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            rand = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            rand_b = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))
            
            offspring[i,j] = np.clip(((1 + b_offspring)*population[parent_1, j] + (1 - b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j])
            
            if(i < population.shape[0] - 1):
                offspring[i+1,j] = np.clip(((1 - b_offspring)*population[parent_1, j] + (1 + b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j])
        
        # Evaluate offspring
        for k in range(1, len(list_of_functions) + 1):
            offspring[i,-k] = list_of_functions[-k](offspring[i,0:offspring.shape[1]-len(list_of_functions)])
    
    return offspring

# Function: Raw Fitness
def raw_fitness_function(population, number_of_functions = 2):    
    strength = np.zeros((population.shape[0], 1))
    raw_fitness = np.zeros((population.shape[0], 1))
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if(i != j):
                if dominance_function(solution_1 = population[i,:], solution_2 = population[j,:], number_of_functions = number_of_functions):
                    strength[i,0] = strength[i,0] + 1
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if(i != j):
                if dominance_function(solution_1 = population[i,:], solution_2 = population[j,:], number_of_functions = number_of_functions):
                    raw_fitness[j,0] = raw_fitness[j,0] + strength[i,0]
    return raw_fitness

# Function: Distance Calculations
def euclidean_distance(x, y):       
    distance = 0
    for j in range(0, len(x)):
        distance = (x[j] - y[j])**2 + distance   
    return distance**(1/2) 

# Function: Fitness
def fitness_calculation(population, raw_fitness, number_of_functions = 2):
    k = int(len(population)**(1/2)) - 1
    fitness  = np.zeros((population.shape[0], 1))
    distance = np.zeros((population.shape[0], population.shape[0]))
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if(i != j):
                x = np.copy(population[i, population.shape[1]-number_of_functions:])
                y = np.copy(population[j, population.shape[1]-number_of_functions:])
                distance[i,j] =  euclidean_distance(x = x, y = y)                    
    for i in range(0, fitness.shape[0]):
        distance_ordered = (distance[distance[:,i].argsort()]).T
        fitness[i,0] = raw_fitness[i,0] + 1/(distance_ordered[i,k] + 2)
    return fitness

# Function: Sort Population by Fitness
def sort_population_by_fitness(population, fitness):
    idx = np.argsort(fitness[:,-1])
    fitness_new = np.zeros((population.shape[0], 1))
    population_new = np.zeros((population.shape[0], population.shape[1]))
    for i in range(0, population.shape[0]):
        fitness_new[i,0] = fitness[idx[i],0] 
        for k in range(0, population.shape[1]):
            population_new[i,k] = population[idx[i],k]
    return population_new, fitness_new

# Function: Selection
# def roulette_wheel(fitness_new): 
#     fitness = np.zeros((fitness_new.shape[0], 2))
#     for i in range(0, fitness.shape[0]):
#         fitness[i,0] = 1/(1+ fitness[i,0] + abs(fitness[:,0].min()))
#     fit_sum = fitness[:,0].sum()
#     fitness[0,1] = fitness[0,0]
#     for i in range(1, fitness.shape[0]):
#         fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
#     for i in range(0, fitness.shape[0]):
#         fitness[i,1] = fitness[i,1]/fit_sum
#     ix = 0
#     random = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
#     for i in range(0, fitness.shape[0]):
#         if (random <= fitness[i, 1]):
#           ix = i
#           break
#     return ix

# Function: Binary Tournament Selection
def binary_tournament_selection(population_new, fitness_new):
    idx1 = random.randint(0, population_new.shape[0] - 1)
    idx2 = random.randint(0, population_new.shape[0] - 1)
    
    # Return index of individual with better (lower) fitness
    if fitness_new[idx1, 0] <= fitness_new[idx2, 0]:
        return idx1
    else:
        return idx2

# Function: Offspring
def breeding(population, fitness, min_values = [-5,-5], max_values = [5,5], mu = 1, list_of_functions = [func_1, func_2]):
    offspring = np.copy(population)
    b_offspring = 0
    for i in range (0, offspring.shape[0]):
        parent_1, parent_2 = binary_tournament_selection(population, fitness), binary_tournament_selection(population, fitness)
        # parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            rand_b = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):  
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))       
            offspring[i,j] = np.clip(((1 + b_offspring)*population[parent_1, j] + (1 - b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j])           
            if(i < population.shape[0] - 1):   
                offspring[i+1,j] = np.clip(((1 - b_offspring)*population[parent_1, j] + (1 + b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j]) 
        for k in range (1, len(list_of_functions) + 1):
            offspring[i,-k] = list_of_functions[-k](offspring[i,0:offspring.shape[1]-len(list_of_functions)])
    return offspring

# Function: Mutation
def mutation(offspring, mutation_rate = 0.1, eta = 1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2]):
    d_mutation = 0            
    for i in range (0, offspring.shape[0]):
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            probability = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (probability < mutation_rate):
                rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                rand_d = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                     
                if (rand <= 0.5):
                    d_mutation = 2*(rand_d)
                    d_mutation = d_mutation**(1/(eta + 1)) - 1
                elif (rand > 0.5):  
                    d_mutation = 2*(1 - rand_d)
                    d_mutation = 1 - d_mutation**(1/(eta + 1))                
                offspring[i,j] = np.clip((offspring[i,j] + d_mutation), min_values[j], max_values[j])                        
        for k in range (1, len(list_of_functions) + 1):
            offspring[i,-k] = list_of_functions[-k](offspring[i,0:offspring.shape[1]-len(list_of_functions)])
    return offspring 

# # SPEA-2 Function
# def strength_pareto_evolutionary_algorithm_2(population_size = 5, archive_size = 5, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2], generations = 50, mu = 1, eta = 1):        
#     count = 0   
#     population = initial_population(population_size = population_size, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions) 
#     archive = initial_population(population_size = archive_size, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)     
#     while (count <= generations):       
#         print("Generation = ", count)
#         population = np.vstack([population, archive])
#         raw_fitness   = raw_fitness_function(population, number_of_functions = len(list_of_functions))
#         fitness    = fitness_calculation(population, raw_fitness, number_of_functions = len(list_of_functions))        
#         population, fitness = sort_population_by_fitness(population, fitness)
#         population, archive, fitness = population[0:population_size,:], population[0:archive_size,:], fitness[0:archive_size,:]
#         population = breeding(population, fitness, mu = mu, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)
#         population = mutation(population, mutation_rate = mutation_rate, eta = eta, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)             
#         count = count + 1              
#     return archive

def strength_pareto_evolutionary_algorithm_2_improved(population_size=5, archive_size=5, mutation_rate=0.1, 
                                                    min_values=[-5,-5], max_values=[5,5], 
                                                    list_of_functions=[func_1, func_2], 
                                                    generations=50, mu=1, eta=1):
    """
    Improved SPEA-II implementation following the original algorithm more closely
    """
    count = 0
    population = initial_population(population_size=population_size, min_values=min_values, 
                                  max_values=max_values, list_of_functions=list_of_functions)
    archive = initial_population(population_size=archive_size, min_values=min_values, 
                               max_values=max_values, list_of_functions=list_of_functions)
    
    while (count <= generations):
        print("Generation = ", count)
        
        # Combine population and archive for fitness evaluation
        combined = np.vstack([population, archive])
        
        # Calculate fitness for combined population
        raw_fitness = raw_fitness_function(combined, number_of_functions=len(list_of_functions))
        fitness = fitness_calculation(combined, raw_fitness, number_of_functions=len(list_of_functions))
        
        # Update archive with proper SPEA-II archive management
        archive = fill_archive(combined, archive_size, number_of_functions=len(list_of_functions))
        
        # Calculate fitness for selection (only on current population + current archive)
        selection_pool = np.vstack([population, archive])
        sel_raw_fitness = raw_fitness_function(selection_pool, number_of_functions=len(list_of_functions))
        sel_fitness = fitness_calculation(selection_pool, sel_raw_fitness, number_of_functions=len(list_of_functions))
        
        # Generate new population through selection, crossover, and mutation
        population = breeding_improved(selection_pool, sel_fitness, mu=mu, min_values=min_values, 
                                     max_values=max_values, list_of_functions=list_of_functions)
        population = mutation(population, mutation_rate=mutation_rate, eta=eta, 
                            min_values=min_values, max_values=max_values, 
                            list_of_functions=list_of_functions)
        
        # Ensure population size
        if population.shape[0] > population_size:
            pop_raw_fitness = raw_fitness_function(population, number_of_functions=len(list_of_functions))
            pop_fitness = fitness_calculation(population, pop_raw_fitness, number_of_functions=len(list_of_functions))
            population, _ = sort_population_by_fitness(population, pop_fitness)
            population = population[:population_size]
        
        count = count + 1
    
    return archive
######################## Part 1 - Usage ####################################

# Schaffer Function 1
def schaffer_f1(variables_values = [0]):
    y = variables_values[0]**2
    return y

# Schaffer Function 2
def schaffer_f2(variables_values = [0]):
    y = (variables_values[0]-2)**2
    return y

# Calling SPEA-2 Function
spea_2_schaffer = strength_pareto_evolutionary_algorithm_2_improved(population_size = 50, archive_size = 50, mutation_rate = 0.1, min_values = [-1000], max_values = [1000], list_of_functions = [schaffer_f1, schaffer_f2], generations = 100, mu = 1, eta = 1)

# Shaffer Pareto Front
schaffer = np.zeros((200, 3))
x = np.arange(0.0, 2.0, 0.01)
for i in range (0, schaffer.shape[0]):
    schaffer[i,0] = x[i]
    schaffer[i,1] = schaffer_f1(variables_values = [schaffer[i,0]])
    schaffer[i,2] = schaffer_f2(variables_values = [schaffer[i,0]])

schaffer_1 = schaffer[:,1]
schaffer_2 = schaffer[:,2]

# Graph Pareto Front Solutions
func_1_values = spea_2_schaffer[:,-2]
func_2_values = spea_2_schaffer[:,-1]
ax1 = plt.figure(figsize = (15,15)).add_subplot(111)
plt.xlabel('Function 1', fontsize = 12)
plt.ylabel('Function 2', fontsize = 12)
ax1.scatter(func_1_values, func_2_values, c = 'red',   s = 25, marker = 'o', label = 'SPEA-2')
ax1.scatter(schaffer_1,    schaffer_2,    c = 'black', s = 2,  marker = 's', label = 'Pareto Front')
plt.legend(loc = 'upper right')
plt.show()

######################## Part 2 - Usage ####################################

# Kursawe Function 1
def kursawe_f1(variables_values = [0, 0]):
    f1 = 0
    if (len(variables_values) == 1):
        f1 = f1 - 10 * math.exp(-0.2 * math.sqrt(variables_values[0]**2 + variables_values[0]**2))
    else:
        for i in range(0, len(variables_values)-1):
            f1 = f1 - 10 * math.exp(-0.2 * math.sqrt(variables_values[i]**2 + variables_values[i + 1]**2))
    return f1

# Kursawe Function 2
def kursawe_f2(variables_values = [0, 0]):
    f2 = 0
    for i in range(0, len(variables_values)):
        f2 = f2 + abs(variables_values[i])**0.8 + 5 * math.sin(variables_values[i]**3)
    return f2

# Calling SPEA-2 Function
spea_2_kursawe = strength_pareto_evolutionary_algorithm_2_improved(population_size = 50, archive_size = 50, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [kursawe_f1, kursawe_f2], generations = 100, mu = 1, eta = 1)

# Kursawe Pareto Front
kursawe = np.zeros((10000, 4))
x = np.arange(-5, 5, 0.1)
count = 0
for j in range (0,100):
    for k in range (0, 100):
            kursawe[count,0] = x[j]
            kursawe[count,1] = x[k]
            count = count + 1
        
for i in range (0, kursawe.shape[0]):
    kursawe[i,2] = kursawe_f1(variables_values = [kursawe[i,0], kursawe[i,1]])
    kursawe[i,3] = kursawe_f2(variables_values = [kursawe[i,0], kursawe[i,1]])

kursawe_1 = kursawe[:,2]
kursawe_2 = kursawe[:,3]

# Graph Pareto Front Solutions
func_1_values = spea_2_kursawe[:,-2]
func_2_values = spea_2_kursawe[:,-1]
ax1 = plt.figure(figsize = (15,15)).add_subplot(111)
plt.xlabel('Function 1', fontsize = 12)
plt.ylabel('Function 2', fontsize = 12)
ax1.scatter(func_1_values, func_2_values, c = 'red',   s = 25, marker = 'o', label = 'SPEA-2')
ax1.scatter(kursawe_1,     kursawe_2,     c = 'black', s = 2,  marker = 's', label = 'Solutions')
plt.legend(loc = 'upper right')
plt.show()
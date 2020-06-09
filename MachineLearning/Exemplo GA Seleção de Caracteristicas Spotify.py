# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:50:29 2020

@author: felip
"""


# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:59:53 2019

@author: felip
"""

import pickle
import matplotlib.pyplot
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features


def classification_accuracy(labels, predictions):
    correct = numpy.where(labels == predictions)[0]
    accuracy = correct.shape[0]/labels.shape[0]
    return accuracy


def cal_pop_fitness(pop, features, labels, train_indices, test_indices):
    accuracies = np.zeros(pop.shape[0])
    idx = 0

    for curr_solution in pop:
        reduced_features = reduce_features(curr_solution, features)
        train_data = reduced_features[train_indices, :]
        test_data = reduced_features[test_indices, :]

        train_labels = labels[train_indices]
        test_labels = labels[test_indices]

        #svm = SVC(kernel='rbf',C=1,gamma=0.01)
        #svm.fit(X=train_data, y=train_labels)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X=train_data, y=train_labels)       
        
        predictions = knn.predict(test_data)
        #predictions = svm.predict(test_data)
        
        
        acc = accuracy_score(test_labels, predictions)
        
        accuracies[idx] = acc
        idx = idx + 1
    return accuracies

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents


def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover, num_mutations=2):
    mutation_idx = np.random.randint(low=0, high=offspring_crossover.shape[1], size=num_mutations)
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        offspring_crossover[idx, mutation_idx] = 1 - offspring_crossover[idx, mutation_idx]
    return offspring_crossover


def remove_features(lista_features):
    for i in lista_features:
        dataset.drop(i, axis=1, inplace=True)
    return 0

# *********************************** 
dataset = pd.read_csv('Datasets\DadosSpotify.csv', sep=',', engine='python')

# Remove features
remove_features(['id','song_title'])

# # Separa a classe dos dados
classes = dataset['target']
dataset.drop('target', axis=1, inplace=True)

# # Label Encoder
enc = LabelEncoder()
inteiros = enc.fit_transform(dataset['artist'])

# Cria uma nova coluna chamada 'artist_inteiros'
dataset['artist_inteiros'] = inteiros
remove_features(['artist'])

# Instancia um objeto do tipo OnehotEncoder
ohe = OneHotEncoder()
# Transforma em arrayn numpy o dataset.
dataset_array = dataset.values
# Pega o numero de linhas.
num_rows = dataset_array.shape[0]

# Transforma a matriz em uma dimensão
inteiros = inteiros.reshape(len(inteiros),1)
# Criar as novas features a partir da matriz de presença
novas_features = ohe.fit_transform(inteiros)

aux = novas_features.toarray()
# Concatena as novas features ao array
dataset_array = np.concatenate([dataset_array, novas_features.toarray()], axis=1)

# Transforma em dataframe e visualiza as colunas
dataset_features = pd.DataFrame(dataset_array)



scaler = StandardScaler()
scaler.fit(dataset_features)
data_inputs = scaler.transform(dataset_features)
#data_inputs = dataset_features
data_outputs = classes

num_samples = data_inputs.shape[0]
num_feature_elements = data_inputs.shape[1]

train_indices = np.arange(1, num_samples, 3)
test_indices = np.arange(0, num_samples, 3)
print("Number of training samples: ", train_indices.shape[0])
print("Number of test samples: ", test_indices.shape[0])

"""
Genetic algorithm parameters:
    Population size
    Mating pool size
    Number of mutations
"""
sol_per_pop = 30 # Population size.
num_parents_mating = 15 # Number of parents inside the mating pool.
num_mutations = 1 # Number of elements to mutate.

# Defining the population shape.
pop_shape = (sol_per_pop, num_feature_elements)

# Creating the initial population.
new_population = np.random.randint(low=0, high=2, size=pop_shape)
print(new_population.shape)

best_outputs = []
num_generations = 20
for generation in range(num_generations):
    print("Generation : ", generation)
    # Measuring the fitness of each chromosome in the population.
    fitness = cal_pop_fitness(new_population, data_inputs, data_outputs, train_indices, test_indices)
    
    best_match_idx = np.where(fitness == np.max(fitness))[0]
    best_match_idx = best_match_idx[0]
    best_solution = new_population[best_match_idx, :]

    best_outputs.append(np.max(fitness))
    # The best result in the current iteration.
    print("Best result : ", best_outputs[-1])
    print("Qtd Features : ",  np.sum(best_solution))

    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(new_population, fitness, num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = crossover(parents, offspring_size=(pop_shape[0]-parents.shape[0], num_feature_elements))

    # Adding some variations to the offspring using mutation.
    offspring_mutation = mutation(offspring_crossover, num_mutations=num_mutations)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

# Getting the best solution after iterating finishing all generations.
# At first, the fitness is calculated for each solution in the final generation.
fitness = cal_pop_fitness(new_population, data_inputs, data_outputs, train_indices, test_indices)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.max(fitness))[0]
best_match_idx = best_match_idx[0]

best_solution = new_population[best_match_idx, :]
best_solution_indices = np.where(best_solution == 1)[0]
best_solution_num_elements = best_solution_indices.shape[0]
best_solution_fitness = fitness[best_match_idx]

print("best_match_idx : ", best_match_idx)
print("best_solution : ", best_solution)
print("Selected indices : ", best_solution_indices)
print("Number of selected elements : ", best_solution_num_elements)
print("Best solution fitness : ", best_solution_fitness)

matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()




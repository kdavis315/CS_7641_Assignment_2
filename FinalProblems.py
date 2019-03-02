import numpy as np
import pandas as pd
import mlrose
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from time import clock
import mlrose as ml
from itertools import combinations
from sklearn.metrics import accuracy_score
import itertools

#####
#Count Ones problem
#####
np.random.seed(3)
fitness = mlrose.OneMax()

random_start = np.random.rand(100)
starting_state = np.where(random_start>0.5, 1, 0)

#Random Hill Climbing
problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val = 2)

start_time = clock()
best_state, best_fitness = mlrose.random_hill_climb(problem, max_attempts = 30, max_iters = 500, \
    init_state = starting_state)
training_time = clock()-start_time

print("")
print("Count Ones problem - Random Hill Climbing")
print("time")
print(training_time)
print("Best State")
print(best_state)
print("Best Fitness")
print(best_fitness)


#Simulated Annealing
problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val = 2)

start_time = clock()
best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=mlrose.ExpDecay(), max_attempts = 40,\
 max_iters = 500, init_state = starting_state)
training_time = clock()-start_time

print("")
print("Count Ones problem - Simulated Annealing")
print("time")
print(training_time)
print("Best State")
print(best_state)
print("Best Fitness")
print(best_fitness)


#Genetic Algorithm
problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val = 2)

start_time = clock()
best_state, best_fitness = mlrose.genetic_alg(problem, pop_size=300, mutation_prob=0.1, \
	max_attempts = 30, max_iters = 500)
training_time = clock()-start_time

print("")
print("Count Ones problem - Genetic Algorithm")
print("time")
print(training_time)
print("Best State")
print(best_state)
print("Best Fitness")
print(best_fitness)


#MIMIC
problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val = 2)

start_time = clock()
best_state, best_fitness = mlrose.mimic(problem, pop_size=200, keep_pct=0.3,\
 max_attempts = 10, max_iters = 500)
training_time = clock()-start_time

print("")
print("Count Ones problem - MIMIC")
print("time")
print(training_time)
print("Best State")
print(best_state)
print("Best Fitness")
print(best_fitness)





#####
#Four Peaks problem
#####
np.random.seed(3)
fitness = mlrose.FourPeaks()

random_start = np.random.rand(100)
starting_state = np.where(random_start>0.5, 1, 0)

#Random Hill Climbing
problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val = 2)

start_time = clock()
best_state, best_fitness = mlrose.random_hill_climb(problem, max_attempts = 20, max_iters = 500, \
    init_state = starting_state)
training_time = clock()-start_time

print("")
print("Four Peaks problem - Random Hill Climbing")
print("time")
print(training_time)
print("Best State")
print(best_state)
print("Best Fitness")
print(best_fitness)


#Simulated Annealing
problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val = 2)

start_time = clock()
best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=mlrose.ExpDecay(), max_attempts = 20,\
 max_iters = 1500, init_state = starting_state)
training_time = clock()-start_time

print("")
print("Four Peaks problem - Simulated Annealing")
print("time")
print(training_time)
print("Best State")
print(best_state)
print("Best Fitness")
print(best_fitness)


#Genetic Algorithm
problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val = 2)

start_time = clock()
best_state, best_fitness = mlrose.genetic_alg(problem, pop_size=300, mutation_prob=0.1, \
	max_attempts = 50, max_iters = 500)
training_time = clock()-start_time

print("")
print("Four Peaks problem - Genetic Algorithm")
print("time")
print(training_time)
print("Best State")
print(best_state)
print("Best Fitness")
print(best_fitness)


#MIMIC
problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val = 2)

start_time = clock()
best_state, best_fitness = mlrose.mimic(problem, pop_size=300, keep_pct=0.2,\
 max_attempts = 10, max_iters = 500)
training_time = clock()-start_time

print("")
print("Four Peaks problem - MIMIC")
print("time")
print(training_time)
print("Best State")
print(best_state)
print("Best Fitness")
print(best_fitness)





#####
#Knapsack problem
#####
np.random.seed(3)
r1 = np.random.rand(100)
weights = np.ceil(r1*10)

r2 = np.random.rand(100)
vals = np.ceil(r2*20)

print("")
print("Knapsack Problem")
print("weights")
print(weights)
print("values")
print(vals)

random_start = np.random.rand(100)
starting_state = np.where(random_start>0.5, 1, 0)

fitness = mlrose.Knapsack(weights, vals, 0.35)


#Random Hill Climbing
problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val = 2)

start_time = clock()
best_state, best_fitness = mlrose.random_hill_climb(problem, max_attempts = 20, max_iters = 500, \
    init_state = starting_state)
training_time = clock()-start_time

print("")
print("Knapsack problem - Random Hill Climbing")
print("time")
print(training_time)
print("Best State")
print(best_state)
print("Best Fitness")
print(best_fitness)


#Simulated Annealing
problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val = 2)

start_time = clock()
best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=mlrose.ExpDecay(), max_attempts = 20,\
 max_iters = 500, init_state = starting_state)
training_time = clock()-start_time

print("")
print("Knapsack problem - Simulated Annealing")
print("time")
print(training_time)
print("Best State")
print(best_state)
print("Best Fitness")
print(best_fitness)


#Genetic Algorithm
problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val = 2)

start_time = clock()
best_state, best_fitness = mlrose.genetic_alg(problem, pop_size=100, mutation_prob=0.1, \
	max_attempts = 20, max_iters = 500)
training_time = clock()-start_time

print("")
print("Knapsack problem - Genetic Algorithm")
print("time")
print(training_time)
print("Best State")
print(best_state)
print("Best Fitness")
print(best_fitness)


#MIMIC
problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val = 2)

start_time = clock()
best_state, best_fitness = mlrose.mimic(problem, pop_size=300, keep_pct=0.2,\
 max_attempts = 10, max_iters = 500)
training_time = clock()-start_time

print("")
print("Knapsack problem - MIMIC")
print("time")
print(training_time)
print("Best State")
print(best_state)
print("Best Fitness")
print(best_fitness)
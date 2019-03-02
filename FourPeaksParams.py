#kdavis315

#references:
# Hayes, G. (2019). mlrose: Machine Learning, Randomized Optimization and SEarch package for Python. https://github.com/gkhayes/mlrose. Accessed: 24 February 2019
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

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

#Four Peaks problem

#Random Hill Climbing
np.random.seed(3)

rhc_fp = []
n=2
while n in range (2, 11):
    N = n*10

    paramdict = {'max_attempts': [20,30,40,50],'max_iters':[500,1000,1500]}

    rhc_tr = []

    # get all the different combinations of parameters
    keys, values = zip(*paramdict.items())
    params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    i=0
    while i in range (0, len(params)):
                    times = params[i]['max_iters']
                    attempt = params[i]['max_attempts']

                    #problem

                    np.random.seed(3)
                    fitness = mlrose.FourPeaks()

                    random_start = np.random.rand(N)
                    starting_state = np.where(random_start>0.5, 1, 0)
                    problem = mlrose.DiscreteOpt(length = N, fitness_fn = fitness, maximize = True, max_val = 2)

                    best_state, best_fitness = mlrose.random_hill_climb(problem, max_attempts = attempt, max_iters = times, \
                        init_state = starting_state)


                    results = (i, best_state, best_fitness, N, times, attempt)
                    rhc_tr.append(results)
                   
                    # iterate  
                    i+=1

    #save the results and export
    regTable = pd.DataFrame(rhc_tr)
    # rename the columns
    regTable.columns = ['Run_no', 'best_state', 'best_fitness', 'bits', 'iterations', 'attempts']
    #regTable.to_csv('./output/{}_{}_reg.csv'.format("RHC_ANN","wine"),index=False)


    # find parameters that max test accuracy
    #bestfit = regTable['best_fitness'].max()
    bestfit = regTable.loc[regTable['best_fitness'].idxmax()]
    rhc_fp.append(bestfit)


    n+=1
#print(rhc_fp)
rhc_fp_table = pd.DataFrame(rhc_fp)
rhc_fp_table.columns = ['Run_no', 'best_state', 'best_fitness', 'bits', 'iterations', 'attempts']
no_bits = rhc_fp_table['bits']
b_fitness = rhc_fp_table['best_fitness']
iters = rhc_fp_table['iterations']
attempts = rhc_fp_table['attempts']

def plot_metrics(X, y, title, ylab):

    plt.figure()
    plt.title(title)
    plt.xlabel("Bits")
    plt.ylabel(ylab)
    plt.grid()
    plt.plot(X, y, 'o-', color="r")
    return plt

title = "Fitness per problem size (RHC)"
ylab = "Fitness"
plot_metrics(no_bits, b_fitness, title, ylab)
plt.show()

title = "Iterations per problem size (RHC)"
ylab = "Iterations"
plot_metrics(no_bits, iters, title, ylab)
plt.show()

title = "Attempts per problem size (RHC)"
ylab = "Attempts"
plot_metrics(no_bits, attempts, title, ylab)
plt.show()



#Simulated Annealing
np.random.seed(3)

sa_fp = []
n=2
while n in range (2, 11):
    N = n*10

    paramdict = {'max_attempts': [20,30,40,50],'max_iters':[500,1000,1500], 'sched': ['ExpDecay', 'GeomDecay', 'ArithDecay'] }

    sa_tr = []

    # get all the different combinations of parameters
    keys, values = zip(*paramdict.items())
    params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    i=0
    while i in range (0, len(params)):
                    times = params[i]['max_iters']
                    attempt = params[i]['max_attempts']
                    schedule = params[i]['sched']

                    if schedule == 'ExpDecay':
                        sch = mlrose.ExpDecay()
                    elif schedule == 'GeomDecay':
                        sch = mlrose.GeomDecay()
                    else:
                        sch = mlrose.ArithDecay()

                    #problem

                    np.random.seed(3)
                    fitness = mlrose.FourPeaks()

                    random_start = np.random.rand(N)
                    starting_state = np.where(random_start>0.5, 1, 0)
                    problem = mlrose.DiscreteOpt(length = N, fitness_fn = fitness, maximize = True, max_val = 2)

                    best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=sch, \
                        max_attempts = attempt, max_iters = times, init_state = starting_state)


                    results = (i, best_state, best_fitness, N, times, attempt, schedule)
                    sa_tr.append(results)
                   
                    # iterate  
                    i+=1

    #save the results and export
    regTable = pd.DataFrame(sa_tr)
    # rename the columns
    regTable.columns = ['Run_no', 'best_state', 'best_fitness', 'bits', 'iterations', 'attempts', 'schedule']
    #regTable.to_csv('./output/{}_{}_reg.csv'.format("sa_ANN","wine"),index=False)


    # find parameters that max test accuracy
    #bestfit = regTable['best_fitness'].max()
    bestfit = regTable.loc[regTable['best_fitness'].idxmax()]
    sa_fp.append(bestfit)


    n+=1
#print(sa_fp)
sa_fp_table = pd.DataFrame(sa_fp)
sa_fp_table.columns = ['Run_no', 'best_state', 'best_fitness', 'bits', 'iterations', 'attempts', 'schedule']
no_bits = sa_fp_table['bits']
b_fitness = sa_fp_table['best_fitness']
iters = sa_fp_table['iterations']
attempts = sa_fp_table['attempts']
schedule = sa_fp_table['schedule']

def plot_metrics(X, y, title, ylab):

    plt.figure()
    plt.title(title)
    plt.xlabel("Bits")
    plt.ylabel(ylab)
    plt.grid()
    plt.plot(X, y, 'o-', color="r")
    return plt

title = "Fitness per problem size (SA)"
ylab = "Fitness"
plot_metrics(no_bits, b_fitness, title, ylab)
plt.show()

title = "Iterations per problem size (SA)"
ylab = "Iterations"
plot_metrics(no_bits, iters, title, ylab)
plt.show()

title = "Attempts per problem size (SA)"
ylab = "Attempts"
plot_metrics(no_bits, attempts, title, ylab)
plt.show()

title = "Schedule per problem size (SA)"
ylab = "Schedule"
plot_metrics(no_bits, schedule, title, ylab)
plt.show()



#Genetic Algorithm
np.random.seed(3)

ga_fp = []
n=2
while n in range (2, 11):
    N = n*10

    paramdict = {'max_attempts': [20,30,40,50],'max_iters':[500],'pop_size':[100,200,300],'m_prob':[0.1,0.2,0.3, 0.4,0.5] }

    ga_tr = []

    # get all the different combinations of parameters
    keys, values = zip(*paramdict.items())
    params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    i=0
    while i in range (0, len(params)):
                    times = params[i]['max_iters']
                    attempt = params[i]['max_attempts']
                    pop = params[i]['pop_size']
                    mprob = params[i]['m_prob']


                    #problem

                    np.random.seed(3)
                    fitness = mlrose.FourPeaks()

                    problem = mlrose.DiscreteOpt(length = N, fitness_fn = fitness, maximize = True, max_val = 2)

                    best_state, best_fitness = mlrose.genetic_alg(problem, pop_size=pop, mutation_prob = mprob, \
                        max_attempts = attempt, max_iters = times)


                    results = (i, best_state, best_fitness, N, times, attempt, pop, mprob)
                    ga_tr.append(results)
                   
                    # iterate  
                    i+=1

    #save the results and export
    regTable = pd.DataFrame(ga_tr)
    # rename the columns
    regTable.columns = ['Run_no', 'best_state', 'best_fitness', 'bits', 'iterations', 'attempts', 'pop_size', 'mprob']
    #regTable.to_csv('./output/{}_{}_reg.csv'.format("sa_ANN","wine"),index=False)


    # find parameters that max test accuracy
    #bestfit = regTable['best_fitness'].max()
    bestfit = regTable.loc[regTable['best_fitness'].idxmax()]
    ga_fp.append(bestfit)


    n+=1
#print(ga_fp)
ga_fp_table = pd.DataFrame(ga_fp)
ga_fp_table.columns = ['Run_no', 'best_state', 'best_fitness', 'bits', 'iterations', 'attempts', 'pop_size', 'mprob']
no_bits = ga_fp_table['bits']
b_fitness = ga_fp_table['best_fitness']
iters = ga_fp_table['iterations']
attempts = ga_fp_table['attempts']
popsize = ga_fp_table['pop_size']
mutprob = ga_fp_table['mprob']

def plot_metrics(X, y, title, ylab):

    plt.figure()
    plt.title(title)
    plt.xlabel("Bits")
    plt.ylabel(ylab)
    plt.grid()
    plt.plot(X, y, 'o-', color="r")
    return plt

title = "Fitness per problem size (GA)"
ylab = "Fitness"
plot_metrics(no_bits, b_fitness, title, ylab)
plt.show()

title = "Iterations per problem size (GA)"
ylab = "Iterations"
plot_metrics(no_bits, iters, title, ylab)
plt.show()

title = "Attempts per problem size (GA)"
ylab = "Attempts"
plot_metrics(no_bits, attempts, title, ylab)
plt.show()

title = "Pop Size per problem size (GA)"
ylab = "Pop Size"
plot_metrics(no_bits, popsize, title, ylab)
plt.show()

title = "Mutation Prob per problem size (GA)"
ylab = "Mutation Prob"
plot_metrics(no_bits, mutprob, title, ylab)
plt.show()



#MIMIC
np.random.seed(3)

mim_fp = []
n=2
while n in range (2, 11):
    N = n*10

    paramdict = {'max_attempts': [10,20,30],'max_iters':[500],'pop_size':[100,200,300],'kpct':[0.2,0.3, 0.4] }

    mim_tr = []

    # get all the different combinations of parameters
    keys, values = zip(*paramdict.items())
    params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    i=0
    while i in range (0, len(params)):
                    times = params[i]['max_iters']
                    attempt = params[i]['max_attempts']
                    pop = params[i]['pop_size']
                    k_pct = params[i]['kpct']

                    print(N)
                    print(i)


                    #problem

                    np.random.seed(3)
                    fitness = mlrose.FourPeaks()

                    problem = mlrose.DiscreteOpt(length = N, fitness_fn = fitness, maximize = True, max_val = 2)

                    best_state, best_fitness = mlrose.mimic(problem, pop_size=pop, keep_pct = k_pct, \
                        max_attempts = attempt, max_iters = times)


                    results = (i, best_state, best_fitness, N, times, attempt, pop, k_pct)
                    mim_tr.append(results)
                   
                    # iterate  
                    i+=1

    #save the results and export
    regTable = pd.DataFrame(mim_tr)
    # rename the columns
    regTable.columns = ['Run_no', 'best_state', 'best_fitness', 'bits', 'iterations', 'attempts', 'pop_size', 'kpct']
    #regTable.to_csv('./output/{}_{}_reg.csv'.format("sa_ANN","wine"),index=False)


    # find parameters that max test accuracy
    #bestfit = regTable['best_fitness'].max()
    bestfit = regTable.loc[regTable['best_fitness'].idxmax()]
    mim_fp.append(bestfit)


    n+=1
#print(mim_fp)
mim_fp_table = pd.DataFrame(mim_fp)
mim_fp_table.columns = ['Run_no', 'best_state', 'best_fitness', 'bits', 'iterations', 'attempts', 'pop_size', 'kpct']
no_bits = mim_fp_table['bits']
b_fitness = mim_fp_table['best_fitness']
iters = mim_fp_table['iterations']
attempts = mim_fp_table['attempts']
popsize = mim_fp_table['pop_size']
keeppct = mim_fp_table['kpct']

def plot_metrics(X, y, title, ylab):

    plt.figure()
    plt.title(title)
    plt.xlabel("Bits")
    plt.ylabel(ylab)
    plt.grid()
    plt.plot(X, y, 'o-', color="r")
    return plt

title = "Fitness per problem size (MIMIC)"
ylab = "Fitness"
plot_metrics(no_bits, b_fitness, title, ylab)
plt.show()

title = "Iterations per problem size (MIMIC)"
ylab = "Iterations"
plot_metrics(no_bits, iters, title, ylab)
plt.show()

title = "Attempts per problem size (MIMIC)"
ylab = "Attempts"
plot_metrics(no_bits, attempts, title, ylab)
plt.show()

title = "Pop Size per problem size (MIMIC)"
ylab = "Pop Size"
plot_metrics(no_bits, popsize, title, ylab)
plt.show()

title = "Keep Pct per problem size (MIMIC)"
ylab = "Keep Pct"
plot_metrics(no_bits, keeppct, title, ylab)
plt.show()
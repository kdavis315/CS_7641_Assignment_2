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

#data prep
winedata = pd.read_csv('./winequality-white.csv')
winedata['High_Quality'] = np.where(winedata['quality']>=6, 1, 0)
winedata.drop(['quality'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(winedata.iloc[:, 0:11], winedata.iloc[:, 11:12], test_size = .15, random_state = 3)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#define plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



#Randomized hill climbing
np.random.seed(3)

rhc_model = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'sigmoid', \
	algorithm = 'random_hill_climb', max_iters = 1000, \
	bias = True, is_classifier = True, learning_rate = 0.0001, \
	early_stopping = True, clip_max = 5, max_attempts = 100)

start_time = clock()
rhc_model.fit(X_train_scaled, y_train)
training_time = clock()-start_time

title = "Learning Curve (RHC)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=3)

start_time = clock()
plot_learning_curve(rhc_model, title, X_train_scaled, y_train, cv=cv)
training_time = clock()-start_time

plt.show()

y_train_pred = rhc_model.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train, y_train_pred)
print("")
print("Random Hill Climbing Training accuracy")
print(y_train_accuracy)

y_test_pred = rhc_model.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test, y_test_pred)
print("Random Hill Climbing Test accuracy")
print(y_test_accuracy)

print("Random Hill Climbing Training Time")
print(training_time)

print("Random Hill Climbing Fitted Weights")
print(rhc_model.fitted_weights)


#simulated annealing
np.random.seed(3)

sa_model = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'sigmoid', \
	algorithm = 'simulated_annealing', schedule = mlrose.GeomDecay(), max_iters = 5000, \
	bias = True, is_classifier = True, learning_rate = 0.0001, \
	early_stopping = True, clip_max = 5, max_attempts = 100)

start_time = clock()
sa_model.fit(X_train_scaled, y_train)
training_time = clock()-start_time

title = "Learning Curve (SA)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=3)

start_time = clock()
plot_learning_curve(sa_model, title, X_train_scaled, y_train, cv=cv)
training_time = clock()-start_time

plt.show()

y_train_pred = sa_model.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train, y_train_pred)
print("")
print("Simulated Annealing Training accuracy")
print(y_train_accuracy)

y_test_pred = sa_model.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test, y_test_pred)
print("Simulated Annealing Test accuracy")
print(y_test_accuracy)

print("Simulated Annealing Training Time")
print(training_time)

print("Simulated Annealing Fitted Weights")
print(sa_model.fitted_weights)


#genetic algorithms
np.random.seed(3)

ga_model = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'sigmoid', \
	algorithm = 'genetic_alg', pop_size = 200, mutation_prob = 0.1, max_iters = 5000, \
	bias = True, is_classifier = True, learning_rate = 0.1, \
	early_stopping = True, clip_max = 5, max_attempts = 100)

start_time = clock()
ga_model.fit(X_train_scaled, y_train)
training_time = clock()-start_time

title = "Learning Curve (GA)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=3)

start_time = clock()
plot_learning_curve(ga_model, title, X_train_scaled, y_train, cv=cv)
training_time = clock()-start_time

plt.show()

y_train_pred = ga_model.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train, y_train_pred)
print("")
print("Genetic Algorithm Training accuracy")
print(y_train_accuracy)

y_test_pred = ga_model.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test, y_test_pred)
print("Genetic Algorithm Test accuracy")
print(y_test_accuracy)

print("Genetic Algorithm Training Time")
print(training_time)

print("Genetic Algorithm Fitted Weights")
print(ga_model.fitted_weights)

#Gradient Descent
np.random.seed(3)

gd_model = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'sigmoid', \
	algorithm = 'gradient_descent', max_iters = 1000, \
	bias = True, is_classifier = True, learning_rate = 0.0001, \
	early_stopping = True, clip_max = 5, max_attempts = 100)

start_time = clock()
gd_model.fit(X_train_scaled, y_train)
training_time = clock()-start_time


title = "Learning Curve (GD)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=3)

start_time = clock()
plot_learning_curve(gd_model, title, X_train_scaled, y_train, cv=cv)
training_time = clock()-start_time

plt.show()

y_train_pred = gd_model.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train, y_train_pred)
print("")
print("Gradient Descent Training accuracy")
print(y_train_accuracy)

y_test_pred = gd_model.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test, y_test_pred)
print("Gradient Descent Test accuracy")
print(y_test_accuracy)

print("Gradient Descent Training Time")
print(training_time)

print("Gradient Descent Fitted Weights")
print(gd_model.fitted_weights)
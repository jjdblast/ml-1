#!/usr/bin/env python

'''
GBDT of sklearn.
Follow the tutorial of DataRobot.
'''

from sklearn.datasets import make_hastie_10_2
from sklearn.cross_validation import train_test_split
from sklearn import ensemble

import numpy as np
import matplotlib.pyplot as plt


def ground_truth(x):
    return x * np.sin(x) + np.sin(2*x)

def gen_data(n_samples=200):
    np.random.seed(13)
    x = np.random.uniform(0, 10, size=n_samples)
    x.sort()
    y = ground_truth(x) + 0.75*np.random.normal(size=n_samples)
    train_mask = np.random.randint(0, 2, size=n_samples).astype(np.bool)
    x_train, y_train = x[train_mask, np.newaxis], y[train_mask]
    x_test, y_test = x[~train_mask, np.newaxis], y[~train_mask]
    return x_train, x_test, y_train, y_test

X_train, X_test, y_train, y_test = gen_data(200)    

x_plot = np.linspace(0, 10, 500)

def plot_data(figsize=(8,5)):
    fit = plt.figure(figsize=figsize)
    gt = plt.plot(x_plot, ground_truth(x_plot),
                alpha=0.4, label='ground truth')
    plt.scatter(X_train, y_train, s=10, alpha=0.4)
    plt.scatter(X_test, y_test, s=10, alpha=0.4, color='red')
    plt.xlim((0,10))
    plt.ylabel('y')
    plt.xlabel('x')

# fit one DT
from sklearn.tree import DecisionTreeRegressor    

#plot_data()
dtr = DecisionTreeRegressor(max_depth=1).fit(X_train, y_train)
plt.plot(x_plot, dtr.predict(x_plot[:, np.newaxis]),
       label='DRT max_depth=1', color='g', alpha=0.9, linewidth=2) 

dtr = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
plt.plot(x_plot, dtr.predict(x_plot[:, np.newaxis]),
        label='DRT max_depth=3', color='g', alpha=0.7, linewidth=1)    

#plt.legend(loc='upper left')
#plt.show()

from sklearn.ensemble import GradientBoostingRegressor
from itertools import islice

plot_data()

gbr = GradientBoostingRegressor(n_estimators=1000,
            max_depth=1, learning_rate=1.0)
gbr.fit(X_train, y_train)

#ax = plt.gca()
for pred in islice(gbr.staged_predict(x_plot[:, np.newaxis]),
                0, 1000, 10):
    plt.plot(x_plot, pred, color='r', alpha=0.2)

pred = gbr.predict(x_plot[:, np.newaxis])    
plt.plot(x_plot, pred, color='r', label='GBRT max_depth=1')
plt.legend(loc='upper left')
plt.show()


# diagnostic, deviance plot

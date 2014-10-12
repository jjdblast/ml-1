#!/usr/bin/env python

from numpy import *
import matplotlib.pyplot as plt


def poly_model(X, Y, d=5, r=1):
    X = asarray([X**i for i in range(0,d+1)]).T
    W = linalg.inv(dot(X.T,X)+r*diag(ones(d+1)))
    b = dot(dot(W,X.T),Y)
    return b

if __name__ == '__main__':
    X = loadtxt('data/ex5Linx.dat') 
    Y = loadtxt('data/ex5Liny.dat') 
    b0 = poly_model(X, Y, 5, 0)
    b1 = poly_model(X, Y, 5, 1)
    b2 = poly_model(X, Y, 5, 10)

    xl = X.min()
    xh = X.max()
    U = linspace(xl-0.1, xh+0.1, 50)
    W = asarray([U**i for i in range(0,6)]).T

    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.scatter(X, Y)
    ax.plot(U, dot(W,b0))

    ax = fig.add_subplot(312)
    ax.scatter(X, Y)
    ax.plot(U, dot(W,b1))

    ax = fig.add_subplot(313)
    ax.scatter(X, Y)
    ax.plot(U, dot(W,b2))

    fig.canvas.draw()
    plt.show()

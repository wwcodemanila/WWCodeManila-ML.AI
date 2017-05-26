import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def gradientDescent(X, y, theta, alpha, num_iters):
	m = len(y)
	J_history = np.zeros((num_iters, 1))

	for iter_ in range(num_iters):
		delta = np.zeros((X.shape[1], 1))
		for i in range(m):
			delta += np.matrix((theta.T.dot(X[i]) - y[i])*X[i]).transpose()
		theta -= alpha*(delta/m)

		J_history[iter_] = computeCost(X, y, theta)
	return theta

def computeCost(X, y, theta):
	m = len(y); 
	# One way to do it with for loops
	J = 0
	for i in range(m):
		J += (theta.T.dot(X[i]) - y[i])**2
	J = J/(2*m)

	# Vectorization
	h = np.dot(theta.T, X.T)

	# Sum of squared errors
	J = np.sum((h-y)**2)/(2*m)
	return J

def plotData(X, y, theta=None):
	plt.scatter(X, y)
	plt.ylabel('Profit in $10,000s');
	plt.xlabel('Population of City in 10,000s');

	if theta != None:
		m = len(y) 
		X = np.c_[np.ones(m), X] # Add a column of ones to x
		plt.plot(X[:, 1:], X.dot(theta), color = 'red')

	plt.show()

def loadData(filename):
	dataframe = pd.read_csv(filename, header=None) 
	X = dataframe.values[:, 0]
	y = dataframe.values[:, 1]
	return X, y

def main():
	filename = 'data.csv'
	num_iters = 1500;
	alpha = 0.01;

	X, y = loadData(filename)
	plotData(X, y)

	m = len(y) 
	X_ = np.c_[np.ones(m), X] # Add a column of ones to x
	theta = np.zeros((X_.shape[1], 1))

	print(computeCost(X, y, theta))
	theta = gradientDescent(X_, y, theta, alpha, num_iters)
	print(theta)
	plotData(X, y, theta=theta)

if __name__ == '__main__':
    main()
#!/usr/bin/env python
# -*- coding: utf8 -*-


####################################################
### You are not allowed to import anything else. ###
####################################################

import numpy as np


def power_sum(l, r, p=1.0):
	"""
		input: l, r - integers, p - float
		returns sum of p powers of integers from [l, r]
	"""
	if type(r) is 'int':
        return sum([i**p for i in range(1, r + 1)])


def solve_equation(a, b, c):
	"""
		input: a, b, c - integers
		returns float solutions x of the following equation: a x ** 2 + b x + c == 0
			In case of two diffrent solution returns tuple / list (x1, x2)
			In case of one solution returns one float
			In case of no float solutions return None 
			In case of infinity number of solutions returns 'inf'
	"""
    if a == 0:
        if b == 0:
            return 'inf'
        else:
            return -c/b
    else:
        D = b**2 - 4*a*c
        if D > 0:
            return [(-b + D ** (1/2.0)) / (2 * a), (-b - D ** (1/2.0)) / (2 * a)]
        else if D < 0:
            return None
        else:
            return -b / (2 * a)


def replace_outliers(x, std_mul=3.0):
	"""
		input: x - numpy vector, std_mul - positive float
		returns copy of x with all outliers (elements, which are beyond std_mul * (standart deviation) from mean)
		replaced with mean  
	"""
	threshold = np.std(x) * std_mul
    m = np.mean(x)
    res = list()
    for i in x:
        if i > threshold:
            res.appned(m)
        else:
            res.append(i)
    return np.array(res)


def get_eigenvector(A, alpha):
	"""
		input: A - square numpy matrix, alpha - float
		returns numpy vector - any eigenvector of A corresponding to eigenvalue alpha, 
		        or None if alpha is not an eigenvalue.
	"""
    eps = 1e-8
	v, w = np.linalg.eig(A)
    for i in range(0 ,len(v)):
        if abs(v[i] - alpha) < eps:
            return w[i]
    return None
    


def discrete_sampler(p):
	"""
		input: p - numpy vector of probability (non-negative, sums to 1)
		returns integer from 0 to len(p) - 1, each integer i is returned with probability p[i] 
	"""
	int(np.random.choice(len(p), 1, p = p))


def gaussian_log_likelihood(x, mu=0.0, sigma=1.0):
	"""
		input: x - numpy vector, mu - float, sigma - positive float
		returns log p(x| mu, sigma) - log-likelihood of x dataset 
		in univariate gaussian model with mean mu and standart deviation sigma
	"""
	return np.log(1 / sigma / (2 * np.pi ** (1 / 2.0)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)))


def gradient_approx(f, x0, eps=1e-8):
	"""
		input: f - callable, function of vector x. x0 - numpy vector, eps - float, represents step for x_i
		returns numpy vector - gradient of f in x0 calculated with finite difference method 
		(for reference use https://en.wikipedia.org/wiki/Numerical_differentiation, search for "first-order divided difference")
	"""
	return (f(x0 + eps) - f(x0 - eps)) / (2 * eps) 


def gradient_method(f, x0, n_steps=1000, learning_rate=1e-2, eps=1e-8):
	"""
		input: f - function of x. x0 - numpy vector, n_steps - integer, learning rate, eps - float.
		returns tuple (f^*, x^*), where x^* is local minimum point, found after n_steps of gradient descent, 
		                                f^* - resulting function value.
		Impletent gradient descent method, given in the lecture. 
		For gradient use finite difference approximation with eps step.
	"""
	raise NotImplementedError


def linear_regression_predict(w, b, X):
	"""
		input: w - numpy vector of M weights, b - bias, X - numpy matrix N x M (object-feature matrix), 
		N - number of objects, M - number of features.
		returns numpy vector of predictions of linear regression model for X
		https://xkcd.com/1725/
	"""
	return X.dot(w) + b


def mean_squared_error(y_true, y_pred):
	"""
		input: two numpy vectors of object targets and model predictions.
		return mse
	"""
	return ((y_true - y_pred) ** 2).mean(axis=0)


def linear_regression_mse_gradient(w, b, X, y_true):
	"""
		input: w, b - weights and bias of a linear regression model,
		        X - object-feature matrix, y_true - targets.
		returns gradient of linear regression model mean squared error w.r.t x and b
	"""
	raise NotImplementedError


class LinearRegressor:
	def fit(self, X_train, y_train, n_steps=1000, learning_rate=1e-2, eps=1e-8):
		"""
			input: object-feature matrix and targets.
			optimises mse w.r.t model parameters 
		"""
		raise NotImplementedError

		return self


	def predict(self, X):
		return linear_regression_predict(self.w, self.b, X)


def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))


def sigmoid_der(x):
	"""
		returns sigmoid derivative w.r.t. x
	"""
	return np.exp(-x)/(1 + np.exp(-x))**2


def relu(x):
	return np.maximum(x, 0)


def relu_der(x):
	"""
		return relu (sub-)derivative w.r.t x
	"""
	if x > 0:
        return 1
    else:
        return 0


class MLPRegressor:
	"""
		simple dense neural network class for regression with mse loss. 
	"""
	def __init__(self, n_units=[32, 32], nonlinearity=relu):
		"""
			input: n_units - number of neurons for each hidden layer in neural network,
			       nonlinearity - activation function applied between hidden layers.
		"""
		self.n_units = n_units
		self.nonlinearity = nonlinearity


	def fit(self, X_train, y_train, n_steps=1000, learning_rate=1e-2, eps=1e-8):
		"""
			input: object-feature matrix and targets.
			optimises mse w.r.t model parameters
			(you may use approximate gradient estimation)
		"""
		raise NotImplementedError


	def predict(self, X):
		"""
			input: object-feature matrix
			returns MLP predictions in X
		"""
		raise NotImplementedError
# small dataset of student test scores and the amount of hours 
# they studied. Intuitively, there must be a relationship right? 
# The more you study, the better your test scores should be. 
# We're going to use linear regression to prove this relationship.
import numpy as np


def compute_error_for_given_points(b, m, points):
	totalError = 0
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += (y - (m * x + b)) ** 2
	return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learning_rate):
	# gradient descent
	b_gradient = 0
	m_gradient = 0 
	N = float(len(points))
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
		m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
	# update b and m
	new_b = b_current - (learning_rate * b_gradient)
	new_m = m_current - (learning_rate * m_gradient)
	return [new_b, new_m]


def gradient_descent_runner(points, learning_rate, starting_b, starting_m, num_iterations):
	b = starting_b
	m = starting_m

	for i in range(num_iterations):
		b, m = step_gradient(b, m, np.array(points), learning_rate)
	return [b, m]


def run():
	# load data from csv file
	# by default genfromtxt has dtype=float, 
	# so dtype=None will try to guess the actual data type
	points = np.genfromtxt('test_study.csv', delimiter=',', dtype=None)
	# hyperparameters, with learning_rate = 0.01, 
	# gradient is overshooting and grad ~ 10e+30
	# so we will decrease learning rate to 0.0001
	learning_rate = 0.0001
	initial_b = 0
	initial_m = 0
	# when dataset is large use more no. of iteration, 
	# here dataset is small
	num_iterations = 1000
	# ideal values of b and m
	[b, m] = gradient_descent_runner(points, learning_rate, initial_b, initial_m, num_iterations)
	print(b, m)


if __name__ == '__main__':
	run()
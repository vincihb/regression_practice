import numpy as np
import math


def create_linear(num_points):
    x = [(a - num_points / 2.0) / (num_points / 20.0) for a in range(0, num_points, 1)]
    epsilon = np.random.standard_normal(num_points)
    m = 2.0
    b = 1.0
    y = [m * x_val + b for x_val in x]
    y = y + epsilon
    np.save("./datasets/linear.npy", (x, y))


def create_logistic(num_points):
    x = [(a - num_points / 2.0) / (num_points / 20.0) for a in range(0, num_points, 1)]
    epsilon = 0
    # epsilon = np.random.standard_normal(num_points)
    m = 2.0
    b = 1.0
    y_temp = [m * x_val + b for x_val in x]
    y_temp = np.array(y_temp) + epsilon
    y = [1.0 / (1 + math.exp(-y_val)) for y_val in y_temp]
    np.save("./datasets/logistic.npy", (x, y))


create_linear(1000)

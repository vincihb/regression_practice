import numpy as np
import matplotlib.pyplot as plt
import math


def estimate_linear(num_points, location="./datasets/linear.npy"):
    x, y = np.load(location)
    mu_y = np.mean(y)
    mu_x = np.mean(x)
    var_y = np.var(y)
    var_x = np.var(x)
    cov_xy = np.mean((x - mu_x) * (y - mu_y))
    b = mu_y - cov_xy / var_x * mu_x
    m = cov_xy / var_x
    print(b)
    print(m)
    x_arr = [(a - num_points / 2.0) / (num_points / 20.0) for a in range(0, num_points, 1)]
    y_hat = [m * x_val + b for x_val in x_arr]
    plt.plot(x, y, "ro")
    plt.plot(x_arr, y_hat)
    plt.show()


def estimate_logistic(num_points, location="./datasets/logistic.npy"):
    x, y = np.load(location)
    print(num_points)
    mu = []
    betas = np.random.standard_normal(2)
    n = 0
    while n < len(x):
        x_val = x[n]
        if n == 1:
            X = np.concatenate((np.array([X]), np.array([np.concatenate((np.ones(1), np.array([x_val])))])))
        elif n > 1:
            X = np.concatenate((np.array(X), np.array([np.concatenate((np.ones(1), np.array([x_val])))])))
        else:
            X = np.concatenate((np.ones(1), np.array([x_val])))
        print(X)
        mu_val = 1.0/(1 + math.exp(- betas[0] + betas[1] * x_val))
        mu.append(mu_val)
        print(mu_val)
        Y = np.array(y[0: n + 1])
        if n == 4:
            break
        n = n + 1


estimate_logistic(1000000)

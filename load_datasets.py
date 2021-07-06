import numpy as np
import matplotlib.pyplot as plt


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
    x_temp = []
    for x_val in x:
        x_temp.append(x_val)
        print(x_val)


estimate_logistic(1000000)

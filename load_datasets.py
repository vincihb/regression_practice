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


def shuffle_dataset(x, y):
    x = list(x)
    y = list(y)
    arr = [[], []]
    while x:
        index = np.random.randint(0, len(x))
        x_val = x.pop(index)
        y_val = y.pop(index)
        arr[0].append(x_val)
        arr[1].append(y_val)
    return arr


def estimate_logistic(num_points, location="./datasets/logistic.npy"):
    x, y = np.load(location)
    shuffled = shuffle_dataset(x, y)
    x = np.array(shuffled[0])
    y = np.array(shuffled[1])
    print(num_points)
    mu = []
    betas = [6, 7]
    n = 0
    while n < len(x):
        x_val = x[n]
        if n == 1:
            X = np.concatenate((np.array([X]), np.array([np.concatenate((np.ones(1), np.array([x_val])))])))
        elif n > 1:
            X = np.concatenate((np.array(X), np.array([np.concatenate((np.ones(1), np.array([x_val])))])))
        else:
            X = np.concatenate((np.ones(1), np.array([x_val])))
        if n % 100 == 0:
            print(n)
            print(betas)
        mu_val = 1.0 / (1 + math.exp(- betas[0] - betas[1] * x_val))
        mu.append(mu_val)
        Y = np.array(y[0: n + 1])
        S = np.diag([mu_val_temp for mu_val_temp in mu])
        if n == 0:
            print("Here")
            betas = [0, 0]
            n = n + 1
            continue
        temp = np.linalg.inv(np.matmul(np.matmul(np.transpose(X), S), X))
        temp2 = np.matmul(np.matmul(S, X), betas) + Y - mu
        temp2 = np.matmul(np.transpose(X), temp2)
        temp2 = np.matmul(temp, temp2)
        if (betas[0] - temp2[0]) ** 2 + (betas[1] - temp2[1]) ** 2 < 0.000001:
            break
        betas = temp2.copy()
        # if n == 4:
        #     break
        n = n + 1
    print(betas)


def estimate_logistic_1(num_points, location="./datasets/logistic.npy"):
    x, y = np.load(location)
    shuffled = shuffle_dataset(x, y)
    x = np.array(shuffled[0])
    y = np.array(shuffled[1])
    print(num_points)
    mu = []
    betas = [0, 0]
    n = 0
    while n < len(x):
        x_val = x[n]
        if n == 1:
            X = np.concatenate((np.array([X]), np.array([np.concatenate((np.ones(1), np.array([x_val])))])))
        elif n > 1:
            X = np.concatenate((np.array(X), np.array([np.concatenate((np.ones(1), np.array([x_val])))])))
        else:
            X = np.concatenate((np.ones(1), np.array([x_val])))
        mu_val = 1.0 / (1 + math.exp(- betas[0] - betas[1] * x_val))
        mu.append(mu_val)
        n = n + 1

    iteration = 0
    while True:
        if iteration % 100 == 0:
            print(iteration)
            print(betas)
        n = 0
        while n < len(x):
            x_val = x[n]
            mu_val = 1.0 / (1 + math.exp(- betas[0] - betas[1] * x_val))
            mu[n] = mu_val
            n = n + 1
        Y = np.array(y)
        S = np.diag(mu)
        temp = np.linalg.inv(np.matmul(np.matmul(np.transpose(X), S), X))
        temp2 = np.matmul(np.matmul(S, X), betas) + Y - mu
        temp2 = np.matmul(np.transpose(X), temp2)
        temp2 = np.matmul(temp, temp2)
        if math.sqrt((betas[0] - temp2[0]) ** 2 + (betas[1] - temp2[1]) ** 2) < 0.00000001:
            break
        iteration = iteration + 1
        betas = temp2.copy()
    print(betas)


estimate_logistic_1(1000000)

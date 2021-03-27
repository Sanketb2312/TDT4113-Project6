from utility import *
import numpy as np
import matplotlib.pyplot as mp


def t_sne(data_set, k, max_iter, alpha, epsilon):
    print("started")
    k = k_nearest_neighbors(pairwise_euclidean_distance(data_set), k)
    print("found k")
    p = (k + np.transpose(k) > 0).astype(float)

    rows = len(data_set)
    print("found p")

    y = np.random.normal(0, 10 ** (-4), (rows, 2))
    gain = np.ones((rows, 2))
    change = np.zeros((rows, 2))

    print("created y")

    q = np.zeros((rows, rows))
    already_calculated_to = 0
    for i in range(rows):
        for j in range(already_calculated_to, rows):
            if i != j:
                q[i][j] = 1 / (1 + square_diff(y[i], y[j]))
                q[j][i] = q[i][j]
        already_calculated_to += 1

    print("created q")

    p_norm = normalize(p)
    q_norm = normalize(q)

    print("normalized p and q")

    # d = 0
    # for i in range(len(p_norm)):
    #     for j in range(len(p_norm[i])):
    #         p_val = p_norm[i][j]
    #         if p_val != 0:
    #             d += p_val * np.log(p_val / q_norm[i][j])

    for it in range(max_iter):
        print(it)
        #  TODO: Vet ikke helt hva rangen til i og j skal være
        for i in range(len(y)):
            grad = np.zeros((1, 2))
            for j in range(len(y)):
                grad += ((p_norm[i][j] - q_norm[i][j]) * q[i][j]) * (y[i] - y[j])

            for k in range(2):
                if np.sign(grad[0][k]) != np.sign(change[i][k]):
                    gain[i][k] += 0.2
                else:
                    gain[i][k] *= 0.8

                if gain[i][k] < 0.01:
                    gain[i][k] = 0.01

            change[i] = alpha * change[i] - epsilon * (gain[i] @ grad[0])
            y[i] += change[i]

    return y


if __name__ == '__main__':
    data = read_data("data_files/digits.csv")
    c = read_data("data_files/digits_label.csv")
    result = t_sne(data[:100, :], 3, 100, 0.8, 500)
    x = result[:, 0]
    y = result[:, 1]
    print(x)
    print(y)
    print(c)
    mp.scatter(x, y, c[:100])
    mp.show()

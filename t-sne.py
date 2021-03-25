from utility import *
import numpy as np


def t_sne(data_set, k, max_iter):
    k = k_nearest_neighbors(pairwise_euclidean_distance(data_set), k)
    p = (k + np.transpose(k) > 0).astype(float)

    # TODO: Flytt ut til hjelpemetode?
    rows = len(data_set)

    y = []
    gain = []
    delta = []
    for i in range(rows):
        y.append([np.random.normal(0, 10**(-4)), np.random.normal(0, 10**(-4))])
        gain.append(1)
        delta.append(0)

    q = np.zeros((rows, rows))
    already_calculated_to = 0
    for i in range(rows):
        for j in range(already_calculated_to, rows):
            if i != j:
                # TODO: Her skal Y brukes, ikke data_set
                q[i][j] = 1 / (1 + square_diff(y[i], y[j]))
                q[j][i] = q[i][j]
        already_calculated_to += 1

    p_norm = normalize(p)
    q_norm = normalize(q)

    d = 0
    for i in range(len(p_norm)):
        for j in range(len(p_norm[i])):
            p_val = p_norm[i][j]
            if p_val != 0:
                d += p_val * np.log(p_val / q_norm[i][j])

    for i in range(max_iter):
        grad = (0, 0)
        for j in range(len(p_norm[i])):

            g_add = ((p_norm[i][j] - q_norm[i][j]) * q[i][j]) * (y[i] - y[j])



if __name__ == '__main__':
    x = [1, 2, 3]
    y = [2, 3, 4]

    x1 = [4]
    y2 = [2]

    x2 = [[1, 2, 3, 2, 4], [2, 4, 3, 3, 2], [2, 4, 4, 3, 1], [2, 4, 3, 3, 1], [2, 2, 4, 3, 1], [0, 0, 1, 0, 0]]

    t_sne(x2, 3, 1000)

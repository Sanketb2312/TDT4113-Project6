# Skal væra algoritme i følge oppgavaeteksten

from utility import *
import numpy as np


def t_sne(data_set, k):
    p = k_nearest_neighbors(pairwise_euclidean_distance(data_set), k)
    for row in p:
        for i in range(len(row)):
            if row[i] != 0:
                row[i] = 1

    q = []
    rows = len(data_set)
    temp = []
    for i in range(rows):
        temp.append(0)
    for i in range(rows):
        q.append(temp.copy())

    already_calculated_to = 0
    for i in range(rows):
        for j in range(already_calculated_to, rows):
            if i != j:
                q[i][j] = 1 / (1 + square_diff(data_set[i], data_set[j]))
                q[j][i] = q[i][j]
        already_calculated_to += 1

    P = normalize(p)
    Q = normalize(q)

    return q


if __name__ == '__main__':
    x = [1, 2, 3]
    y = [2, 3, 4]

    x1 = [4]
    y2 = [2]

    x2 = [[1, 2, 3, 2, 4], [2, 4, 3, 3, 2], [2, 4, 4, 3, 1], [2, 4, 3, 3, 1], [2, 2, 4, 3, 1], [0, 0, 1, 0, 0]]

    print(k_nearest_neighbors(pairwise_euclidean_distance(x2), 3))

    print(t_sne(x2, 3))

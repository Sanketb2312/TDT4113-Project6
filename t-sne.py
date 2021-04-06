"""modul"""
from utility import *
import numpy as np
import matplotlib.pyplot as mp


def t_sne(data_set, col, max_iter, alpha, epsilon):
    """t_sne functon"""
    np.random.seed(1)

    col = k_nearest_neighbors(pairwise_euclidean_distance(data_set), col)
    print("found k")
    p = (col + np.transpose(col) > 0).astype(float)
    print("found p")

    rows = len(data_set)

    y_y = np.random.normal(0, 10 ** (-4), (rows, 2))

    gain = np.ones((rows, 2))
    change = np.zeros((rows, 2))

    p_norm = normalize(p)

    for it in range(max_iter):
        print("iteration: ", it)

        a_a = None
        if it < 250:
            a_a = 0.5
        else:
            a_a = alpha

        q = np.divide(1, 1 + pairwise_euclidean_distance(y_y))
        np.fill_diagonal(q, 0)
        q_norm = normalize(q)
        print("big q found")

        g_g = None
        if it < 100:
            g_g = (4 * p_norm - q_norm) * q
        else:
            g_g = (p_norm - q_norm) * q

        s_s = np.diag(np.sum(g_g, axis=1))
        grad = 4 * (s_s - g_g) @ y_y
        print("grad found")
        for i in range(rows):
            for col in range(2):
                if np.sign(grad[i][col]) != np.sign(change[i][col]):
                    gain[i][col] += 0.2
                else:
                    gain[i][col] *= 0.8

                if gain[i][col] < 0.01:
                    gain[i][col] = 0.01

            change[i] = a_a * change[i] - epsilon * (gain[i] * grad[i])
            y_y[i] += change[i]
        print("end")

    return y_y


if __name__ == '__main__':
    data = read_data("data_files/digits.csv")
    c = read_data("data_files/digits_label.csv")
    result = t_sne(data, 50, 1000, 0.8, 500)
    x = result[:, 0]
    y = result[:, 1]
    mp.scatter(x, y, s=10, c=c, marker=".", cmap="jet")
    mp.show()

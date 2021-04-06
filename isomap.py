"""Functions for isomap"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.graph_shortest_path import graph_shortest_path

from utility import k_nearest_neighbors, pairwise_euclidean_distance


def compute_geodesics(d_matrix, k):
    """Computes geodesics matrix"""
    dist_matrix = np.sqrt(np.abs(pairwise_euclidean_distance(d_matrix)))
    d_knn = k_nearest_neighbors(dist_matrix, k)
    return graph_shortest_path(d_knn)


def multidimensional_scaling(d_geodesics):
    """Computes the multidimensional scaling"""
    print("MSD Starting")
    # Task a)
    d_2 = np.square(d_geodesics)

    # Task b)
    rows = len(d_2)
    j_matrix = np.identity(rows, dtype=int) - 1 / rows

    j_x_d2 = np.matmul(j_matrix, d_2)
    j_x_d2_x_j = np.matmul(j_x_d2, j_matrix)
    var_b = j_x_d2_x_j * (-1 / 2)

    # Task c)
    var_m = 2
    eigvals, eigvecs = np.linalg.eig(var_b)  # FØLER eigh er feil, skrev eig istden

    sorted_indexes = eigvals.argsort()[::-1]
    m_largest_eigvals = eigvals[sorted_indexes][0:var_m]
    sorted_eigvecs = (eigvecs.transpose()[sorted_indexes]).transpose()
    m_largest_eigvecs = sorted_eigvecs[:, 0:var_m]

    # Task d)
    e_m = m_largest_eigvecs
    tmp = [0] * var_m
    a_1_2 = []
    for i in range(var_m):
        a_1_2.append(tmp.copy())
        a_1_2[i][i] = np.sqrt(
            m_largest_eigvals[i])  # SKal egt være A_1_2[i][i] = np.sqrt(m_largest_eigvals[i])

    var_y = np.matmul(e_m, a_1_2)
    print(var_y)
    return var_y


if __name__ == '__main__':

    CHOOSE = 1

    if CHOOSE == 0:
        matrix = np.genfromtxt('data_files/swiss_data.csv', delimiter=',')
        SELECTED_K = 30
        selected_d_geodesics = compute_geodesics(matrix, SELECTED_K)
        mds = multidimensional_scaling(selected_d_geodesics).transpose()
        plt.scatter(mds[0], mds[1], c=np.arange(2000), cmap="jet")

    if CHOOSE == 1:
        matrix = np.genfromtxt('data_files/digits.csv', delimiter=',')
        SELECTED_K = 50
        selected_d_geodesics = compute_geodesics(matrix, SELECTED_K)
        mds = multidimensional_scaling(selected_d_geodesics).transpose()
        plt.scatter(mds[0], mds[1], c=np.genfromtxt('data_files/digits_label.csv'), cmap="jet",
                    s=10, marker=".")

    plt.show()

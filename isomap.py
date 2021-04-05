#skal væra method i følge oppgavaeteksten
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from sklearn.utils.graph_shortest_path import graph_shortest_path

from utility import k_nearest_neighbors, pairwise_euclidean_distance


def compute_geodesics(d_matrix, k):
    dist_matrix = pairwise_euclidean_distance(d_matrix)
    d_kNN = k_nearest_neighbors(dist_matrix, k)
    return graph_shortest_path(d_kNN)


def multidimensional_scaling(d_geodesics):
    print("MSD Starting")
    # Task a)
    d_2 = np.square(d_geodesics)

    # Task b)
    rows = len(d_2)
    tmp = [None] * rows
    '''j_matrix = []
    for i in range(rows):
        j_matrix.append(tmp.copy())
    for i in range(rows):
        for j in range(rows):
            if i == j:
                j_matrix[i][j] = 1 - 1 / (rows)
            else:
                j_matrix[i][j] = 0 - 1 / (rows)'''
    j_matrix = np.identity(rows, dtype=int)-1/rows

    j_x_d2 = np.matmul(j_matrix, d_2)
    j_x_d2_x_j = np.matmul(j_x_d2, j_matrix)
    b = j_x_d2_x_j * (-1 / 2)


    # Task c)
    m = 2
    eigvals, eigvecs = np.linalg.eig(b) # FØLER eigh er feil, skrev eig istden

    sortedIndexes = eigvals.argsort()[::-1]
    m_largest_eigvals = eigvals[sortedIndexes][0:m]
    sorted_eigvecs = (eigvecs.transpose()[sortedIndexes]).transpose()
    m_largest_eigvecs = sorted_eigvecs[:, 0:m]

    # Task d)
    E_m = m_largest_eigvecs
    tmp = [0]*m
    A_1_2 = []
    for i in range(m):
        A_1_2.append(tmp.copy())
        A_1_2[i][i] = np.sqrt(m_largest_eigvals[i]) # SKal egt være A_1_2[i][i] = np.sqrt(m_largest_eigvals[i])

    Y = np.matmul(E_m, A_1_2)
    print(Y)
    return Y

if __name__ == '__main__':

    #d_matrix = np.genfromtxt('data_files/swiss_data_fake.csv', delimiter=',')
    choose = 1

    if choose == 0:
        d_matrix = np.genfromtxt('data_files/swiss_data.csv', delimiter=',')
        k = 30
        d_geodesics = compute_geodesics(d_matrix, k)
        mds = multidimensional_scaling(d_geodesics).transpose()
        plt.scatter(mds[0], mds[1], c=np.arange(2000), cmap="jet")

    if choose == 1:
        d_matrix = np.genfromtxt('data_files/digits.csv', delimiter=',')
        k = 50
        d_geodesics = compute_geodesics(d_matrix, k)
        mds = multidimensional_scaling(d_geodesics).transpose()
        plt.scatter(mds[0], mds[1], c=np.genfromtxt('data_files/digits_label.csv'), cmap="jet", s=10, marker=".")





    plt.show()
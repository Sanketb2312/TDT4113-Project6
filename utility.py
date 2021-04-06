"""modul"""
import numpy as np


def square_diff(point1, point2):
    """Computes Euclidean distance between the two given points.
        Returns the square difference."""
    p1 = np.array([point1])
    p2 = np.array([point2])
    sum_of_p1s_squared = np.sum(np.multiply(p1, p1))
    sum_of_p2s_squared = np.sum(np.multiply(p2, p2))
    sum_of_p1p2s = np.sum(np.multiply(p1, p2))

    return sum_of_p1s_squared + sum_of_p2s_squared - 2 * sum_of_p1p2s


def pairwise_euclidean_distance(data_points):
    """Computes the distance between each point in the input matrix.
        Returns a result matrix r where r[i, j] is the distance between point i and point j."""
    d_1 = np.sum(data_points ** 2, axis=1, keepdims=True)
    d_2 = np.sum(data_points ** 2, axis=1)
    d_3 = np.dot(data_points, data_points.T)
    return d_1 + d_2 - 2 * d_3


def k_nearest_neighbors(d_matrix, k):
    """Computes the k nearest neighbors of each point given a distance matrix.
       Returns a result matrix r where r[i, j]
        is 0 if point j is not one of point i's k nearest neighbors."""
    k_matrix = d_matrix.copy()
    rows = len(d_matrix)
    for i in range(rows):
        sorted_indexes = np.argsort(d_matrix[i])
        for index in sorted_indexes[k + 1:]:
            k_matrix[i][index] = 0
            # print(index)
    return k_matrix


def normalize(matrix):
    """normalize"""
    return np.divide(matrix, np.sum(matrix))


def read_data(file_path):
    """read data"""
    data = np.genfromtxt(file_path, delimiter=',')
    return data

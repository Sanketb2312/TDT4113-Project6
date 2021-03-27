import numpy as np
from sklearn.utils.graph_shortest_path import graph_shortest_path


def square_diff(point1, point2):
    """Computes Euclidean distance between the two fiven points.
        Returns the square of the differance."""
    p1 = np.array([point1])
    p2 = np.array([point2])
    sum_of_p1s_squared = np.sum(np.multiply(p1, p1))
    sum_of_p2s_squared = np.sum(np.multiply(p2, p2))
    sum_of_p1p2s = np.sum(np.multiply(p1, p2))

    return sum_of_p1s_squared + sum_of_p2s_squared - 2 * sum_of_p1p2s


def pairwise_euclidean_distance(high_dimensional_data_points):
    """Computes the distance between each point in the input matrix.
        Returns a result matrix r where r[i, j] is the distance between point i and point j."""
    rows = len(high_dimensional_data_points)
    square_diff_matrix = []
    for amount in range(rows):
        pass

    temp = []
    for i in range(rows):
        temp.append(0)
    for i in range(rows):
        square_diff_matrix.append(temp.copy())

    already_calculated_to = 0
    for i in range(rows):
        for j in range(already_calculated_to, rows):
            if i != j:
                square_diff_matrix[i][j] = square_diff(high_dimensional_data_points[i],
                                                       high_dimensional_data_points[j])
                square_diff_matrix[j][i] = square_diff_matrix[i][j]
        already_calculated_to += 1
    return square_diff_matrix


def k_nearest_neighbors(d_matrix, k):
    """Computes the k nearest neighbors of each point given a distance matrix.
       Returns a result matrix r where r[i, j] is 0 if point j is not one of point i's k nearest neighbors."""
    rows = len(d_matrix)

    already_calculated_to = 0
    for i in range(rows):
        # k+1 to avoid counting already existing
        for j in range(rows - k - 1):
            max_value = max(d_matrix[i])
            max_index = d_matrix[i].index(max_value)
            d_matrix[i][max_index] = 0
        already_calculated_to += 1
    return d_matrix


def compute_geodesics(d_matrix, k):
    d_kNN = k_nearest_neighbors(pairwise_euclidean_distance(d_matrix), k)
    return graph_shortest_path(d_kNN)


def multidimensional_scaling(d_geodesics):
    # Task a)
    d_2 = np.multiply(d_geodesics, d_geodesics)

    # Task b)
    rows = len(d_2)
    tmp = [None] * rows
    j_matrix = []
    for i in range(rows):
        j_matrix.append(tmp.copy())
    for i in range(rows):
        for j in range(rows):
            if i == j:
                j_matrix[i][j] = 1 - 1 / (rows ** 2)
            else:
                j_matrix[i][j] = 0 - 1 / (rows ** 2)
    d2_x_j = np.matmul(d_2, j_matrix)
    b = d2_x_j * (-1 / 2)

    # Task c)
    m = 2
    eigvals, eigvecs = np.linalg.eig(b)
    m_largest_eigvecs = [None, None]
    m_largest_eigvals = [None, None]
    for i in range(len(eigvals)):
        for j in range(len(m_largest_eigvals)):
            if m_largest_eigvals[j] is None or eigvals[i] > m_largest_eigvals[j]:
                m_largest_eigvals[j] = eigvals[i]
                m_largest_eigvecs[j] = eigvecs[:, i]
    print(m_largest_eigvecs)
    return m_largest_eigvecs


if __name__ == '__main__':
    x = [1, 2, 3]
    y = [2, 3, 4]
    #print(square_diff(x, y))

    x1 = [4]
    y2 = [2]
    #print(square_diff(x1, y2))

    x2 = [[1, 2, 3, 2, 4], [2, 4, 3, 3, 2], [2, 4, 4, 3, 1], [2, 4, 3, 3, 1], [2, 2, 4, 3, 1],
          [0, 0, 1, 0, 0]]

    #print(pairwise_euclidean_distance(x2))

    #print(k_nearest_neighbors(pairwise_euclidean_distance(x2), 3))

    multidimensional_scaling(compute_geodesics(x2, 3))

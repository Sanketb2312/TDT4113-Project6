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
    return d_1 + d_2 - 2*d_3


def k_nearest_neighbors(d_matrix, k):
    """Computes the k nearest neighbors of each point given a distance matrix.
       Returns a result matrix r where r[i, j] is 0 if point j is not one of point i's k nearest neighbors."""
    k_matrix = d_matrix.copy()
    rows = len(d_matrix)
    for i in range(rows):
        #print(i)
        s = np.argsort(d_matrix[i])
        for index in s[k + 1:]:
            k_matrix[i][index] = 0
            #print(index)

    return k_matrix


def normalize(matrix):
    return np.divide(matrix, np.sum(matrix))

def read_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',')
    return data


if __name__ == '__main__':
    x = [1, 2, 3]
    y = [2, 3, 4]
    # print(square_diff(x, y))

    x1 = [4]
    y2 = [2]
    # print(square_diff(x1, y2))

    x2 = [[1, 2, 3, 2, 4], [2, 4, 3, 3, 2], [2, 4, 4, 3, 1], [2, 4, 3, 3, 1], [2, 2, 4, 3, 1], [0, 0, 1, 0, 0]]

    # print(pairwise_euclidean_distance(x2))

    #print(k_nearest_neighbors(pairwise_euclidean_distance(x2), 3))

    pairwise_euclidean_distance(read_data("data_files/digits.csv"))

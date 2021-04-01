from utility import *
import numpy as np
import matplotlib.pyplot as mp


def t_sne(data_set, k, max_iter, alpha, epsilon):
    print("started")
    x = np.genfromtxt("data_files/digits.csv", delimiter=",")

    k = k_nearest_neighbors(pairwise_euclidean_distance(data_set), k)
    print("found k")
    p = (k + np.transpose(k) > 0).astype(float)

    #rows = len(x)
    rows = max_iter
    print("HJKHKHBHKJ", rows)
    print("found p", p)

    y = np.random.normal(0, 10 ** (-4), (rows, 2))

    gain = np.ones((rows, 2))
    change = np.zeros((rows, 2))

    print("created y")

    q = np.zeros((rows, rows))
    already_calculated_to = 0
    for i in range(rows):
        for j in range(already_calculated_to, rows):
            if i != j:
                q[i][j] = (1 / (1 + square_diff(y[i], y[j])))
                q[j][i] = q[i][j]
        already_calculated_to += 1

    print("created q")

    p_norm = normalize(p)
    print(p_norm)
    q_norm = normalize(q)
    print(q_norm)

    print("normalized p and q")


    #Kommenter vekk enten 1 eller 2
    #METODE 1
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
                #print("p_norm", p_norm[i][j])
                #print("Q_norm", q_norm[i][j])
                #print("q,i,i", q[i][j])
                #print("yi", y[i] -y[j])
                #print("yj", np.subtract(y[i], y[j]))
                #print("yj", y[j])
                # print(((p_norm[i][j] - q_norm[i][j]) * q[i][j]) * (y[i] -
                # y[j]))
                grad += ((p_norm[i][j] - q_norm[i][j])
                         * q[i][j]) * (y[i] - y[j])

            for k in range(2):
                if np.sign(grad[0][k]) != np.sign(change[i][k]):
                    gain[i][k] += 0.2
                else:
                    gain[i][k] *= 0.8

                if gain[i][k] < 0.01:
                    gain[i][k] = 0.01

            print(gain[i], "gi")
            print(grad[0], "g0")
            change[i] = alpha * change[i] - epsilon * (gain[i] @ grad[0])
            y[i] += change[i]
            return y

    #METODE 2
    gain = np.ones((100, 2))
    change = np.zeros((100, 2))
    lie_p = 4 * p
    for count in range(max_iter):
        print(count)
        if count > 70:
            alpha = 0.8
            g_matrix = np.subtract(p, q) * q_norm
        else:
            g_matrix = np.subtract(lie_p, q) * q_norm
        s_matrix = np.diag(np.sum(g_matrix, axis=1))
        down_delta = np.matmul(4 * (s_matrix - g_matrix), y)
        for i in range(100):
            for column in range(2):
                if np.sign(down_delta[i][column]) != np.sign(change[i][column]):
                    gain[i][column] = gain[i][column] + 0.2
                else:
                    gain[i][column] = gain[i][column] * 0.8
                if gain[i][column] < 0.01:
                    gain[i][column] = 0.01
                change[i] = alpha * change[i] - epsilon * gain[i] * down_delta[i]
                y[i] = y[i] + change[i]
            print("Iteration: " + str(count))
        return y


if __name__ == '__main__':
    print("0.")
    data = read_data("data_files/digits.csv")
    print("1.")
    c = read_data("data_files/digits_label.csv")
    print("2.")
    result = t_sne(data[:3000, :], 25, 3000, 0.8, 500)
    print(result)
    print("3.")
    x = result[:, 0]
    print("4.")
    y = result[:, 1]
    print("5.")
    # print(x)
    # print(y)
    # print(c)
    #mp.scatter(x, y, c[:500])
    mp.scatter(x, y, s=10, c=c[:3000], marker=".", cmap='jet')
    print("6.")
    mp.show()

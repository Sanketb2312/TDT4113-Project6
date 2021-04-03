#skal væra class i følge oppgavaeteksten
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs, eigsh
class PCA:
    def __init__(self):
        self.data =''

    def read_data(self, path):
        self.data = np.genfromtxt(path, delimiter=',')


    def fit(self):
        centered_matrix = self.data-self.data.mean(axis=0)
        cov_matrix = np.cov(centered_matrix.transpose())
        D = self.data.shape[1]
        print(D)
        if D - 1 > 2:
            [eigenvalues, eigenvectors] = eigs(cov_matrix, k=2)
        else:
            [eigenvalues, eigenvectors] = eigsh(cov_matrix, k=2)
            ind_sorted = np.argsort(eigenvalues)
            eigenvectors = eigenvectors[:, ind_sorted[-2:]]
            eigenvectors = np.real(eigenvectors)
        return eigenvectors


    def transform(self):
        centered_matrix = self.data-self.data.mean(axis=0)
        f = self.fit()
        trans = f.transpose()
        return (trans@centered_matrix.transpose()).transpose()

    def plot(self, color):
        y_plot = self.transform()
        print(y_plot)
        if color == 2000:
            c_c = np.arange(color)
            plt.scatter(y_plot[:, 0], y_plot[:, 1], s=10, c=c_c, marker=".", cmap='jet')
            plt.show()
        elif color == 5620:
            color_plot = np.genfromtxt('data_files/digits_label.csv', delimiter=',')
            plt.scatter(y_plot[:, 0], y_plot[:, 1], s=10, c=color_plot, marker=".", cmap='jet')
            plt.show()





if __name__ == '__main__':
    pca = PCA()
    pca.read_data('data_files/swiss_data.csv')
    pca.plot(2000)
    pca.read_data('data_files/digits.csv')
    pca.plot(5620)

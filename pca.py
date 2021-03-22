#skal væra class i følge oppgavaeteksten
import numpy as np
import csv
from scipy.sparse.linalg import eigs, eigsh
class PCA:
    def __init__(self):
        self.numbers = []

    def read_data(self):
        data = np.genfromtxt('data_files/digits.csv', delimiter=',')
        print(data)
        return data

    def fit(self):
        my = self.read_data().mean()
        centered_matrix = self.read_data()-my
        cov_matrix = np.cov(centered_matrix)
        print(cov_matrix)
        D = len(self.read_data())
        if D - 1 > 1:
            [eigenvalues, eigenvectors]= eigs(cov_matrix, k=1)
        elif D-1 == 1:
            [eigenvalues, eigenvectors] = eigsh(cov_matrix, k=1)
        return [eigenvalues, eigenvectors]


        def transform(self):
            pass




p = PCA()
p.read_data()
p.fit()
import numpy as np
from math import e

"""
Algo EM correspondant au bloc C de l'article de Vidaure

"""

def block_C(X, y, n_samples, n_time_points, n_regions, decoding_mats_, n_clusters, max_iter):
    gamma = np.zeros((n_clusters, 1))
    for _ in range(max_iter):
        # E step
        somme_E = 0
        for cluster in range(n_clusters):
            somme_E += (e**(-0.5*(y - X@decoding_mats_[cluster])**2))
        for cluster in range(n_clusters):
            gamma[cluster,0] = (e**(-0.5*(y - X@decoding_mats_[cluster])**2)) / somme_E

        # M step
        somme_M = 0
        for cluster in range(n_clusters):
            somme_M += (gamma[cluster,0])
        # Xk =
        # Yk =

block_C()


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.stats as stats
import scipy.sparse as sp
from scipy.io import savemat
import os
from matplotlib import rc, rcParams

def random_unitary(n):
    return stats.ortho_group.rvs(n)

def conditioned_diagonal(n, kappa):
    diag = np.linspace(1.0, kappa, n)
    D = np.diag(diag)
    return D

def random_rhs(n):
    b = np.random.normal(scale=2.0, size=n) - 1.0
    b = b / la.norm(b)
    return b

iter_counts = []
def callback(xk):
    iter_counts.append(xk)

os.system("clear")

n_list = [3, 4, 5, 6]
kappa_list = [1.1, 1.5, 2, 3, 4, 5]
counter = 0
for i in range(len(n_list)):
    for j in range(len(kappa_list)):
        n = n_list[i]
        kappa = kappa_list[j]
        U = random_unitary(n)
        D = conditioned_diagonal(n, kappa)
        A = U @ D @ U.T

        b = random_rhs(n)

        iter_counts = []
        x, info = sp.linalg.cg(A, b, callback=callback, rtol=1e-6)
        num_iter = len(iter_counts)
        
        mdic = {"A": A, "b": b, "x": x, "num_iter": num_iter}
        savemat(f"data/test_{counter}.mat", mdic)
        
        counter += 1
from scipy.io import loadmat
import numpy.linalg as la
import numpy as np
import os
import gurobipy as grb

from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BQM

def conversion_mat(n, r, lower, upper):
    R0 = np.zeros((1, r))
    for i in range(r):
        R0[0, i] = 2**i
        
    R = np.zeros((n, n * r))
    for i in range(n):
        R[i, i*r:(i+1)*r] = R0 * (upper[i] - lower[i]) / 2**r
        
    z = lower.copy()
    
    return R, z

def quantum_classical_opt_sub(Q):
    model = grb.Model()
    n = Q.shape[0]
    x = model.addMVar(n, vtype=grb.GRB.BINARY, name="x")
    
    objective = grb.quicksum(grb.quicksum(Q[i][j] * x[i] * x[j] for j in range(n)) for i in range(n))
    
    model.setObjective(objective, grb.GRB.MINIMIZE)
    
    model.Params.OutputFlag = 0
    
    model.optimize()
    
    v = model.getVars()
    
    chi = np.zeros((n, 1))
    for i in range(n):
        chi[i] = v[i].x
    
    return chi

def classical_classical_opt_sub(A, b):
    model = grb.Model()
    n = A.shape[0]
    x = model.addMVar(n, vtype=grb.GRB.CONTINUOUS, name="x")
    
    objective1 = grb.quicksum(A[i][j] * x[i] * x[j] for i in range(n) for j in range(n))
    objective2 = grb.quicksum(b[i] * x[i] for i in range(n))
    
    model.setObjective(objective1 - objective2, grb.GRB.MINIMIZE)
    
    model.optimize()
    
    v = model.getVars()
    
    chi = np.zeros((n, 1))
    for i in range(n):
        chi[i] = v[i].x
    
    return chi

def classical_sub(x, r, lower, upper):
    n = x.shape[0]
    chi = np.zeros((n * r, 1))
    for i in range(n):
        estimation = np.copy(lower[i])
        for j in reversed(range(r)):
            R0 = 2**j * (upper[i] - lower[i]) / 2**r
            if estimation + R0 <= x[i]:
                estimation += R0
                chi[i*r+j] = 1
    
    return chi

def quantum_sub(Q):
    n = Q.shape[0]
    
    Q1 = np.zeros((n, n))
    for i in range(n):
        Q1[i, i] = Q[i, i]
        
    for i in range (n):
        for j in range(i):
            Q1[i, j] = Q[i, j] + Q[j, i]
    
    bqm = BQM(Q1, "BINARY")
    
    # put your token here
    sampler = EmbeddingComposite(DWaveSampler(
        token=""))
    sampler_set = sampler.sample(
        bqm, label='quantum sub', chain_strength=2, num_reads=500)
    
    solution = sampler_set.first.sample
    
    x = np.zeros((n, 1))
    selected_item_indices = [key for key, val in solution.items() if val == 1.0]
    x[selected_item_indices] = 1
    
    return x

def quantum_solver(A, b, x, r, max_iter=5):
    n = A.shape[0]
    y = np.zeros((n, 1))
    lower = np.ones((n, 1)) * -1
    upper = np.ones((n, 1))
    for it in range(max_iter):
        R, z = conversion_mat(n, r, lower, upper)
        
        Q1 = R.T @ A.T @ A @ R
        Q2 = np.diag(((A @ z - b).T @ A @ R).flatten())
        
        Q = Q1 + 2 * Q2
        
        chi = quantum_sub(Q)
        y = R @ chi + z
        old_lower = lower
        old_upper = upper
        lower = y - (old_upper - old_lower) / 2**(r-3)
        upper = y + (old_upper - old_lower) / 2**(r-3)
    
    return np.linalg.norm(x - y) / np.linalg.norm(x)

os.system("clear")

for i in range(6):
    data = loadmat(f"data/test_{i}.mat")
    A = data["A"]
    b = data["b"].T
    x = data["x"].T
    num_iter = data["num_iter"]
    
    # A = np.array([[3, 2, 1], [2, 5, 2], [1, 2, 6]])
    # x = np.array([[.5], [.5], [-0.5]])
    # b = A @ x
    
    print(A.shape)
    
    R = 6
    
    print(quantum_solver(A, b, x, R))
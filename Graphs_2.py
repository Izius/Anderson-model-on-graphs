import networkx as nx
import numpy as np
import sys
import random
import json
from set_mkl_lib import *
import os
from qutip import Qobj, ptrace, entropy_vn

try:

    ncores = mkl_rt.MKL_Get_Max_Threads()

except NameError:

    ncores = int(os.environ["OMP_NUM_THREADS"])


class Graphs:

    def __init__(self, L, type):
        self.N = 2 ** L
        self.type = type

    def create_graph(self, D, param):

        if self.type == 'RRG':
            return nx.random_regular_graph(D, self.N)

        elif self.type == 'SWN':
            return nx.newman_watts_strogatz_graph(self.N, D, param)

        elif self.type == 'ER':
            return nx.erdos_renyi_graph(self.N, param)

        elif self.type == 'URG':

            M = 2 ** param * self.N
            G = nx.random_regular_graph(D - 1, self.N)
            nodes = list(G.nodes())
            random.shuffle(nodes)

            i = 0
            j = 0
            m = 1

            while i < M / 2:

                if not G.has_edge(nodes[j], nodes[j + m]) and G.degree[nodes[j]] < D and G.degree[nodes[j + m]] < D:
                    G.add_edge(nodes[j], nodes[j + m])
                    i += 1
                    j += 2

                else:
                    j += 1

                if j >= self.N:
                    j = 0
                    m += 1

            return G

        else:
            pass

    def tight_binding(self, G, t):

        H = np.zeros((self.N, self.N))

        for i in G.nodes():
            for j in G.neighbors(i):
                H[i, j] = -t

        return H

    def disorder(self, W, distribution='uniform'):

        if distribution == 'uniform':
            diag = W / 2 * np.random.uniform(-1, 1, size=self.N)

        elif distribution == 'gaussian':
            diag = W / 2 * np.random.normal(0, 1, size=self.N)

        H = np.diag(diag)

        return H

    def diagonalize_and_filter(self, H, num_values=500):

        eigenvalues = np.linalg.eigvalsh(H)
        E = np.array(eigenvalues)
        mean = np.mean(E)
        absolute_diff = np.abs(E - mean)
        sorted_indices = np.argsort(absolute_diff)
        nearest_values = np.sort(E[sorted_indices[:num_values]])

        return nearest_values

    def separation(self, E):

        gaps = np.diff(E)
        ratios = np.ones(shape=len(gaps) - 1)

        for i in range(len(ratios)):
            pair = np.array([gaps[i], gaps[i + 1]])
            ratios[i] = np.min(pair) / np.max(pair)

        r_mean = np.mean(ratios)

        return r_mean

    def fractal_dimension(self, H, H_w):

        _, evecs1 = np.linalg.eigh(H)
        _, evecs2 = np.linalg.eigh(H_w)
        cc_list = []
        for ev1 in evecs1:
            c_list = []
            for ev2 in evecs2:
                c = np.abs(np.dot(ev1, ev2))**4 # on-site disorder basis
                c_list.append(c)
            cc_list.append(np.log(np.sum(c_list)))

        D = -np.mean(cc_list)/np.log(self.N)
        return D

    def entangelment_entropy(self, H, frac):

        frac = 2**(-frac)
        _, evecs = np.linalg.eigh(H)
        rho = sum(1/self.N*np.outer(vec, vec.conj()) for vec in evecs.T)
        rho_qobj = Qobj(rho)
        sub = list(range(frac*self.N))
        rho_partial = ptrace(rho_qobj, sub)
        S = entropy_vn(rho_partial, base=np.e)

        return S




# Arguments:
# 1. index of the W value to iterate through, 2. system size (7, 10, 14...), 3. graph type (RRG, SWN, ER, URG),
# 4. Parameter of a certain graph, for SWN and ER a float between 0 and 1, for URG -1, -2, -3...
# 5. number of disorder realizations (2000, 5000...)

idx = int(sys.argv[1])
L = int(sys.argv[2])
type = sys.argv[3]
D = L
param = float(sys.argv[4])
iter_W = int(sys.argv[5])
quantity = str(sys.argv[6])
frac = int(sys.argv[7])

t = 1.0

if L > 6 and L < 10:
    num_values = 50

elif L == 10:
    num_values = 100

elif L > 10 and L < 13:
    num_values = 200

elif L > 12:
    num_values = 500



start = 5.0
end = 55.0
step = 0.2
num_points = int((end - start) / step)
W_array = np.linspace(start, end, num_points)
W = W_array[idx - 1]

graph = Graphs(L, type)
G = graph.create_graph(D, param)
H_kin = graph.tight_binding(G, t)
vec = []

for i in range(iter_W):
    H_w = graph.disorder(W)
    H = H_kin + H_w
    if quantity == 'r':
        E = graph.diagonalize_and_filter(H, num_values)
        r_mean = graph.separation(E)
        vec.append(r_mean)

    if quantity == 'd':
        D = graph.fractal_dimension(H, H_w)
        vec.append(D)

    if quantity == 's':
        S = graph.entangelment_entropy(H, frac)
        vec.append(S)

q = np.mean(vec)

if idx < 10:
    with open(f"data/{type}_{L}_0{idx}", "w") as file:
        json.dump(q, file)

else:
    with open(f"data/{type}_{L}_{idx}", "w") as file:
        json.dump(q, file)
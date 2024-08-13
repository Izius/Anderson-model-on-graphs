import networkx as nx
import numpy as np
import sys
import random
import json
from set_mkl_lib import *
import os
from qutip import Qobj, ptrace, entropy_vn
from scipy.stats import gmean

try:

    ncores = mkl_rt.MKL_Get_Max_Threads()

except NameError:

    ncores = int(os.environ["OMP_NUM_THREADS"])


class Graphs:

    def __init__(self, L, type):
        self.N = 2 ** L
        self.type = type

    def create_graph(self, D, param):

        if self.type == 'RRG':  # random regular graph
            return nx.random_regular_graph(D, self.N)

        elif self.type == 'SWN':  # small-world network
            return nx.newman_watts_strogatz_graph(self.N, D, param)

        elif self.type == 'ER':  # Erdos-Renyi graph
            return nx.erdos_renyi_graph(self.N, param)

        elif self.type == 'URG':  # uniform random graph

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

    def tight_binding(self, G, t):  # tight-binding Hamiltonian

        H = np.zeros((self.N, self.N))

        for i in G.nodes():
            for j in G.neighbors(i):
                H[i, j] = -t

        return H

    def disorder(self, W, distribution='uniform'):  # diagonal disorder

        if distribution == 'uniform':
            diag = W / 2 * np.random.uniform(-1, 1, size=self.N)

        elif distribution == 'gaussian':
            diag = W / 2 * np.random.normal(0, 1, size=self.N)

        H = np.diag(diag)

        return H

    def diagonalize(self, H):  # diagonalize Hamiltonian

        eigenvalues = np.linalg.eigvalsh(H)
        E = np.array(eigenvalues)
        mean = np.mean(E)
        absolute_diff = np.abs(E - mean)
        sorted_indices = np.argsort(absolute_diff)
        nearest_values = np.sort(E[sorted_indices[:num_values]])

        return nearest_values

    def filter(self, E, num_values=500):  # filter eigenvalues

        mean = np.mean(E)
        absolute_diff = np.abs(E - mean)
        sorted_indices = np.argsort(absolute_diff)
        nearest_values = np.sort(E[sorted_indices[:num_values]])

        return nearest_values
    def separation(self, E):  # mean ratio of consecutive energy gaps

        gaps = np.diff(E)
        ratios = np.ones(shape=len(gaps) - 1)

        for i in range(len(ratios)):
            pair = np.array([gaps[i], gaps[i + 1]])
            ratios[i] = np.min(pair) / np.max(pair)

        r_mean = np.mean(ratios)

        return r_mean

    def level_spacing(self, E, param):  # mean level spacing

        gaps = np.diff(E)
        if param == 'arithmetic':
            return np.mean(gaps)

        if param == 'geometric':
            return gmean(gaps)


    def fractal_dimension(self, H):  # fractal dimension

        _, evecs = np.linalg.eigh(H)
        cc_list = []
        for ev in evecs:
            c_list = []
            for v in ev:
                c = np.abs(v)**4 # on-site disorder basis
                c_list.append(c)
            cc_list.append(np.log(np.sum(c_list)))

        D = -np.mean(cc_list)/np.log(self.N)
        return D

    def entangelment_entropy(self, H, frac):  # entanglement entropy

        frac = 2**(-frac)
        _, evecs = np.linalg.eigh(H)
        rho = sum(1/self.N*np.outer(vec, vec.conj()) for vec in evecs.T)
        rho_qobj = Qobj(rho)
        sub = list(range(frac*self.N))
        rho_partial = ptrace(rho_qobj, sub)
        S = entropy_vn(rho_partial, base=np.e)

        return S

    def site_occupation_operator(self, H):  # site occupation operator

        num_values = 500
        E, vec = np.linalg.eigh(H)
        mean = np.mean(E)
        absolute_diff = np.abs(E - mean)
        sorted_indices = np.argsort(absolute_diff)
        nearest_vecs = vec[sorted_indices[:num_values]]
        i = 0
        a_vec = []

        for nv in nearest_vecs:
            a = ((self.N*np.abs(nv[i])**2)-1)/(np.sqrt(self.N-1))
            a_vec.append(a)


        return a_vec

    def nn_neighbour_correlation(self, H, G):  # next-nearest neighbour correlation

        num_values = 500
        E, vec = np.linalg.eigh(H)
        mean = np.mean(E)
        absolute_diff = np.abs(E - mean)
        sorted_indices = np.argsort(absolute_diff)
        nearest_vecs = vec[sorted_indices[:num_values]]
        i = 0
        j = list(G.neighbors(i))[0]
        a_vec = []

        for nv in nearest_vecs:
            a = 2*np.sqrt(self.N/2)*(np.real(nv[i])*np.real(nv[j]) + np.imag(nv[i])*np.imag(nv[j]))
            a_vec.append(a)


        return a_vec

    def kinetic_energy(self, H, G):  # kinetic energy operator

        num_values = 500
        E, vec = np.linalg.eigh(H)
        mean = np.mean(E)
        absolute_diff = np.abs(E - mean)
        sorted_indices = np.argsort(absolute_diff)
        nearest_vecs = vec[sorted_indices[:num_values]]
        a_vec = []

        for nv in nearest_vecs:
            g_vec = []
            for i in G.nodes():
                for j in G.neighbors(i):
                    a = 2 * (np.real(nv[i]) * np.real(nv[j]) + np.imag(nv[i]) * np.imag(nv[j]))
                    g_vec.append(a)
            a_vec.append(np.sum(g_vec))

        return a_vec

    def fdme(self, o_vec):  # fluctuations of the diagonal matrix elements

        return np.mean(np.abs(np.diff(o_vec)))


# Arguments:
# 1. index of the W value to iterate through, 2. system size L (7, 10, 14...), 3. graph type (RRG, SWN, ER, URG),
# 4. Parameter of a certain graph, for SWN and ER a float between 0 and 1, for URG -1, -2, -3. For calculation
# of level spacing use 'arithmetic' or 'geometric'. For calculating observables us 'n', 'h' or 't'.
# 5. number of disorder realizations (2000, 5000...) # 6. ergodicity indicator to calculate (r, d, s, ls)
# or fluctuations of the diagonal matrix elements 'fdme' 7. fraction of the system size to calculate
# the entanglement entropy # 8. where to save your data


idx = int(sys.argv[1])
L = int(sys.argv[2])
type = sys.argv[3]
D = L
param = sys.argv[4]
iter_W = int(sys.argv[5])
quantity = str(sys.argv[6])
frac = int(sys.argv[7])
data = str(sys.argv[8])

t = 1.0

if L > 6 and L < 10:
    num_values = 50

elif L == 10:
    num_values = 100

elif L > 10 and L < 13:
    num_values = 200

elif L > 12:
    num_values = 500

#  W values for the Anderson model, depends on what you want to look at

# start = 5.0
# end = 55.0
# step = 1
# num_points = int((end - start) / step)
# W_array = np.linspace(start, end, num_points)
# W = W_array[idx - 1]

W_array = [3, 0.47*L**2, 255]
W = W_array[idx - 1]


graph = Graphs(L, type)
G = graph.create_graph(D, param)
H_kin = graph.tight_binding(G, t)
vec = []

for i in range(iter_W):
    H_w = graph.disorder(W)
    H = H_kin + H_w
    if quantity == 'r':
        E = graph.diagonalize(H)
        E = graph.filter(E, num_values)
        r_mean = graph.separation(E)
        vec.append(r_mean)

    if quantity == 'd':
        D = graph.fractal_dimension(H)
        vec.append(D)

    if quantity == 's':
        S = graph.entangelment_entropy(H, frac)
        vec.append(S)

    if quantity == 'ls':
        E = graph.diagonalize(H)
        ls = graph.level_spacing(E, param)
        vec.append(ls)

    if quantity == 'fdme':

        if param == 'n':
            o_vec = graph.site_occupation_operator(H)
            o = graph.fdme(o_vec)
            vec.append(o)

        if param == 'h':
            o_vec = graph.nn_neighbour_correlation(H, G)
            o = graph.fdme(o_vec)
            vec.append(o)

        if param == 't':
            o_vec = graph.kinetic_energy(H, G)
            o = graph.fdme(o_vec)
            vec.append(o)


q = np.mean(vec)

if idx < 10:
    with open(f"{data}/{type}_{L}_0{idx}", "w") as file:
        json.dump(q, file)

else:
    with open(f"{data}/{type}_{L}_{idx}", "w") as file:
        json.dump(q, file)
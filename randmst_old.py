#! /usr/bin/env python

from random import uniform
import heapq
from math import inf
import sys
import matplotlib.pyplot as plt

class WeightedGraph:
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.neighbors = [[] for _ in self.vertices]
        self.weights = {}

    # O(1)
    def get_weight(self, u,v):
        if (u,v) in self.weights.keys():
            return self.weights[(u,v)]
        else:
            return self.weights[(v,u)]

class UnifWeightCompleteGraph(WeightedGraph):
    def __init__(self, n, prune=False):
        super().__init__()
        self.vertices = list(range(n))
        self.edges = []
        self.neighbors = [[] for _ in range(n)]
        self.weights = {}

        if prune:
            prune_cutoff = n ** (-1/2)
        # O(n^2)
        for i in range(n):
            for j in range(i+1,n):
                weight = uniform(0,1)

                # Prune heavy edges from graph
                if prune \
                        and weight >= prune_cutoff \
                        and n >= 64 \
                        and len(self.neighbors[i]) >= 3 \
                        and len(self.neighbors[j]) >= 3:
                    continue

                self.edges.append((i,j))
                self.neighbors[i].append(j)
                self.neighbors[j].append(i)
                self.weights[(i,j)] = weight

class UnifUnitCubeDistances(WeightedGraph):
    def __init__(self, n, dim=2, prune=False):
        super().__init__()
        self.vertices = list(range(n))
        self.coordinates = {
                v: tuple(uniform(0,1) for _ in range(dim))
                for v in range(n)
                }
        self.edges = []
        self.neighbors = [[] for _ in range(n)]
        self.weights = {}

        if prune:
            prune_cutoff = n**(-1/(1.4*dim))

            # print(f"Pruning edges heavier than {prune_cutoff}")

        # O(n^2)
        for i in range(n):
            for j in range(i+1,n):
                coords_i = self.coordinates[i]
                coords_j = self.coordinates[j]
                weight = sum(
                        (xi-xj)**2
                        for xi,xj in zip(coords_i,coords_j)
                        )**(1/2)

                # Prune heavy edges from graph
                if prune \
                        and weight >= prune_cutoff \
                        and n >= 64 \
                        and len(self.neighbors[i]) >= 3 \
                        and len(self.neighbors[j]) >= 3:
                    continue

                self.edges.append((i,j))
                self.neighbors[i].append(j)
                self.neighbors[j].append(i)
                self.weights[(i,j)] = weight


class SpanningTree():
    def __init__(self, G, init_node, predecessors):
        self.parent = G
        self.vertices = G.vertices
        self.init_node = init_node
        self.predecessors = predecessors

    def get_weight(self, v):
        if v == self.init_node:
            return 0

        else:
            return self.parent.get_weight(self.predecessors[v], v)

    def total_weight(self):
        return sum(self.get_weight(v) for v in self.vertices)

    def max_weight(self):
        return max(self.get_weight(v) for v in self.vertices)

    def weights(self):
        return [self.get_weight(v) for v in self.vertices]

    def __str__(self):
        return str(self.predecessors)

# Prim's algorithm
def find_mst(G: WeightedGraph):
    # print("Finding MST...")
    # measure performance
    # steps = 0
    # If G is empty, we return an empty graph
    if len(G.vertices) == 0:
        return WeightedGraph()

    # Pick an arbitrary base vertex
    init_node = G.vertices[0]

    # Initialize distance table
    q = [(0, init_node, inf)]
    predecessors = {}

    while q:
        cost, v, pred = heapq.heappop(q)
        if v in predecessors.keys():
            continue
        predecessors[v] = pred
        for u in G.neighbors[v]:
            # steps += 1
            if u not in predecessors.keys():
                c = G.get_weight(v,u)
                heapq.heappush(q, (c, u, v))

    # print(f"Found MST in {steps} steps")

    return SpanningTree(G, init_node, predecessors)


def run_experiment(num_points, num_trials, dim=2, prune=False, show_hist = False):
    """
    Compute a random complete graph with n vertices,
    and return the weight of its MST.

    n: number of vertices in the graph
    dim: dimension of the unit hypercube from which vertices are uniformly sampled.
    Use dim = 0 to weight edges as independent uniform random variables.
    """
    mst_weights = []
    max_weights = []
    if show_hist:
        all_weights = []

    if dim==0:
        if prune and num_points >= 100:
            print(f"Genererating {num_trials} pruned uniform random graphs with {num_points} vertices")
        else:
            print(f"Genererating {num_trials} uniform random graphs with {num_points} vertices")
        for trial in range(num_trials):
            G = UnifWeightCompleteGraph(num_points, prune=prune)
            mst = find_mst(G)
            mst_weights.append(mst.total_weight())
            max_weights.append(mst.max_weight())
            if show_hist:
                all_weights += mst.weights()
    else:
        if prune and num_points >= 100:
            print(f"Genererating {num_trials} pruned {dim}-cube graphs with {num_points} vertices")
        else:
            print(f"Genererating {num_trials} {dim}-cube graphs with {num_points} vertices")
        for trial in range(num_trials):
            G = UnifUnitCubeDistances(num_points, dim, prune=prune)
            mst = find_mst(G)
            mst_weights.append(mst.total_weight())
            max_weights.append(mst.max_weight())
            if show_hist:
                all_weights += mst.weights()

    avg_weight = sum(mst_weights)/num_trials
    max_weight = sum(max_weights)/num_trials

    if show_hist:
        # Draw histogram of weights
        plt.hist(all_weights)
        plt.xlabel("Weight")
        plt.ylabel("Frequency")
        plt.show()

    print(f"Avg weight of MST: {avg_weight:.6g}")
    print(f"Avg max weight edge in MST: {max_weight:.6g}")

    return avg_weight

if __name__ == "__main__":
    argv = sys.argv[1:]

    prune, num_points, num_trials, dim = (int(s) for s in argv[:4])


    run_experiment(num_trials, num_points, dim, prune=prune, show_hist=True)


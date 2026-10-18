import time


class DisjointSet:
    def __init__(self, vertices):
        self.parent = {vertex: vertex for vertex in vertices}
        self.rank = {vertex: 0 for vertex in vertices}

    def find(self, vertex):
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])

        return self.parent[vertex]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)

        if root_u == root_v:
            return False

        if self.rank[root_u] < self.rank[root_v]:
            self.parent[root_u] = root_v

        elif self.rank[root_u] > self.rank[root_v]:
            self.parent[root_v] = root_u

        else:
            self.parent[root_v] = root_u
            self.rank[root_u] += 1

        return True


def kruskal_mst(graph):
    start_time = time.perf_counter()

    vertices = list(graph.nodes())
    edges = list(graph.edges(data=True))

    # Sort edges according to weight
    edges.sort(key=lambda edge: edge[2]["weight"])

    disjoint_set = DisjointSet(vertices)

    mst_edges = []
    total_cost = 0

    for u, v, data in edges:

        if disjoint_set.union(u, v):
            mst_edges.append((u, v, data["weight"]))
            total_cost += data["weight"]

        if len(mst_edges) == len(vertices) - 1:
            break

    execution_time = time.perf_counter() - start_time

    return mst_edges, total_cost, execution_time
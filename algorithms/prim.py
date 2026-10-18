import time
import heapq


def prim_mst(graph):
    start_time = time.perf_counter()

    if len(graph.nodes()) == 0:
        return [], 0, 0

    start_node = list(graph.nodes())[0]

    visited = set()
    priority_queue = []

    mst_edges = []
    total_cost = 0

    visited.add(start_node)

    for neighbor in graph.neighbors(start_node):
        weight = graph[start_node][neighbor]["weight"]

        heapq.heappush(
            priority_queue,
            (weight, start_node, neighbor)
        )

    while priority_queue and len(visited) < len(graph.nodes()):

        weight, u, v = heapq.heappop(priority_queue)

        if v in visited:
            continue

        visited.add(v)

        mst_edges.append((u, v, weight))
        total_cost += weight

        for neighbor in graph.neighbors(v):

            if neighbor not in visited:

                edge_weight = graph[v][neighbor]["weight"]

                heapq.heappush(
                    priority_queue,
                    (edge_weight, v, neighbor)
                )

    execution_time = time.perf_counter() - start_time

    return mst_edges, total_cost, execution_time
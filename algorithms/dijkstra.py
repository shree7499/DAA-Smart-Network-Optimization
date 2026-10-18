import time
import heapq


def dijkstra(graph, source, destination):

    start_time = time.perf_counter()

    distances = {
        node: float("inf")
        for node in graph.nodes()
    }

    previous = {
        node: None
        for node in graph.nodes()
    }

    distances[source] = 0

    priority_queue = [(0, source)]

    while priority_queue:

        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        if current_node == destination:
            break

        for neighbor in graph.neighbors(current_node):

            weight = graph[current_node][neighbor]["weight"]

            distance = current_distance + weight

            if distance < distances[neighbor]:

                distances[neighbor] = distance
                previous[neighbor] = current_node

                heapq.heappush(
                    priority_queue,
                    (distance, neighbor)
                )

    path = []

    current = destination

    while current is not None:
        path.append(current)
        current = previous[current]

    path.reverse()

    if distances[destination] == float("inf"):
        path = []
        shortest_distance = None
    else:
        shortest_distance = distances[destination]

    execution_time = time.perf_counter() - start_time

    return path, shortest_distance, execution_time
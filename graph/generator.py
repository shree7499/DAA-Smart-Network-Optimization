import random
import networkx as nx


def generate_network(
    num_nodes=10,
    extra_edges=10,
    min_weight=1,
    max_weight=50,
    seed=None
):

    if seed is not None:
        random.seed(seed)

    graph = nx.Graph()

    # Add routers
    for i in range(1, num_nodes + 1):
        graph.add_node(f"R{i}")

    nodes = list(graph.nodes())

    # First create a connected backbone
    shuffled_nodes = nodes.copy()
    random.shuffle(shuffled_nodes)

    for i in range(len(shuffled_nodes) - 1):

        u = shuffled_nodes[i]
        v = shuffled_nodes[i + 1]

        weight = random.randint(
            min_weight,
            max_weight
        )

        graph.add_edge(
            u,
            v,
            weight=weight
        )

    # Add additional random connections
    possible_edges = []

    for i in range(num_nodes):

        for j in range(i + 1, num_nodes):

            u = nodes[i]
            v = nodes[j]

            if not graph.has_edge(u, v):
                possible_edges.append((u, v))

    random.shuffle(possible_edges)

    number_of_extra_edges = min(
        extra_edges,
        len(possible_edges)
    )

    for i in range(number_of_extra_edges):

        u, v = possible_edges[i]

        weight = random.randint(
            min_weight,
            max_weight
        )

        graph.add_edge(
            u,
            v,
            weight=weight
        )

    return graph
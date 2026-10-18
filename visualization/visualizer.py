import matplotlib.pyplot as plt
import networkx as nx


def draw_network(
    graph,
    highlighted_edges=None,
    shortest_path=None,
    title="Network"
):

    fig, ax = plt.subplots(figsize=(11, 7))

    pos = nx.spring_layout(
        graph,
        seed=42
    )

    # Draw all nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=900,
        ax=ax
    )

    # Draw all edges
    nx.draw_networkx_edges(
        graph,
        pos,
        width=1.5,
        alpha=0.5,
        ax=ax
    )

    # Highlight MST edges
    if highlighted_edges:

        mst_edges = [
            (u, v)
            for u, v, weight in highlighted_edges
        ]

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=mst_edges,
            width=4,
            ax=ax
        )

    # Highlight shortest path
    if shortest_path and len(shortest_path) > 1:

        path_edges = list(
            zip(
                shortest_path[:-1],
                shortest_path[1:]
            )
        )

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=path_edges,
            width=5,
            ax=ax
        )

    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=10,
        font_weight="bold",
        ax=ax
    )

    edge_labels = nx.get_edge_attributes(
        graph,
        "weight"
    )

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        ax=ax
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold"
    )

    ax.axis("off")

    return fig
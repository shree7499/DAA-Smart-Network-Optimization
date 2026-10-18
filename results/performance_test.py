import pandas as pd
from pathlib import Path

from graph.generator import generate_network
from algorithms.kruskal import kruskal_mst
from algorithms.prim import prim_mst
from algorithms.dijkstra import dijkstra


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Results directory
RESULTS_DIR = BASE_DIR / "results"

# Output CSV
OUTPUT_FILE = RESULTS_DIR / "performance_results.csv"


results = []

sizes = [10, 25, 50, 100, 250, 500]


print("\nRunning algorithm performance test...\n")


for size in sizes:

    print(f"Testing network with {size} routers...")

    graph = generate_network(
        num_nodes=size,
        extra_edges=size * 2,
        min_weight=1,
        max_weight=100,
        seed=42
    )

    # -------------------------
    # Kruskal
    # -------------------------

    _, _, kruskal_time = kruskal_mst(graph)

    # -------------------------
    # Prim
    # -------------------------

    _, _, prim_time = prim_mst(graph)

    # -------------------------
    # Dijkstra
    # -------------------------

    nodes = list(graph.nodes())

    source = nodes[0]
    destination = nodes[-1]

    _, _, dijkstra_time = dijkstra(
        graph,
        source,
        destination
    )

    results.append(
        {
            "Nodes": size,
            "Edges": len(graph.edges()),
            "Kruskal Time": kruskal_time,
            "Prim Time": prim_time,
            "Dijkstra Time": dijkstra_time
        }
    )


# Create DataFrame
df = pd.DataFrame(results)


# Display results
print("\n==============================================")
print("       ALGORITHM PERFORMANCE RESULTS")
print("==============================================\n")

print(df.to_string(index=False))


# Save CSV
df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n==============================================")
print("Performance test completed successfully!")
print("==============================================")

print(f"\nCSV saved at:")
print(OUTPUT_FILE)
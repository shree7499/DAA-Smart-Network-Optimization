import matplotlib

# Use a non-GUI backend
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Results folder
RESULTS_DIR = BASE_DIR / "results"

# Input CSV
INPUT_FILE = RESULTS_DIR / "performance_results.csv"

# Output graph
OUTPUT_FILE = RESULTS_DIR / "algorithm_comparison.png"


# Load performance data
df = pd.read_csv(INPUT_FILE)


print("\nPerformance Data:")
print(df.to_string(index=False))


# Create graph
plt.figure(figsize=(10, 6))


plt.plot(
    df["Nodes"],
    df["Kruskal Time"],
    marker="o",
    label="Kruskal"
)

plt.plot(
    df["Nodes"],
    df["Prim Time"],
    marker="o",
    label="Prim"
)

plt.plot(
    df["Nodes"],
    df["Dijkstra Time"],
    marker="o",
    label="Dijkstra"
)


plt.xlabel("Number of Routers")

plt.ylabel("Execution Time (seconds)")

plt.title(
    "Algorithm Execution Time Comparison"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


# Save graph
plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)


# Close figure instead of opening Tkinter
plt.close()


print("\n==============================================")
print("Performance graph created successfully!")
print("==============================================")

print("\nGraph saved at:")
print(OUTPUT_FILE)
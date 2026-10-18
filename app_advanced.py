"""Advanced Smart Network Optimization dashboard.

Run with: python -m streamlit run app_advanced.py
Requires: streamlit, networkx, matplotlib, pandas, numpy
"""

from __future__ import annotations

import io
import json
import time
from itertools import islice
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Smart Network Optimization | Advanced",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(140deg, #0b1220 0%, #101b2d 55%, #0b1525 100%); }
    section[data-testid="stSidebar"] { background: #0a1322; border-right: 1px solid #24334a; }
    .hero { padding: 1.2rem 1.4rem; border-radius: 18px; background: linear-gradient(120deg,#142943,#102034); border: 1px solid #2b4564; margin-bottom: 1rem; }
    .hero h1 { margin: 0; color: #f4f8ff; font-size: 2.05rem; }
    .hero p { margin: .45rem 0 0 0; color: #b9cce3; }
    .small-note { color: #9db0c8; font-size: .88rem; }
    div[data-testid="stMetric"] { background: rgba(23,43,66,.65); border: 1px solid rgba(130,175,220,.18); padding: 12px 14px; border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Graph generation, editing validation, algorithms, and visualization helpers
# -----------------------------------------------------------------------------
def generate_network(n: int, extra_edges: int, min_weight: int, max_weight: int,
                     seed: int, topology_style: str) -> nx.Graph:
    """Generate a connected, undirected, positively weighted network."""
    if n < 2:
        raise ValueError("Choose at least two routers.")
    if min_weight < 1 or max_weight <= min_weight:
        raise ValueError("Link costs must be positive and minimum cost must be below maximum cost.")

    rng = np.random.default_rng(int(seed))
    graph = nx.Graph()
    nodes = [f"R{i}" for i in range(1, n + 1)]
    graph.add_nodes_from(nodes)

    # A randomized chain guarantees the initial topology is connected.
    shuffled = nodes.copy()
    rng.shuffle(shuffled)
    for u, v in zip(shuffled, shuffled[1:]):
        graph.add_edge(u, v, weight=int(rng.integers(min_weight, max_weight + 1)))

    possible_count = n * (n - 1) // 2 - (n - 1)
    if topology_style == "Sparse / tree-like":
        target_extra = 0
    elif topology_style == "Balanced":
        target_extra = min(possible_count, max(n, extra_edges))
    elif topology_style == "Dense":
        target_extra = min(possible_count, max(2 * n, extra_edges))
    else:
        target_extra = min(possible_count, extra_edges)

    candidates = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)
                  if not graph.has_edge(nodes[i], nodes[j])]
    rng.shuffle(candidates)
    for u, v in candidates[:target_extra]:
        graph.add_edge(u, v, weight=int(rng.integers(min_weight, max_weight + 1)))
    return graph


def edge_frame(graph: nx.Graph) -> pd.DataFrame:
    rows = [{"Source": str(u), "Target": str(v), "Cost": int(data.get("weight", 1))}
            for u, v, data in graph.edges(data=True)]
    return pd.DataFrame(rows, columns=["Source", "Target", "Cost"])


def graph_from_edge_frame(nodes: list[str], frame: pd.DataFrame) -> nx.Graph:
    """Validate an editable edge table and build its graph; reject ambiguous edits."""
    result = nx.Graph()
    result.add_nodes_from(nodes)
    seen = set()
    for row_number, row in enumerate(frame.to_dict("records"), start=1):
        source = row.get("Source")
        target = row.get("Target")
        cost = row.get("Cost")
        vals = [source, target, cost]
        if all(pd.isna(value) or str(value).strip() == "" for value in vals):
            continue  # blank dynamic-editor row
        if pd.isna(source) or pd.isna(target) or pd.isna(cost) or str(source).strip() == "" or str(target).strip() == "":
            raise ValueError(f"Row {row_number}: complete Source, Target, and Cost for every link.")
        source, target = str(source).strip(), str(target).strip()
        if source not in nodes or target not in nodes:
            raise ValueError(f"Row {row_number}: router names must exist in the router list.")
        if source == target:
            raise ValueError(f"Row {row_number}: self-links are not allowed.")
        try:
            numeric_cost = float(cost)
        except (TypeError, ValueError):
            raise ValueError(f"Row {row_number}: cost must be a number.")
        if not np.isfinite(numeric_cost) or numeric_cost <= 0 or not numeric_cost.is_integer():
            raise ValueError(f"Row {row_number}: cost must be a positive whole number.")
        key = tuple(sorted((source, target)))
        if key in seen:
            raise ValueError(f"Duplicate link: {source} — {target}. Each undirected link should appear once.")
        seen.add(key)
        result.add_edge(source, target, weight=int(numeric_cost))
    return result


def _edge_sort_key(edge):
    u, v, data = edge
    left, right = sorted((str(u), str(v)))
    return (float(data.get("weight", 1)), left, right)


def mst_kruskal(graph: nx.Graph):
    parent = {node: node for node in graph.nodes()}
    rank = {node: 0 for node in graph.nodes()}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    chosen = []
    for u, v, data in sorted(graph.edges(data=True), key=_edge_sort_key):
        if union(u, v):
            chosen.append((u, v, int(data.get("weight", 1))))
            if len(chosen) == max(0, graph.number_of_nodes() - 1):
                break
    return chosen, sum(weight for _, _, weight in chosen)


def mst_prim(graph: nx.Graph):
    if not graph.number_of_nodes():
        return [], 0
    import heapq
    start = sorted(graph.nodes(), key=str)[0]
    visited = {start}
    heap = []
    for neighbor, data in graph[start].items():
        heapq.heappush(heap, (int(data.get("weight", 1)), str(start), str(neighbor), start, neighbor))
    chosen = []
    while heap and len(visited) < graph.number_of_nodes():
        weight, _, _, source, target = heapq.heappop(heap)
        if target in visited:
            continue
        visited.add(target)
        chosen.append((source, target, weight))
        for neighbor, data in graph[target].items():
            if neighbor not in visited:
                heapq.heappush(heap, (int(data.get("weight", 1)), str(target), str(neighbor), target, neighbor))
    return chosen, sum(weight for _, _, weight in chosen)


def mst_boruvka(graph: nx.Graph):
    parent = {node: node for node in graph.nodes()}
    size = {node: 1 for node in graph.nodes()}
    component_count = graph.number_of_nodes()

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        nonlocal component_count
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        component_count -= 1
        return True

    ordered_edges = sorted(graph.edges(data=True), key=_edge_sort_key)
    chosen = []
    while component_count > 1:
        cheapest = {}
        for u, v, data in ordered_edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                continue
            edge = (int(data.get("weight", 1)), min(str(u), str(v)), max(str(u), str(v)), u, v)
            if ru not in cheapest or edge[:3] < cheapest[ru][:3]:
                cheapest[ru] = edge
            if rv not in cheapest or edge[:3] < cheapest[rv][:3]:
                cheapest[rv] = edge
        if not cheapest:
            break  # disconnected graph
        changed = False
        for weight, _, _, u, v in sorted(cheapest.values(), key=lambda item: item[:3]):
            if union(u, v):
                chosen.append((u, v, weight))
                changed = True
        if not changed:
            break
    return chosen, sum(weight for _, _, weight in chosen)


def mst_reverse_delete(graph: nx.Graph):
    """Compute an MST by removing expensive edges when connectivity is preserved.

    This is intentionally included as an educational comparison algorithm. It is
    usually less efficient than Kruskal, Prim, or Boruvka because it repeatedly
    checks graph connectivity.
    """
    if graph.number_of_nodes() == 0:
        return [], 0
    if not nx.is_connected(graph):
        raise ValueError("Reverse-Delete MST requires a connected graph.")

    working = graph.copy()
    # Try the most expensive links first; stable tie-breaking makes runs repeatable.
    ordered_edges = sorted(
        graph.edges(data=True),
        key=lambda edge: (float(edge[2].get("weight", 1)),
                          min(str(edge[0]), str(edge[1])),
                          max(str(edge[0]), str(edge[1]))),
        reverse=True,
    )
    for u, v, data in ordered_edges:
        if not working.has_edge(u, v):
            continue
        saved_data = dict(working[u][v])
        working.remove_edge(u, v)
        if not nx.is_connected(working):
            working.add_edge(u, v, **saved_data)

    chosen = [
        (u, v, int(data.get("weight", 1)))
        for u, v, data in sorted(working.edges(data=True), key=_edge_sort_key)
    ]
    return chosen, sum(weight for _, _, weight in chosen)


def route_cost(graph: nx.Graph, path: list[str]) -> int:
    return int(sum(graph[u][v].get("weight", 1) for u, v in zip(path, path[1:])))


def draw_graph(graph: nx.Graph, title: str, highlight_edges=None, highlight_path=None):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    if graph.number_of_nodes() == 0:
        ax.set_title(title)
        ax.text(.5, .5, "No routers in this network", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig
    pos = nx.spring_layout(graph, seed=42, weight="weight", k=None)
    # Algorithm result edges may be (u, v) pairs or (u, v, weight) triples.
    # Normalize both formats here so visualizing an MST never fails on weights.
    highlight = set()
    for edge in (highlight_edges or []):
        if edge is not None and len(edge) >= 2:
            u, v = edge[0], edge[1]
            highlight.add(frozenset((str(u), str(v))))
    if highlight_path:
        highlight.update(frozenset((str(u), str(v))) for u, v in zip(highlight_path, highlight_path[1:]))
    normal_edges, selected_edges = [], []
    for u, v in graph.edges():
        (selected_edges if frozenset((str(u), str(v))) in highlight else normal_edges).append((u, v))
    nx.draw_networkx_edges(graph, pos, edgelist=normal_edges, ax=ax, edge_color="#6b7f98", width=1.4, alpha=.7)
    if selected_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=selected_edges, ax=ax, edge_color="#ffb84d", width=3.2, alpha=1)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="#5bc0eb", edgecolors="#e6f7ff", linewidths=1.2, node_size=650)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_weight="bold", font_color="#0b1525")
    if graph.number_of_edges() <= 65:
        labels = {(u, v): data.get("weight", 1) for u, v, data in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, ax=ax, font_size=7, font_color="#f1f5f9", rotate=False,
                                     bbox={"alpha": .65, "color": "#142237", "pad": 1})
    ax.set_title(title, color="#eaf2ff", fontsize=13, pad=12)
    ax.set_facecolor("#101c2c")
    fig.patch.set_facecolor("#101c2c")
    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_and_close(graph: nx.Graph, title: str, highlight_edges=None, highlight_path=None):
    fig = draw_graph(graph, title, highlight_edges, highlight_path)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def add_graph_version():
    st.session_state.graph_version = st.session_state.get("graph_version", 0) + 1
    st.session_state.mst_results = None
    st.session_state.path_result = None
    st.session_state.all_pairs = None
    st.session_state.benchmark_df = None


def connected(graph: nx.Graph) -> bool:
    return graph.number_of_nodes() > 0 and nx.is_connected(graph)


def generate_benchmark_graph(n: int, seed: int):
    # Keep tests sparse, but connected, so timings remain practical on a laptop.
    return generate_network(n=n, extra_edges=min(max(n, int(n * 1.5)), n * (n - 1) // 2 - (n - 1)),
                            min_weight=1, max_weight=100, seed=seed, topology_style="Custom edge count")


# -----------------------------------------------------------------------------
# Session state defaults
# -----------------------------------------------------------------------------
if "graph" not in st.session_state:
    st.session_state.graph = generate_network(12, 14, 1, 50, 42, "Balanced")
if "graph_version" not in st.session_state:
    st.session_state.graph_version = 0
for key in ("mst_results", "path_result", "all_pairs", "benchmark_df"):
    if key not in st.session_state:
        st.session_state[key] = None

G: nx.Graph = st.session_state.graph


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Network controls")
    st.caption("Start with a generated network, then edit individual links below.")
    n_nodes = st.slider("Routers to generate", min_value=5, max_value=100, value=12, step=1)
    topology_style = st.selectbox(
        "Starting topology",
        ["Sparse / tree-like", "Balanced", "Dense", "Custom edge count"],
        index=1,
        help="Sparse starts with a spanning-tree-like network; Balanced and Dense add more links. Every generated graph starts connected.",
    )
    extra_edges = st.slider("Extra links (Custom mode)", min_value=0, max_value=300, value=14, step=1)
    left, right = st.columns(2)
    with left:
        min_cost = st.number_input("Minimum link cost", min_value=1, max_value=1000, value=1, step=1)
    with right:
        max_cost = st.number_input("Maximum link cost", min_value=2, max_value=10000, value=50, step=1)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    if st.button("🎲 Generate / regenerate", type="primary", use_container_width=True):
        if min_cost >= max_cost:
            st.error("Minimum cost must be smaller than maximum cost.")
        else:
            st.session_state.graph = generate_network(n_nodes, extra_edges, int(min_cost), int(max_cost), int(seed), topology_style)
            add_graph_version()
            st.rerun()

    st.markdown("---")
    st.markdown("### Add / remove a router")
    new_node = st.text_input("New router ID", placeholder="e.g. DataCenter", key="new_node_input")
    if st.button("＋ Add router", use_container_width=True):
        label = new_node.strip()
        if not label:
            st.warning("Enter a router ID first.")
        elif label in st.session_state.graph:
            st.error("That router already exists.")
        else:
            st.session_state.graph.add_node(label)
            add_graph_version()
            st.rerun()
    current_nodes_sidebar = sorted(st.session_state.graph.nodes(), key=str)
    if current_nodes_sidebar:
        node_to_remove = st.selectbox("Router to remove", current_nodes_sidebar, key="node_remove_select")
        if st.button("− Remove router and its links", use_container_width=True):
            if len(st.session_state.graph.nodes) <= 2:
                st.error("Keep at least two routers in the network.")
            else:
                st.session_state.graph.remove_node(node_to_remove)
                add_graph_version()
                st.rerun()

    st.markdown("---")
    if st.button("↩ Reset to starter network", use_container_width=True):
        st.session_state.graph = generate_network(12, 14, 1, 50, 42, "Balanced")
        add_graph_version()
        st.rerun()


G = st.session_state.graph
nodes = sorted(G.nodes(), key=str)
edge_count = G.number_of_edges()
node_count = G.number_of_nodes()
weight_sum = sum(int(data.get("weight", 1)) for _, _, data in G.edges(data=True))
density = nx.density(G) if node_count > 1 else 0.0

st.markdown(
    "<div class='hero'><h1>🌐 Smart Network Optimization</h1>"
    "<p>Design your topology · edit link costs · compare MST algorithms · find routes · evaluate resilience</p></div>",
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Routers", f"{node_count:,}")
m2.metric("Links", f"{edge_count:,}")
m3.metric("All-link cost", f"{weight_sum:,}")
m4.metric("Network density", f"{density:.3f}")
m5.metric("Components", nx.number_connected_components(G) if node_count else 0)

if node_count == 0:
    st.error("The network has no routers. Generate a network from the sidebar.")
elif not connected(G):
    st.warning("This topology is disconnected. Connect the components before running MST optimization or routes across components.")


# -----------------------------------------------------------------------------
# Main app tabs
# -----------------------------------------------------------------------------
tab_network, tab_mst, tab_routes, tab_perf, tab_insights = st.tabs(
    ["🛠️ Network Designer", "🌳 MST Comparison", "🛣️ Route Planner", "📈 Performance Lab", "🛡️ Insights & Export"]
)

with tab_network:
    st.subheader("Network topology and link editor")
    st.markdown("Change a source, destination, or cost; add new rows for links; delete a row to remove a link. Click **Apply link changes** to update the actual graph.")
    top_left, top_right = st.columns([1.7, 1])
    with top_left:
        draw_and_close(G, "Current network topology")
    with top_right:
        st.markdown("#### Topology summary")
        st.write(f"**Routers:** {node_count}  ")
        st.write(f"**Links:** {edge_count}  ")
        st.write(f"**Connected:** {'Yes' if connected(G) else 'No'}  ")
        if edge_count:
            costs = [int(data.get("weight", 1)) for _, _, data in G.edges(data=True)]
            st.write(f"**Minimum / maximum link cost:** {min(costs)} / {max(costs)}")
            st.write(f"**Average link cost:** {np.mean(costs):.2f}")
        st.caption("Link cost can represent expense, latency, distance, or another metric. Use one consistent unit per analysis.")

    st.markdown("#### Edit links")
    editor_key = f"edge_editor_{st.session_state.graph_version}"
    current_edges = edge_frame(G)
    edited_edges = st.data_editor(
        current_edges,
        key=editor_key,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Source": st.column_config.SelectboxColumn("Source router", options=nodes, required=True),
            "Target": st.column_config.SelectboxColumn("Target router", options=nodes, required=True),
            "Cost": st.column_config.NumberColumn("Link cost", min_value=1, max_value=10_000_000, step=1, format="%d", required=True),
        },
        disabled=False,
    )
    if st.button("✅ Apply link changes", type="primary", key="apply_edge_changes"):
        try:
            updated = graph_from_edge_frame(nodes, edited_edges)
            st.session_state.graph = updated
            add_graph_version()
            st.success("Topology updated. MSTs and routes will use the new network.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    with st.expander("Import a custom network from CSV"):
        st.caption("CSV columns must be `Source`, `Target`, and `Cost` (capitalization is flexible). The imported routers are inferred from the links.")
        uploaded = st.file_uploader("Choose an edge-list CSV", type=["csv"], key="edge_csv_upload")
        if uploaded is not None and st.button("Import CSV topology", key="import_csv_graph"):
            try:
                incoming = pd.read_csv(uploaded)
                mapping = {str(col).strip().lower(): col for col in incoming.columns}
                source_col = next((mapping[k] for k in ("source", "router a", "u", "from") if k in mapping), None)
                target_col = next((mapping[k] for k in ("target", "router b", "v", "to") if k in mapping), None)
                cost_col = next((mapping[k] for k in ("cost", "weight", "link cost") if k in mapping), None)
                if not source_col or not target_col or not cost_col:
                    raise ValueError("CSV needs source, target, and cost/weight columns.")
                incoming = incoming.rename(columns={source_col: "Source", target_col: "Target", cost_col: "Cost"})[["Source", "Target", "Cost"]]
                imported_nodes = sorted(set(incoming["Source"].dropna().astype(str)) | set(incoming["Target"].dropna().astype(str)))
                imported_graph = graph_from_edge_frame(imported_nodes, incoming)
                if imported_graph.number_of_nodes() < 2:
                    raise ValueError("The file must define at least two routers and one link.")
                st.session_state.graph = imported_graph
                add_graph_version()
                st.success(f"Imported {imported_graph.number_of_nodes()} routers and {imported_graph.number_of_edges()} links.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not import CSV: {exc}")

    export_edges = edge_frame(G).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download current topology (CSV)", data=export_edges, file_name="network_topology.csv", mime="text/csv")

with tab_mst:
    st.subheader("Compare four classic Minimum Spanning Tree algorithms")
    st.markdown(
        "Kruskal, Prim, Borůvka (also known as Sollin's algorithm), and Reverse-Delete "
        "are run on the **same weighted topology**. For a connected graph, they should "
        "return the same minimum total cost, although tied link weights can produce "
        "different valid MST edge sets."
    )
    st.caption(
        "Reverse-Delete is included for learning and comparison; its repeated connectivity "
        "checks often make it slower than the other three algorithms."
    )
    if not connected(G):
        st.error("A single MST requires a connected graph. Open Network Designer and add links until the component count is one.")
    else:
        if st.button("⚡ Compare all four MST algorithms", type="primary", key="run_mst_compare"):
            implementations = {
                "Kruskal": mst_kruskal,
                "Prim": mst_prim,
                "Borůvka / Sollin": mst_boruvka,
                "Reverse-Delete": mst_reverse_delete,
            }
            reference_tree = nx.minimum_spanning_tree(G, algorithm="kruskal", weight="weight")
            reference_cost = int(sum(data.get("weight", 1) for _, _, data in reference_tree.edges(data=True)))
            result = {}
            for name, function in implementations.items():
                t0 = time.perf_counter()
                selected, cost = function(G)
                elapsed = time.perf_counter() - t0
                selected_graph = nx.Graph()
                selected_graph.add_nodes_from(G.nodes())
                selected_graph.add_weighted_edges_from(selected)
                result[name] = {
                    "edges": selected,
                    "cost": int(cost),
                    "seconds": elapsed,
                    "valid_tree": nx.is_tree(selected_graph),
                    "matches_reference": int(cost) == reference_cost,
                    "reference_cost": reference_cost,
                }
            st.session_state.mst_results = result
        result = st.session_state.mst_results
        if result:
            rows = []
            for name, info in result.items():
                rows.append({
                    "Algorithm": name,
                    "Minimum total cost": info["cost"],
                    "Links selected": len(info["edges"]),
                    "Runtime (ms)": round(info["seconds"] * 1000, 4),
                    "Cost reduction vs all links (%)": round((1 - info["cost"] / weight_sum) * 100, 2) if weight_sum else 0,
                    "Valid spanning tree": "Yes" if info["valid_tree"] else "No",
                    "Matches reference MST cost": "Yes" if info["matches_reference"] else "No",
                })
            result_df = pd.DataFrame(rows)
            st.dataframe(result_df, hide_index=True, use_container_width=True)
            best_cost = min(item["cost"] for item in result.values())
            all_valid = all(item["valid_tree"] and item["matches_reference"] for item in result.values())
            if all_valid and all(item["cost"] == best_cost for item in result.values()):
                st.success(f"All four algorithms returned a valid spanning tree with the same minimum total cost: {best_cost}.")
            else:
                st.error("At least one result failed validation against the reference MST. Inspect the algorithm and graph data before relying on the result.")

            # A small visual runtime comparison helps users interpret performance results.
            runtime_df = result_df.set_index("Algorithm")[["Runtime (ms)"]]
            st.markdown("#### Runtime comparison")
            st.bar_chart(runtime_df, use_container_width=True)

            st.markdown("#### Inspect an algorithm's selected MST")
            chosen_name = st.selectbox("Choose an MST to visualize", list(result.keys()), key="mst_view_choice")
            chosen = result[chosen_name]
            draw_and_close(G, f"{chosen_name} minimum spanning tree", highlight_edges=chosen["edges"])
            st.caption("Highlighted links belong to the selected MST. Other links remain visible for comparison with the original network.")
            st.markdown("#### Selected links")
            mst_frame = pd.DataFrame(chosen["edges"], columns=["Source", "Target", "Cost"])
            st.dataframe(mst_frame, hide_index=True, use_container_width=True)
            st.download_button("⬇️ Download all-algorithm comparison", data=result_df.to_csv(index=False).encode("utf-8"),
                               file_name="mst_algorithm_comparison.csv", mime="text/csv")
            st.download_button("⬇️ Download selected MST links", data=mst_frame.to_csv(index=False).encode("utf-8"),
                               file_name=f"{chosen_name.lower().replace('/', '_').replace(' ', '_')}_mst_edges.csv", mime="text/csv")
            st.info("The cost-reduction metric compares retaining every current link against retaining only the selected MST links. It is not a real-world monetary saving unless link costs are calibrated to actual costs.")
        else:
            st.info("Click **Compare all four MST algorithms** to see total cost, selected links, execution time, validation, and topology for each method.")

with tab_routes:
    st.subheader("Route planner: choose the routers and what 'best' means")
    if node_count < 2:
        st.warning("Add at least two routers first.")
    else:
        route_left, route_right = st.columns(2)
        with route_left:
            source = st.selectbox("Starting router", nodes, index=0, key="route_source")
        with route_right:
            destinations = [node for node in nodes if node != source]
            destination = st.selectbox("Destination router", destinations, index=0, key="route_destination")
        route_algorithm = st.selectbox(
            "Routing algorithm / objective",
            ["Dijkstra — lowest total cost", "Bellman–Ford — weighted route", "Breadth-first search — fewest hops"],
            help="Dijkstra and Bellman–Ford minimize the sum of link costs here. BFS minimizes number of links (hops), not the weighted cost.",
        )
        st.caption("Change link costs or add/remove a link in Network Designer, apply the edits, and compute the route again to see the new optimal path.")
        if st.button("Find route", type="primary", use_container_width=True, key="compute_route"):
            try:
                start_time = time.perf_counter()
                if route_algorithm.startswith("Dijkstra"):
                    path = nx.dijkstra_path(G, source, destination, weight="weight")
                elif route_algorithm.startswith("Bellman"):
                    path = nx.bellman_ford_path(G, source, destination, weight="weight")
                else:
                    path = nx.shortest_path(G, source, destination)  # unweighted: minimum hops
                elapsed = time.perf_counter() - start_time
                st.session_state.path_result = {
                    "version": st.session_state.graph_version,
                    "source": source, "destination": destination,
                    "algorithm": route_algorithm, "path": path,
                    "cost": route_cost(G, path), "hops": len(path) - 1,
                    "seconds": elapsed,
                }
            except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
                st.session_state.path_result = None
                st.error(f"No route is available for this pair: {exc}")
            except nx.NetworkXUnbounded:
                st.session_state.path_result = None
                st.error("Bellman–Ford detected a negative cycle. Shortest routes are undefined for this network.")
        path_result = st.session_state.path_result
        if (path_result and path_result.get("version") == st.session_state.graph_version
                and path_result.get("source") == source
                and path_result.get("destination") == destination
                and path_result.get("algorithm") == route_algorithm):
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Total link cost", path_result["cost"])
            p2.metric("Hops", path_result["hops"])
            p3.metric("Routers on route", len(path_result["path"]))
            p4.metric("Execution", f"{path_result['seconds'] * 1000:.4f} ms")
            st.markdown("**Selected route**")
            st.code("  →  ".join(path_result["path"]), language="text")
            draw_and_close(G, f"{path_result['algorithm']} route", highlight_path=path_result["path"])
            route_frame = pd.DataFrame({"Step": range(1, len(path_result["path"]) + 1), "Router": path_result["path"]})
            st.dataframe(route_frame, hide_index=True, use_container_width=True)
            st.download_button("⬇️ Download route", data=route_frame.to_csv(index=False).encode("utf-8"),
                               file_name="selected_route.csv", mime="text/csv")

            with st.expander("Show alternative routes (up to 3 lowest-cost simple paths)"):
                try:
                    path_iter = nx.shortest_simple_paths(G, source, destination, weight="weight")
                    alts = []
                    for candidate in islice(path_iter, 3):
                        alts.append({"Route": " → ".join(candidate), "Cost": route_cost(G, candidate), "Hops": len(candidate) - 1})
                    alts_df = pd.DataFrame(alts)
                    st.dataframe(alts_df, hide_index=True, use_container_width=True)
                    if len(alts) > 1:
                        st.caption("These alternatives are ranked by total link cost, independently of the selected routing algorithm.")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    st.info("No alternative route exists.")
        elif not path_result:
            st.info("Choose the source, destination, and routing algorithm, then click **Find route**.")

    with st.expander("All-pairs shortest-path matrix (Floyd–Warshall)"):
        st.caption("This computes the lowest weighted cost between every pair of routers. For responsiveness, use it on networks with up to 35 routers.")
        if node_count > 35:
            st.warning("Reduce the network to 35 routers or fewer to enable the all-pairs matrix.")
        elif st.button("Compute all-pairs matrix", key="all_pairs_button"):
            st.session_state.all_pairs = {"version": st.session_state.graph_version,
                                          "matrix": dict(nx.floyd_warshall(G, weight="weight"))}
        all_pairs = st.session_state.all_pairs
        if all_pairs and all_pairs.get("version") == st.session_state.graph_version:
            matrix = pd.DataFrame(all_pairs["matrix"]).reindex(index=nodes, columns=nodes)
            st.dataframe(matrix, use_container_width=True)
            st.download_button("⬇️ Download all-pairs matrix", data=matrix.to_csv().encode("utf-8"),
                               file_name="all_pairs_shortest_paths.csv", mime="text/csv")

with tab_perf:
    st.subheader("Performance lab")
    st.markdown("Benchmark algorithms on reproducible generated graphs. We report the median run time over repeated executions to reduce noise from a single measurement.")
    pc1, pc2 = st.columns(2)
    with pc1:
        max_benchmark_nodes = st.select_slider("Largest benchmark network", options=[50, 100, 150, 200, 300], value=150)
    with pc2:
        repeats = st.slider("Repeated runs per size", min_value=1, max_value=7, value=3)
    if st.button("Run benchmark suite", type="primary", key="run_benchmarks"):
        sizes = sorted(set([10, 25, 50, min(100, max_benchmark_nodes), max_benchmark_nodes]))
        sizes = [size for size in sizes if size <= max_benchmark_nodes]
        bench_rows = []
        progress = st.progress(0, text="Preparing repeatable benchmark graphs...")
        for idx, size in enumerate(sizes):
            bench_graph = generate_benchmark_graph(size, 2026 + size)
            times = {"Kruskal": [], "Prim": [], "Borůvka": [], "Reverse-Delete": [], "Dijkstra": []}
            for _ in range(repeats):
                for name, fn in (("Kruskal", mst_kruskal), ("Prim", mst_prim),
                                 ("Borůvka", mst_boruvka), ("Reverse-Delete", mst_reverse_delete)):
                    t0 = time.perf_counter(); fn(bench_graph); times[name].append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                nx.dijkstra_path(bench_graph, sorted(bench_graph.nodes(), key=str)[0], sorted(bench_graph.nodes(), key=str)[-1], weight="weight")
                times["Dijkstra"].append(time.perf_counter() - t0)
            bench_rows.append({"Routers": size, "Links": bench_graph.number_of_edges(),
                               **{f"{name} median ms": round(float(np.median(values)) * 1000, 5) for name, values in times.items()}})
            progress.progress((idx + 1) / len(sizes), text=f"Benchmarked {size} routers")
        st.session_state.benchmark_df = pd.DataFrame(bench_rows)
        progress.empty()
    bench_df = st.session_state.benchmark_df
    if bench_df is not None:
        st.dataframe(bench_df, hide_index=True, use_container_width=True)
        plot_df = bench_df.set_index("Routers")[["Kruskal median ms", "Prim median ms", "Borůvka median ms", "Reverse-Delete median ms", "Dijkstra median ms"]]
        st.line_chart(plot_df, use_container_width=True)
        st.download_button("⬇️ Download benchmark results", data=bench_df.to_csv(index=False).encode("utf-8"),
                           file_name="benchmark_results.csv", mime="text/csv")
        st.caption("Benchmark timings depend on hardware, Python version, graph density, and background processes. Compare runs on the same machine; do not treat tiny timing differences as universal rankings.")
    else:
        st.info("Choose the largest graph size and number of repetitions, then run the benchmark suite.")

with tab_insights:
    st.subheader("Network resilience and exports")
    component_count = nx.number_connected_components(G) if node_count else 0
    bridges = list(nx.bridges(G)) if node_count and edge_count else []
    articulation = list(nx.articulation_points(G)) if node_count else []
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Connected components", component_count)
    i2.metric("Bridge links", len(bridges))
    i3.metric("Articulation routers", len(articulation))
    i4.metric("Average clustering", f"{nx.average_clustering(G):.3f}" if node_count > 1 else "—")
    if connected(G):
        st.success("All routers can currently reach one another.")
    else:
        st.warning("Some routers cannot reach others. This must be corrected before a single MST can span the full network.")
    st.markdown("#### Resilience interpretation")
    st.write("**Bridge links** are single connections whose failure splits a connected network. **Articulation routers** are routers whose removal splits the network. Fewer critical bridges/articulation points generally indicates more alternate connectivity, though real reliability also depends on link failure probabilities and capacity.")
    if bridges:
        st.write("**Bridge links:** " + ", ".join(f"{u}—{v}" for u, v in bridges))
    else:
        st.write("**Bridge links:** None detected." if connected(G) else "**Bridge links:** Not interpreted until the network is connected.")
    if articulation:
        st.write("**Articulation routers:** " + ", ".join(map(str, sorted(articulation, key=str))))
    else:
        st.write("**Articulation routers:** None detected." if connected(G) else "**Articulation routers:** Not interpreted until the network is connected.")

    with st.expander("Simulate a single-link failure without changing the saved topology"):
        available_edges = list(G.edges())
        if available_edges:
            edge_labels = [f"{u} — {v} (cost {G[u][v].get('weight', 1)})" for u, v in available_edges]
            edge_idx = st.selectbox("Link to simulate as failed", range(len(available_edges)), format_func=lambda idx: edge_labels[idx])
            if st.button("Run failure simulation", key="run_failure_sim"):
                failure_graph = G.copy()
                failed_edge = available_edges[edge_idx]
                failure_graph.remove_edge(*failed_edge)
                comps = list(nx.connected_components(failure_graph))
                st.write(f"After removing **{failed_edge[0]}—{failed_edge[1]}**, component count becomes **{len(comps)}**.")
                if len(comps) == 1:
                    st.success("The network remains connected; another route exists between all routers.")
                else:
                    st.error("This link failure disconnects the network. Consider adding redundant links.")
                draw_and_close(failure_graph, "Topology after simulated link failure")
        else:
            st.info("Add at least one link to simulate a failure.")

    st.markdown("#### Budget feasibility (optional scenario input)")
    required_cost = None
    if connected(G):
        _, required_cost = mst_kruskal(G)
    budget_default = int(required_cost or 0)
    budget = st.number_input("Available deployment budget (same unit as link costs)", min_value=0, value=budget_default, step=10)
    if required_cost is not None:
        if budget >= required_cost:
            st.success(f"Budget feasible for the minimum-cost connected topology. Minimum required: {required_cost}; remaining: {budget - required_cost}.")
        else:
            st.error(f"Budget is below the MST cost by {required_cost - budget}. Add budget or revise the link-cost assumptions.")
    else:
        st.info("Reconnect the graph to estimate its minimum spanning deployment cost.")

    st.markdown("#### Export this network")
    current_edges_export = edge_frame(G)
    st.download_button("⬇️ Export edge list (CSV)", data=current_edges_export.to_csv(index=False).encode("utf-8"),
                       file_name="network_topology.csv", mime="text/csv", key="export_edges_insights")
    payload = {
        "project": "Smart Network Optimization",
        "directed": False,
        "nodes": [{"id": str(node)} for node in nodes],
        "edges": [{"source": str(u), "target": str(v), "cost": int(data.get("weight", 1))}
                  for u, v, data in G.edges(data=True)],
        "metrics": {"routers": node_count, "links": edge_count, "density": density, "all_link_cost": weight_sum,
                    "connected_components": component_count},
    }
    st.download_button("⬇️ Export network + metrics (JSON)", data=json.dumps(payload, indent=2).encode("utf-8"),
                       file_name="network_topology.json", mime="application/json")
    st.markdown("---")
    with st.expander("Quick guide: which algorithm should I choose?"):
        st.markdown("""
        - **Kruskal:** sorts links by cost and joins components; useful for edge-list demonstrations.
        - **Prim:** grows one tree by repeatedly taking the cheapest link leaving the current tree.
        - **Borůvka / Sollin:** each component selects its cheapest outgoing link in rounds; these are two names for the same classic method.
        - **Reverse-Delete:** considers links from most expensive to least expensive and removes a link when the remaining network stays connected. It is educational but often slower.
        - **Dijkstra:** minimizes total link cost when all link costs are non-negative.
        - **Bellman–Ford:** weighted shortest paths and can support negative edges, provided no reachable negative cycle exists (this dashboard's editor intentionally keeps costs positive).
        - **BFS:** minimizes the number of hops, not the sum of link costs.
        - **Floyd–Warshall:** all-pairs weighted shortest-path costs; best for smaller networks because its time and memory requirements grow quickly.
        """)

st.markdown("<p class='small-note' style='text-align:center;margin-top:2rem'>Smart Network Optimization · Interactive DAA simulator · Synthetic link costs unless you import calibrated data</p>", unsafe_allow_html=True)

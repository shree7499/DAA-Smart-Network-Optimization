import streamlit as st
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

from graph.generator import generate_network
from algorithms.kruskal import kruskal_mst
from algorithms.prim import prim_mst
from algorithms.dijkstra import dijkstra
from visualization.visualizer import draw_network


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Network Optimization",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(30, 80, 150, 0.20),
                transparent 30%
            ),
            radial-gradient(
                circle at top right,
                rgba(0, 180, 200, 0.12),
                transparent 25%
            ),
            #08111f;
        color: #f5f7fa;
    }


    /* Sidebar */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #07101d 0%,
                #0b1728 100%
            );
        border-right: 1px solid rgba(255,255,255,0.08);
    }


    /* Sidebar text */

    section[data-testid="stSidebar"] * {
        color: #e7edf5;
    }


    /* Main title */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0;
        background:
            linear-gradient(
                90deg,
                #ffffff,
                #72d8ff,
                #7c9cff
            );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }


    .subtitle {
        color: #91a4bb;
        font-size: 16px;
        margin-top: 5px;
        margin-bottom: 25px;
    }


    /* Cards */

    .metric-card {
        background:
            linear-gradient(
                145deg,
                rgba(20,35,55,0.96),
                rgba(10,22,38,0.96)
            );

        border: 1px solid rgba(120,190,255,0.13);

        border-radius: 16px;

        padding: 20px;

        min-height: 125px;

        box-shadow:
            0 10px 30px rgba(0,0,0,0.20);

        transition: 0.2s;
    }


    .metric-card:hover {
        border-color: rgba(100,210,255,0.35);
        transform: translateY(-2px);
    }


    .metric-label {
        color: #91a4bb;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }


    .metric-value {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }


    .metric-sub {
        color: #68d391;
        font-size: 13px;
        margin-top: 5px;
    }


    /* Section headers */

    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: #ffffff;
        margin-top: 25px;
        margin-bottom: 5px;
    }


    .section-description {
        color: #8ea2b9;
        margin-bottom: 18px;
    }


    /* Algorithm cards */

    .algorithm-card {
        background:
            linear-gradient(
                145deg,
                rgba(16,31,50,0.95),
                rgba(8,18,32,0.95)
            );

        border: 1px solid rgba(255,255,255,0.08);

        border-radius: 16px;

        padding: 22px;

        min-height: 190px;
    }


    .algorithm-name {
        font-size: 21px;
        font-weight: 750;
        color: #ffffff;
    }


    .algorithm-type {
        color: #66d9ff;
        font-size: 13px;
        margin: 6px 0 15px;
    }


    .algorithm-description {
        color: #9caec2;
        font-size: 14px;
        line-height: 1.6;
    }


    /* Status badge */

    .status {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background: rgba(70,200,130,0.12);
        color: #67e8a0;
        border: 1px solid rgba(70,200,130,0.2);
        font-size: 12px;
        font-weight: 600;
    }


    /* Info box */

    .info-box {
        background: rgba(25,45,70,0.55);
        border-left: 4px solid #58c7ff;
        padding: 15px 18px;
        border-radius: 8px;
        color: #b8c7d9;
        line-height: 1.6;
    }


    /* Footer */

    .footer {
        text-align: center;
        color: #647890;
        padding: 30px 0 10px;
        font-size: 13px;
    }


    /* Tabs */

    button[data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 600;
    }


    /* Dataframe */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "graph" not in st.session_state:
    st.session_state.graph = None

if "mst_edges" not in st.session_state:
    st.session_state.mst_edges = None

if "mst_cost" not in st.session_state:
    st.session_state.mst_cost = None

if "mst_algorithm" not in st.session_state:
    st.session_state.mst_algorithm = None

if "shortest_path" not in st.session_state:
    st.session_state.shortest_path = None

if "shortest_distance" not in st.session_state:
    st.session_state.shortest_distance = None

if "source" not in st.session_state:
    st.session_state.source = None

if "destination" not in st.session_state:
    st.session_state.destination = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌐 Smart Network Optimization</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-inspired graph optimization simulator using MST and shortest-path algorithms'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<span class="status">● SYSTEM READY</span>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Network Configuration")

    st.markdown("---")

    num_nodes = st.slider(
        "Number of Routers",
        min_value=5,
        max_value=100,
        value=10,
        step=1
    )

    extra_edges = st.slider(
        "Additional Connections",
        min_value=0,
        max_value=150,
        value=10,
        step=1
    )

    st.markdown("### ⚖️ Link Cost Range")

    min_weight = st.number_input(
        "Minimum Cost",
        min_value=1,
        max_value=100,
        value=1
    )

    max_weight = st.number_input(
        "Maximum Cost",
        min_value=2,
        max_value=500,
        value=50
    )

    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=999999,
        value=42
    )

    st.markdown("---")

    generate_button = st.button(
        "🚀 Generate Network",
        use_container_width=True,
        type="primary"
    )

    reset_button = st.button(
        "🔄 Reset Analysis",
        use_container_width=True
    )

    st.markdown("---")

    st.markdown("### 📌 Project")

    st.caption(
        "Smart Network Optimization Using "
        "Minimum Spanning Tree and Shortest Path Algorithms"
    )

    st.markdown("---")

    st.markdown("### 🧠 Algorithms")

    st.caption("Kruskal — Minimum Spanning Tree")

    st.caption("Prim — Minimum Spanning Tree")

    st.caption("Dijkstra — Shortest Path")


# ============================================================
# RESET
# ============================================================

if reset_button:

    st.session_state.graph = None
    st.session_state.mst_edges = None
    st.session_state.mst_cost = None
    st.session_state.mst_algorithm = None
    st.session_state.shortest_path = None
    st.session_state.shortest_distance = None

    st.rerun()


# ============================================================
# GENERATE NETWORK
# ============================================================

if generate_button:

    if min_weight >= max_weight:

        st.error(
            "Minimum cost must be smaller than maximum cost."
        )

    else:

        st.session_state.graph = generate_network(
            num_nodes=num_nodes,
            extra_edges=extra_edges,
            min_weight=min_weight,
            max_weight=max_weight,
            seed=seed
        )

        st.session_state.mst_edges = None
        st.session_state.mst_cost = None
        st.session_state.mst_algorithm = None
        st.session_state.shortest_path = None
        st.session_state.shortest_distance = None

        st.success(
            "Network generated successfully."
        )


# ============================================================
# IF NO GRAPH
# ============================================================

if st.session_state.graph is None:

    st.markdown(
        """
        <div class="info-box">
        <b>Welcome to Smart Network Optimization.</b><br><br>
        Configure the network from the sidebar and click
        <b>Generate Network</b> to start the simulation.
        <br><br>
        The system will generate routers, weighted communication
        links, minimum spanning trees and shortest communication paths.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## 🔍 What this simulator demonstrates")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="algorithm-card">

            <div class="algorithm-name">
            🌳 Minimum Spanning Tree
            </div>

            <div class="algorithm-type">
            Network Cost Optimization
            </div>

            <div class="algorithm-description">
            Kruskal and Prim algorithms determine a
            minimum-cost topology connecting all routers.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="algorithm-card">

            <div class="algorithm-name">
            🛣️ Shortest Path
            </div>

            <div class="algorithm-type">
            Efficient Routing
            </div>

            <div class="algorithm-description">
            Dijkstra's algorithm determines the minimum-cost
            communication route between two routers.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="algorithm-card">

            <div class="algorithm-name">
            📈 Performance Analysis
            </div>

            <div class="algorithm-type">
            DAA Complexity Study
            </div>

            <div class="algorithm-description">
            Execution time is measured across increasing
            network sizes to study algorithm performance.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.stop()


# ============================================================
# GRAPH VARIABLES
# ============================================================

graph = st.session_state.graph

nodes = list(graph.nodes())

number_of_nodes = graph.number_of_nodes()

number_of_edges = graph.number_of_edges()

original_cost = sum(
    data["weight"]
    for _, _, data in graph.edges(data=True)
)

density = nx.density(graph)


# ============================================================
# DASHBOARD KPIs
# ============================================================

st.markdown(
    '<div class="section-title">📊 Network Overview</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Live statistics for the generated network topology.'
    '</div>',
    unsafe_allow_html=True
)

k1, k2, k3, k4, k5 = st.columns(5)


with k1:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-label">Routers</div>
        <div class="metric-value">{number_of_nodes}</div>
        <div class="metric-sub">Network Nodes</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with k2:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-label">Connections</div>
        <div class="metric-value">{number_of_edges}</div>
        <div class="metric-sub">Communication Links</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with k3:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-label">Network Cost</div>
        <div class="metric-value">{original_cost}</div>
        <div class="metric-sub">Original Cost</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with k4:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-label">Density</div>
        <div class="metric-value">{density:.2f}</div>
        <div class="metric-sub">Connectivity Level</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with k5:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-label">Avg. Link Cost</div>
        <div class="metric-value">
        {original_cost / number_of_edges:.1f}
        </div>
        <div class="metric-sub">Weighted Average</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🌐 Network Simulation",
        "🌳 MST Optimization",
        "🛣️ Shortest Path",
        "📈 Performance Analysis"
    ]
)


# ============================================================
# TAB 1 — NETWORK
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">🌐 Network Topology</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Generated weighted communication network.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([2.2, 1])

    with col1:

        fig = draw_network(
            graph,
            title="Generated Network Topology"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    with col2:

        st.markdown("### 📋 Network Information")

        info_df = pd.DataFrame(
            {
                "Property": [
                    "Routers",
                    "Connections",
                    "Total Cost",
                    "Density",
                    "Minimum Link Cost",
                    "Maximum Link Cost"
                ],
                "Value": [
                    number_of_nodes,
                    number_of_edges,
                    original_cost,
                    f"{density:.3f}",
                    min(
                        data["weight"]
                        for _, _, data in graph.edges(data=True)
                    ),
                    max(
                        data["weight"]
                        for _, _, data in graph.edges(data=True)
                    )
                ]
            }
        )

        st.dataframe(
            info_df,
            hide_index=True,
            use_container_width=True
        )

        st.markdown("### 🔗 Network Links")

        edge_data = []

        for u, v, data in graph.edges(data=True):

            edge_data.append(
                {
                    "Router A": u,
                    "Router B": v,
                    "Cost": data["weight"]
                }
            )

        edge_df = pd.DataFrame(edge_data)

        st.dataframe(
            edge_df,
            hide_index=True,
            use_container_width=True,
            height=300
        )


# ============================================================
# TAB 2 — MST
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">🌳 Minimum Spanning Tree Optimization</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Compare Kruskal and Prim to construct the minimum-cost network.'
        '</div>',
        unsafe_allow_html=True
    )

    b1, b2 = st.columns(2)

    with b1:

        kruskal_button = st.button(
            "⚡ Run Kruskal",
            use_container_width=True
        )

    with b2:

        prim_button = st.button(
            "🌲 Run Prim",
            use_container_width=True
        )


    if kruskal_button:

        with st.spinner("Running Kruskal's algorithm..."):

            mst_edges, mst_cost, execution_time = kruskal_mst(
                graph
            )

        st.session_state.mst_edges = mst_edges
        st.session_state.mst_cost = mst_cost
        st.session_state.mst_algorithm = "Kruskal"

        st.success(
            f"Kruskal completed in {execution_time:.8f} seconds."
        )


    if prim_button:

        with st.spinner("Running Prim's algorithm..."):

            mst_edges, mst_cost, execution_time = prim_mst(
                graph
            )

        st.session_state.mst_edges = mst_edges
        st.session_state.mst_cost = mst_cost
        st.session_state.mst_algorithm = "Prim"

        st.success(
            f"Prim completed in {execution_time:.8f} seconds."
        )


    if st.session_state.mst_edges is not None:

        mst_edges = st.session_state.mst_edges

        mst_cost = st.session_state.mst_cost

        savings = original_cost - mst_cost

        savings_percentage = (
            savings / original_cost * 100
            if original_cost > 0
            else 0
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:

            st.metric(
                "Optimized Cost",
                mst_cost
            )

        with m2:

            st.metric(
                "Cost Saved",
                savings
            )

        with m3:

            st.metric(
                "Savings",
                f"{savings_percentage:.2f}%"
            )

        with m4:

            st.metric(
                "Algorithm",
                st.session_state.mst_algorithm
            )


        st.markdown("### 📊 Optimized Network")

        mst_fig = draw_network(
            graph,
            highlighted_edges=mst_edges,
            title=(
                f"{st.session_state.mst_algorithm} "
                "Minimum Spanning Tree"
            )
        )

        st.pyplot(
            mst_fig,
            use_container_width=True
        )

        plt.close(mst_fig)


        st.markdown("### 🔗 Selected MST Connections")

        mst_df = pd.DataFrame(
            mst_edges,
            columns=[
                "Router A",
                "Router B",
                "Link Cost"
            ]
        )

        st.dataframe(
            mst_df,
            hide_index=True,
            use_container_width=True
        )


        csv_data = mst_df.to_csv(
            index=False
        )

        st.download_button(
            "📥 Download MST Results",
            data=csv_data,
            file_name="mst_results.csv",
            mime="text/csv"
        )


    else:

        st.markdown(
            """
            <div class="info-box">
            Select <b>Run Kruskal</b> or <b>Run Prim</b> to
            optimize the network topology.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# TAB 3 — DIJKSTRA
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">🛣️ Shortest Path Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Find the minimum-cost communication route between routers.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        source = st.selectbox(
            "📡 Source Router",
            nodes,
            index=0
        )

    with c2:

        destination_options = [
            node for node in nodes
            if node != source
        ]

        destination = st.selectbox(
            "🎯 Destination Router",
            destination_options
        )


    if st.button(
        "🚀 Find Shortest Path",
        use_container_width=True,
        type="primary"
    ):

        if source == destination:

            st.error(
                "Source and destination must be different."
            )

        else:

            with st.spinner(
                "Calculating shortest path..."
            ):

                path, distance, execution_time = dijkstra(
                    graph,
                    source,
                    destination
                )

            st.session_state.shortest_path = path

            st.session_state.shortest_distance = distance

            st.session_state.source = source

            st.session_state.destination = destination

            st.success(
                f"Dijkstra completed in "
                f"{execution_time:.8f} seconds."
            )


    if (
        st.session_state.shortest_path is not None
        and st.session_state.shortest_distance is not None
    ):

        path = st.session_state.shortest_path

        distance = st.session_state.shortest_distance


        r1, r2, r3, r4 = st.columns(4)

        with r1:

            st.metric(
                "Source",
                st.session_state.source
            )

        with r2:

            st.metric(
                "Destination",
                st.session_state.destination
            )

        with r3:

            st.metric(
                "Path Cost",
                distance
            )

        with r4:

            st.metric(
                "Hop Count",
                len(path) - 1
            )


        st.markdown("### 🛣️ Optimal Communication Route")

        route_text = "  →  ".join(path)

        st.markdown(
            f"""
            <div class="info-box">
            <b>Optimal Route:</b><br><br>
            {route_text}
            <br><br>
            <b>Total Cost:</b> {distance}
            </div>
            """,
            unsafe_allow_html=True
        )


        path_fig = draw_network(
            graph,
            shortest_path=path,
            title="Dijkstra Shortest Path"
        )

        st.pyplot(
            path_fig,
            use_container_width=True
        )

        plt.close(path_fig)


# ============================================================
# TAB 4 — PERFORMANCE
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-title">📈 Algorithm Performance Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Experimental execution-time analysis for increasing network sizes.'
        '</div>',
        unsafe_allow_html=True
    )


    results_file = (
        Path(__file__).resolve().parent
        / "results"
        / "performance_results.csv"
    )


    if results_file.exists():

        performance_df = pd.read_csv(
            results_file
        )

        st.markdown("### ⏱️ Execution Time")

        fig, ax = plt.subplots(
            figsize=(11, 5)
        )

        ax.plot(
            performance_df["Nodes"],
            performance_df["Kruskal Time"],
            marker="o",
            linewidth=2,
            label="Kruskal"
        )

        ax.plot(
            performance_df["Nodes"],
            performance_df["Prim Time"],
            marker="o",
            linewidth=2,
            label="Prim"
        )

        ax.plot(
            performance_df["Nodes"],
            performance_df["Dijkstra Time"],
            marker="o",
            linewidth=2,
            label="Dijkstra"
        )

        ax.set_xlabel(
            "Number of Routers"
        )

        ax.set_ylabel(
            "Execution Time (seconds)"
        )

        ax.set_title(
            "Graph Algorithm Performance"
        )

        ax.grid(
            True,
            alpha=0.25
        )

        ax.legend()

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


        st.markdown("### 📋 Experimental Results")

        st.dataframe(
            performance_df,
            hide_index=True,
            use_container_width=True
        )


        st.markdown("### 🧠 Complexity Analysis")

        complexity_df = pd.DataFrame(
            {
                "Algorithm": [
                    "Kruskal",
                    "Prim",
                    "Dijkstra"
                ],
                "Purpose": [
                    "Minimum Spanning Tree",
                    "Minimum Spanning Tree",
                    "Shortest Path"
                ],
                "Time Complexity": [
                    "O(E log E)",
                    "O(E log V)",
                    "O((V + E) log V)"
                ],
                "Strategy": [
                    "Greedy",
                    "Greedy",
                    "Greedy"
                ]
            }
        )

        st.dataframe(
            complexity_df,
            hide_index=True,
            use_container_width=True
        )


        st.markdown("### 📊 Performance Interpretation")

        st.markdown(
            """
            <div class="info-box">

            As the number of routers and communication links increases,
            the execution time of the graph algorithms also increases.

            <br><br>

            This experiment demonstrates the relationship between
            <b>network size</b> and <b>algorithm execution time</b>.

            <br><br>

            The measured results depend on graph structure,
            implementation details and the computer hardware used
            during the experiment.

            </div>
            """,
            unsafe_allow_html=True
        )


        performance_csv = performance_df.to_csv(
            index=False
        )

        st.download_button(
            "📥 Download Performance Results",
            data=performance_csv,
            file_name="performance_results.csv",
            mime="text/csv"
        )


    else:

        st.warning(
            "performance_results.csv was not found."
        )

        st.info(
            "Run: python -m results.performance_test"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    Smart Network Optimization System • DAA Project • Python + NetworkX + Streamlit

    <br>

    Kruskal • Prim • Dijkstra • Graph Optimization • Performance Analysis

    </div>
    """,
    unsafe_allow_html=True
)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from sklearn.cluster import KMeans

st.set_page_config(page_title="Supply Chain System Visualization", layout="wide")

st.title("System-Level Supply Chain Visualization")
st.caption("AI-assisted, physics-informed visualization for managerial understanding")

# -----------------------------
# Synthetic Data (Conceptual)
# -----------------------------
np.random.seed(42)

nodes = pd.DataFrame({
    "Node": [f"Supplier {i}" for i in range(1, 7)]
            + [f"Warehouse {i}" for i in range(1, 4)]
            + [f"Distributor {i}" for i in range(1, 4)],
    "Type": ["Supplier"] * 6 + ["Warehouse"] * 3 + ["Distributor"] * 3,
    "Risk": np.random.randint(2, 10, 12),
    "Delay": np.random.randint(1, 15, 12),
    "Inventory": np.random.randint(100, 500, 12)
})

edges = [
    ("Supplier 1", "Warehouse 1"),
    ("Supplier 2", "Warehouse 1"),
    ("Supplier 3", "Warehouse 2"),
    ("Supplier 4", "Warehouse 2"),
    ("Supplier 5", "Warehouse 3"),
    ("Supplier 6", "Warehouse 3"),
    ("Warehouse 1", "Distributor 1"),
    ("Warehouse 2", "Distributor 2"),
    ("Warehouse 3", "Distributor 3"),
]

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Visualization Controls")

view_mode = st.sidebar.radio(
    "Supply Chain View",
    ["Blind (Traditional Dashboard)", "Visible (System View)"]
)

layer = st.sidebar.selectbox(
    "Primary Visual Layer",
    ["Risk", "Delay", "Inventory"]
)

stress_level = st.sidebar.slider(
    "System Stress Level",
    0, 3, 0,
    help="Simulates disruption stress to reveal bottlenecks"
)

# Apply stress visually (no prediction)
nodes["Adjusted Delay"] = nodes["Delay"] + stress_level * 3

# -----------------------------
# BLIND VIEW
# -----------------------------
if view_mode == "Blind (Traditional Dashboard)":
    st.subheader("Fragmented Dashboard View")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(
            nodes,
            x="Node",
            y="Risk",
            title="Risk Metrics"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(
            nodes,
            x="Node",
            y="Adjusted Delay",
            title="Delay Metrics"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "This view presents isolated metrics without showing interdependencies "
        "or system-wide ripple effects."
    )

# -----------------------------
# VISIBLE SYSTEM VIEW
# -----------------------------
else:
    st.subheader("Integrated System View")

    G = nx.DiGraph()
    for _, row in nodes.iterrows():
        G.add_node(
            row["Node"],
            risk=row["Risk"],
            delay=row["Adjusted Delay"],
            inventory=row["Inventory"]
        )

    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)

    # Edge drawing
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="gray"),
        hoverinfo="none"
    )

    # Node drawing
    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        if layer == "Risk":
            node_color.append(G.nodes[node]["risk"])
            node_size.append(20)
        elif layer == "Delay":
            node_color.append(G.nodes[node]["delay"])
            node_size.append(20)
        else:
            node_color.append(G.nodes[node]["inventory"])
            node_size.append(G.nodes[node]["inventory"] / 25)

        node_text.append(
            f"{node}<br>"
            f"Risk: {G.nodes[node]['risk']}<br>"
            f"Delay: {G.nodes[node]['delay']}<br>"
            f"Inventory: {G.nodes[node]['inventory']}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[n.split()[0] for n in G.nodes()],
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Reds",
            showscale=True
        )
    )

    fig_net = go.Figure(data=[edge_trace, node_trace])
    fig_net.update_layout(showlegend=False)
    st.plotly_chart(fig_net, use_container_width=True)

    # -----------------------------
    # AI-Assisted Pattern Highlight
    # -----------------------------
    st.subheader("AI-Assisted Pattern Highlighting")

    X = nodes[["Risk", "Adjusted Delay"]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    nodes["Cluster"] = kmeans.fit_predict(X)

    fig_cluster = px.scatter(
        nodes,
        x="Risk",
        y="Adjusted Delay",
        color="Cluster",
        size="Inventory",
        hover_name="Node",
        title="Risk and Delay Clustering"
    )

    st.plotly_chart(fig_cluster, use_container_width=True)

    # -----------------------------
    # Managerial Interpretation Panel
    # -----------------------------
    st.subheader("Managerial Interpretation")

    bottlenecks = nodes[nodes["Adjusted Delay"] > nodes["Adjusted Delay"].mean()]
    high_risk = nodes[nodes["Risk"] > nodes["Risk"].mean()]

    st.markdown(
        f"""
        Key observations from the system view:

        - Risk and delays are not evenly distributed across the network
        - {len(high_risk)} nodes show above-average systemic risk
        - {len(bottlenecks)} nodes act as potential bottlenecks under stress
        - Disruptions propagate downstream rather than remaining isolated
        """
    )

# -----------------------------
# Ethics and Transparency
# -----------------------------
with st.expander("Model and Ethics Notes"):
    st.markdown(
        """
        - All data used is synthetic and illustrative
        - AI is used only for clustering and pattern highlighting
        - No automated decisions or recommendations are generated
        - Visualizations are exploratory and support managerial judgment
        """
    )

st.caption(
    "Conceptual prototype aligned with systems thinking and managerial decision support"
)

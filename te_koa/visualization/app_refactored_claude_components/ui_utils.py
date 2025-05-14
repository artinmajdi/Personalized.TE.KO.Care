"""
UI utilities, styling and constants for the TE-KOA dashboard.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import base64

# Define app configuration
APP_TITLE = "TE-KOA Clinical Research Dashboard"
APP_SUBTITLE = "Phenotyping and Heterogeneity of Treatment Effects in Knee Osteoarthritis"
PHASE_TITLE = "Phase I: Data Preparation"

# Define color scheme
COLOR_PALETTE = {
    'primary': '#4472C4',  # Blue
    'secondary': '#ED7D31',  # Orange
    'tertiary': '#A5A5A5',  # Gray
    'success': '#70AD47',  # Green
    'warning': '#FFC000',  # Yellow
    'danger': '#FF0000',  # Red
    'highlight': '#5B9BD5',  # Light blue
    'background': '#F5F5F5',  # Light gray
    'text': '#333333'  # Dark gray
}

# Treatment group colors
TREATMENT_COLORS = {
    'Control (Sham)': '#90CAF9',  # Light blue
    'Experimental': '#FF8A65',  # Light orange
    'tDCS': '#81C784',  # Light green
    'Meditation': '#E1BEE7',  # Light purple
    'Control (No tDCS, No Meditation)': '#9FA8DA',  # Indigo
    'tDCS Only': '#A5D6A7',  # Green
    'Meditation Only': '#CE93D8',  # Purple
    'tDCS + Meditation': '#FFAB91'  # Deep orange
}

def apply_custom_css():
    """Apply custom CSS styles to the dashboard."""
    st.markdown("""
    <style>
    .main-header {
        background-color: #4472C4;
        padding: 1rem;
        color: white;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .phase-indicator {
        background-color: #ED7D31;
        padding: 0.3rem 0.6rem;
        color: white;
        border-radius: 3px;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        display: inline-block;
    }
    .section-header {
        background-color: #F5F5F5;
        padding: 0.5rem;
        border-left: 4px solid #4472C4;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
        flex: 1;
        min-width: 200px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4472C4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .stButton>button {
        background-color: #4472C4;
        color: white;
    }
    .sidebar-section {
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

def create_download_link(data, filename, label="Download", file_format="csv"):
    """Create a download link for data."""
    if file_format == "csv":
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
    elif file_format == "json":
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="{filename}">{label}</a>'
    elif file_format == "excel":
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{label}</a>'
    elif file_format == "markdown":
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:text/markdown;base64,{b64}" download="{filename}">{label}</a>'
    else:
        return None

    return href

def plot_correlation_network(corr_matrix, threshold=0.5):
    """Create a network graph visualization from a correlation matrix."""
    # Get variables above correlation threshold
    edges = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                edges.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    abs(corr_matrix.iloc[i, j])
                ))

    if not edges:
        return None

    # Create network graph
    G = nx.Graph()
    for var in corr_matrix.columns:
        G.add_node(var)

    for source, target, weight in edges:
        G.add_edge(source, target, weight=weight)

    # Get positions using a layout algorithm
    pos = nx.spring_layout(G, seed=42)

    # Create edges trace
    edge_x = []
    edge_y = []
    edge_weights = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2]['weight'])

    # Normalize edge widths
    min_width = 1
    max_width = 10
    if edge_weights:
        normalized_weights = [
            min_width + (w - min(edge_weights)) * (max_width - min_width) / (max(edge_weights) - min(edge_weights))
            if max(edge_weights) > min(edge_weights) else 5
            for w in edge_weights
        ]
    else:
        normalized_weights = []

    # Create nodes trace
    node_x = []
    node_y = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    # Calculate node degree for node size
    node_degrees = dict(G.degree())
    node_sizes = [30 + 10 * node_degrees[node] for node in G.nodes()]

    # Create the figure
    fig = go.Figure()

    # Add edges with varying widths
    edge_segments = np.reshape(np.array(edge_x), (-1, 3)) if edge_x else []
    for i, segment in enumerate(edge_segments):
        if i < len(normalized_weights):
            fig.add_trace(go.Scatter(
                x=segment,
                y=edge_y[i*3:(i+1)*3],
                mode='lines',
                line=dict(width=normalized_weights[i], color='rgba(68, 114, 196, 0.5)'),
                hoverinfo='none'
            ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=COLOR_PALETTE['primary'],
            line=dict(width=2, color='white')
        ),
        text=node_text,
        textposition='top center',
        hoverinfo='text',
        hovertext=[f"{node}<br>Connections: {node_degrees[node]}" for node in G.nodes()]
    ))

    # Update layout
    fig.update_layout(
        title='Correlation Network Graph',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=800,
        height=600
    )

    return fig

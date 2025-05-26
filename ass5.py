import math
import random
import subprocess
import sys

import pandas as pd

try:
    import networkx as nx
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx"])
    import networkx as nx

try:
    import powerlaw
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "powerlaw"])
    import powerlaw

try:
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as spread_plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as spread_plt

try:
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np


def edge_removal_analysis(graph_name, weight_attr, step_num):
    G = nx.read_graphml(f"graphmls/{graph_name}.graphml")
    # Get all unique weights

    weights = [data[weight_attr] for _, _, data in G.edges(data=True)]
    max_weight = max(weights)

    # Create reasonable thresholds
    step = max(1, int(max_weight / step_num))  # Create about 10 thresholds
    thresholds = list(range(0, int(max_weight) + step, step))

    # For each threshold, remove edges with weight <= threshold
    sizes = []
    removed_edges_size = []
    print(thresholds)
    for threshold in thresholds:
        print(threshold)
        # Create a copy of the graph
        G_temp = G.copy()

        # Remove edges below threshold
        edges_to_remove = [(u, v) for u, v, d in G_temp.edges(data=True)
                           if d[weight_attr] <= threshold]
        G_temp.remove_edges_from(edges_to_remove)

        # Find giant component size
        components = list(nx.connected_components(G_temp))
        giant_size = max([len(c) for c in components]) if components else 0
        sizes.append(giant_size)
        removed_edges_size.append(len(edges_to_remove))

    plt.figure()
    plt.plot(thresholds,
             sizes,
             linestyle='-',
             )
    plt.title("Thresh to Giant Component Size")
    plt.xlabel(f"Removal Threshold")
    plt.ylabel(f"Giant Component Size")
    plt.savefig(f"images/{graph_name} edge removal figure.png")

    plt.figure()
    plt.plot(thresholds,
             removed_edges_size,
             linestyle='-',
             )
    plt.title("Thresh to Removed Edges Count")
    plt.xlabel(f"Removal Threshold")
    plt.ylabel(f"Removed edges")
    plt.savefig(f"images/{graph_name} removed edges.png")


def neighbourhood_overlap(graph_name, weight_attr):
    G = nx.read_graphml(f"graphmls/{graph_name}.graphml")

    # Calculate neighborhood overlap (Jaccard similarity) for all edges
    overlaps = []

    for u, v, data in G.edges(data=True):
        # Get neighbors of both nodes
        u_neighbors = set(G.neighbors(u))
        v_neighbors = set(G.neighbors(v))

        # Remove the nodes themselves from neighbor sets
        u_neighbors.discard(v)
        v_neighbors.discard(u)

        # Calculate Jaccard similarity
        union = len(u_neighbors.union(v_neighbors))
        intersection = len(u_neighbors.intersection(v_neighbors))

        # Calculate overlap (avoid division by zero)
        overlap = intersection / union if union > 0 else 0

        # Store result
        overlaps.append({
            'source': u,
            'target': v,
            'weight': data[weight_attr],
            'overlap': overlap
        })
    # Convert to DataFrame and group by weight
    df = pd.DataFrame(overlaps)
    print(df)
    by_weight = df.groupby('weight').agg({
        'overlap': 'mean',
        'source': 'count'  # Count how many edges have each weight
    }).reset_index()
    by_weight.columns = ['weight', 'avg_overlap', 'count']

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Plot average overlap vs weight
    sizes = 20 + (by_weight['count'] / by_weight['count'].max() * 100)
    plt.scatter(by_weight['weight'], by_weight['avg_overlap'], s=sizes, alpha=0.7)

    # Add trend line
    z = np.polyfit(by_weight['weight'], by_weight['avg_overlap'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(min(by_weight['weight']), max(by_weight['weight']), 100)
    plt.plot(x_range, p(x_range), 'r--', alpha=0.7)

    # Add labels and title
    plt.xlabel(f'Edge Weight ({weight_attr})')
    plt.ylabel('Average Neighborhood Overlap')
    plt.title('Neighborhood Overlap as Function of Edge Weight')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set y-axis limits
    plt.ylim(0, 1.0)

    # Add correlation text
    correlation = np.corrcoef(by_weight['weight'], by_weight['avg_overlap'])[0, 1]
    plt.figtext(0.5, 0.01, f"Correlation: {correlation:.2f}", ha="center", fontsize=10)

    plt.tight_layout()

    plt.savefig(f"images/{graph_name} neighbourhood overlap.png")


def add_weights(graph_name):
    G = nx.read_graphml(f"graphmls/{graph_name}.graphml")
    edges = []
    for u, v in G.edges:
        # print(G.nodes[u]["Degree"] + G.nodes[v]["Degree"])
        weight = G.nodes[u]["Degree"] + G.nodes[v]["Degree"]
        edges.append((u, v, weight))
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    nx.write_graphml(G, "graphmls/facebook_weights.graphml")
    return G


weighted_facebook = add_weights("facebook")

steps = 20
# edge_removal_analysis("facebook_pa_model", "weight", steps)
edge_removal_analysis("facebook_weights", "weight", steps)
# edge_removal_analysis("tribes", "weight", steps)

# neighbourhood_overlap("tribes", "weight")
# neighbourhood_overlap("facebook_pa_model", "weight")
neighbourhood_overlap("facebook_weights", "weight")

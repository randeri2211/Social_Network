import math
import random
import subprocess
import sys

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

GROUP = "government"
GROUP_COL = "group"
graph_name = "facebook"
nxG = nx.read_graphml(f"graphmls/{graph_name}.graphml")


def get_initial_seeds(G, k, group=None):
    if group is None:
        raise ValueError("No group in seed")
    node_color = nx.get_node_attributes(G, GROUP_COL)  # or "opinion"
    nodes = list(map(lambda x: x[0], filter(lambda x: x[1] == group, node_color.items())))
    print(f"Seed of type {GROUP} length {len(nodes)}")
    if k > len(nodes):
        return nodes
    return random.sample(list(nodes), k)


def run_ic_homophily(G, seeds, p_homo=1.0, q_hetero=0.0):
    active = set(seeds)  # currently influenced
    newly_active = set(seeds)  # activated in the last round
    node_color = nx.get_node_attributes(G, GROUP_COL)  # or "opinion"
    lengths = []
    while newly_active:
        next_newly_active = set()
        for u in newly_active:
            for v in G.neighbors(u):
                if v in active:
                    continue
                # Use p or q depending on similarity
                if node_color.get(u) == node_color.get(v):
                    prob = p_homo
                else:
                    prob = q_hetero
                if random.random() < prob:
                    next_newly_active.add(v)
        newly_active = next_newly_active
        active.update(newly_active)
        lengths.append(len(active))

    return lengths  # number of influenced nodes


def apply_rlr_regulation(G, rho):
    G = G.copy()
    edges = list(G.edges)
    num_to_rewire = int(rho * len(edges))
    removed = random.sample(edges, num_to_rewire)
    G.remove_edges_from(removed)

    nodes = list(G.nodes)
    added = 0
    while added < num_to_rewire:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v) and u != v:
            G.add_edge(u, v)
            added += 1
    return G


def apply_brr_regulation(G, rho):
    """
    Rewires rho% of edges, but preferentially connects nodes with different 'group' values.
    """
    G = G.copy()
    edges = list(G.edges)
    num_to_rewire = int(rho * len(edges))
    removed = random.sample(edges, num_to_rewire)
    G.remove_edges_from(removed)

    node_color = nx.get_node_attributes(G, GROUP_COL)
    nodes = list(G.nodes)
    added = 0

    while added < num_to_rewire:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v) and u != v:
            # Prefer cross-group links
            if node_color.get(u) != node_color.get(v):
                G.add_edge(u, v)
                added += 1
    return G


def sperad(graph, max_seed_size, title, reps=1):
    active_set_size = []
    seed_sizes = range(max_seed_size)
    step_sizes = []
    spreads = []

    # Get reps data points
    for seed_size in seed_sizes:
        print(f'initial seed size:{seed_size}')
        seeds = get_initial_seeds(graph, seed_size, GROUP)
        temp_spreads = []
        temp_steps = []
        temp_active = []
        for rep in range(reps):
            if seed_size == 0:
                temp_active.append((0, 0))
                temp_steps.append((0, 0))
                temp_spreads.append(([0], [0]))
                continue
            p = 1.0
            q = 1 - p
            spread1 = run_ic_homophily(graph, seeds, p_homo=p, q_hetero=q)

            p = 0.7
            q = 1 - p
            spread2 = run_ic_homophily(graph, seeds, p_homo=p, q_hetero=q)

            temp_active.append((max(spread1), max(spread2)))
            temp_steps.append((len(spread1), len(spread2)))
            temp_spreads.append((spread1, spread2))

        active_set_size.append(temp_active)
        step_sizes.append(temp_steps)
        spreads.append(temp_spreads)

    # Average Steps and Active set sizes
    def avg(lst):
        for i in range(len(lst)):
            temp_val = list(map(lambda x: x / reps, lst[i][0]))
            for j in range(len(lst[i][0])):
                for rep in range(1, reps):
                    temp_val[j] += lst[i][rep][j] / reps
            lst[i] = temp_val
        return lst

    avg(step_sizes)
    avg(active_set_size)

    def average_spreads(spreads):
        averaged = []
        for seed_index in range(len(spreads)):
            rep_spreads_p1 = [rep[0] for rep in spreads[seed_index]]
            rep_spreads_p07 = [rep[1] for rep in spreads[seed_index]]

            def average_time_series(series_list):
                max_len = max(map(len, series_list))
                avg_series = [0.0] * max_len
                for series in series_list:
                    for i in range(max_len):
                        val = series[i] if i < len(series) else series[-1]
                        avg_series[i] += val / len(series_list)
                return avg_series

            avg_p1 = average_time_series(rep_spreads_p1)
            avg_p07 = average_time_series(rep_spreads_p07)
            averaged.append((avg_p1, avg_p07))
        return averaged

    averaged_spreads = average_spreads(spreads)

    return active_set_size, step_sizes, averaged_spreads, title


def draw_graph(*kargs, spread=False):
    def draw(index, x_label, y_label, title):
        plt.figure()
        fig_name = title
        for i in range(len(kargs)):
            graph_data = kargs[i]
            fig_name += f"_{graph_data[-1]}"
            if spread:
                spreads1 = list(map(lambda x: x[0], graph_data[index]))
                spreads07 = list(map(lambda x: x[1], graph_data[index]))
                for j in range(5, min(6, len(spreads1), len(spreads07))):
                    plt.plot(range(len(spreads1[j])),
                             spreads1[j],
                             label=f"p=1.0 {graph_data[-1]} seed size={j}",
                             linestyle='-',
                             linewidth=i + 1,
                             )

                    plt.plot(range(len(spreads07[j])),
                             spreads07[j],
                             label=f"p=0.7 {graph_data[-1]} seed size={j}",
                             linestyle='--',
                             linewidth=i + 1,
                             )
            else:
                plt.plot(range(len(graph_data[index])),
                         list(map(lambda x: x[0], graph_data[index])),
                         label=f"p=1.0 {graph_data[-1]}",
                         linestyle='-',
                         )

                plt.plot(range(len(graph_data[index])),
                         list(map(lambda x: x[1], graph_data[index])),
                         label=f"p=0.7 {graph_data[-1]}",
                         linestyle='--',
                         )

        plt.xlabel(f"{x_label}")
        plt.ylabel(f"{y_label}")
        plt.title(f"{title}")
        plt.legend()
        plt.grid(True)
        print(f"{fig_name}.png")
        plt.savefig(f"images/{fig_name}.png")

    if spread:
        draw(2, "Step", "Affected Nodes", "Spread by Step")
    else:
        draw(0, "Seed Set Size", "Final Spread (Active Nodes)", "Final Spread")
        draw(1, "Seed Set Size", "Step Count", "Steps")


G_RLR025 = apply_rlr_regulation(nxG, rho=0.25)
G_RLR05 = apply_rlr_regulation(nxG, rho=0.5)
G_BRR025 = apply_brr_regulation(nxG, rho=0.25)
G_BRR05 = apply_brr_regulation(nxG, rho=0.5)

seed_size = 20
reps = 30

base_data = sperad(nxG, seed_size, "Base", reps)
rlr025_data = sperad(G_RLR025, seed_size, "RLR 0.25", reps)
rlr05_data = sperad(G_RLR05, seed_size, "RLR 0.5", reps)
brr025_data = sperad(G_BRR025, seed_size, "BRR 0.25", reps)
brr05_data = sperad(G_BRR05, seed_size, "BRR 0.5", reps)
#
data = [
    base_data,
    rlr025_data,
    rlr05_data,
    # brr025_data,
    # brr05_data
]

draw_graph(*data, spread=False)
draw_graph(*data, spread=True)

data = [
    base_data,
    # rlr025_data,
    # rlr05_data,
    brr025_data,
    brr05_data
]

draw_graph(*data, spread=False)
draw_graph(*data, spread=True)

data = [
    # base_data,
    rlr025_data,
    rlr05_data,
    brr025_data,
    brr05_data
]

draw_graph(*data, spread=False)
draw_graph(*data, spread=True)

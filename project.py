import time

import networkx as nx
import matplotlib.pyplot as plt
import csv

NODE_LIMIT = 1000000
SIZE_FACTOR = 1000
CULL = False
CULL_SIZE_FACTOR = 20
HISTOGRAM = False
CENTRALITY = False
RANDOM_GRAPH = False

g = nx.Graph()

start_time = time.time()

# Adding Nodes
nodes = []
page_types = set()
with open("facebook_large/musae_facebook_target.csv", 'r', encoding='UTF-8') as file:
    reader = csv.reader(file)
    fields = next(reader)
    print(f'node fields:{fields}')
    node_count = 0
    for user in reader:
        dct = {}
        for i in range(1, len(fields)):
            if fields[i] == 'page_name':
                dct['title'] = user[i]
            elif fields[i] == 'page_type':
                page_types.add(user[i])
            dct[fields[i]] = user[i]
        node = (int(user[0]), dct)
        nodes.append(node)  # ID
        node_count += 1
        if node_count >= NODE_LIMIT:
            break

page_types = list(page_types)
g.add_nodes_from(nodes)
print(f'Amount of nodes:{g.number_of_nodes()}')
# Adding edges
edges = []
with open("facebook_large/musae_facebook_edges.csv", 'r', encoding='UTF-8') as file:
    reader = csv.reader(file)
    fields = next(reader)
    print(f'edges fields:{fields}')
    for edge in reader:
        e1, e2 = int(edge[0]), int(edge[1])
        if e1 < NODE_LIMIT and e2 < NODE_LIMIT:
            edges.append((e1, e2))

g.add_edges_from(edges)
print(f'Amount of edges:{g.number_of_edges()}')
print(f'Graph Data:')
print(f'Number of connected components is:{nx.number_connected_components(g)}')
gen = nx.connected_components(g)
biggest_connect_component = max(gen, key=len)
print(f'Size of biggest connected component:{len(biggest_connect_component)}')
if CULL:
    print("Switching to biggest connected component as main graph")
    g = g.subgraph(biggest_connect_component)
    print(f'Amount of nodes:{g.number_of_nodes()}')
    print(f'Amount of edges:{g.number_of_edges()}')
    print(f'Graph diameter:{nx.diameter(g)}')
    print(f'Average shortest path:{nx.average_shortest_path_length(g)}')

# Calculate degree average
deg_avg = 0
for node in g.degree:
    deg_avg += node[1]
deg_avg /= g.number_of_nodes()
print(f'Average node degree:{deg_avg}')

# Calculate degree centrality
deg_list = list(sorted(g.degree, key=lambda x: x[1], reverse=True))
max_degree = deg_list[0][1]
deg_norm = list(map(lambda x: (x[0], x[1] / max_degree), deg_list))
print(f'max degree:{max_degree}')
for i in g.nodes:
    g.nodes[i]['size'] = SIZE_FACTOR * g.degree[i] / max_degree
    g.nodes[i]['title'] += f' {format(g.degree[i] / max_degree, ".2f")}'

top_x = 5
print(f'Top {top_x} nodes by degree centrality:')
for i in range(top_x):
    print(deg_norm[i])

# Show Small World property
print(f'Average clustering:{nx.average_clustering(g)}')
if CENTRALITY:
    sorted_closeness = sorted(nx.closeness_centrality(g).items(), key=lambda x: x[1], reverse=True)
    print(f'Closeness centrality:{sorted_closeness[:5]}')

# Calculate degree spread
print(f'Counting degree spread')
deg_spread = [0 for _ in range(max_degree + 1)]
for i in g.degree:
    deg_spread[i[1]] += 1
# print(deg_spread)

n = g.number_of_nodes()
k = int(deg_avg)  # average degree (must be even for Watts-Strogatz)
p = 0.1  # Rewiring probability (adjustable)

if RANDOM_GRAPH:
    print(f'\n--- Random Graph Comparison ---')
    print(f'Generating Watts-Strogatz model with n={n}, k={k}, p={p}...')

    max_try = 30
    for i in range(max_try):
        try:
            random_g = nx.watts_strogatz_graph(n=n, k=k, p=p)
            # Compare clustering coefficient
            random_clustering = nx.average_clustering(random_g)
            print(f'Random graph clustering coefficient: {random_clustering:.4f}')
            # Compare average shortest path
            random_avg_path = nx.average_shortest_path_length(random_g)
            print(f'Random graph average shortest path length: {random_avg_path:.4f}')
            break
        except nx.NetworkXError:
            print("Random graph is not connected, so average shortest path length can't be computed.")

# Show degree histogram
if HISTOGRAM:
    plt.bar(x=[i for i in range(max_degree + 1)], height=deg_spread, label='[i for i in range(max_degree + 1)]')
    plt.show()


finish_time = time.time()
print(f'time taken:{finish_time - start_time}')
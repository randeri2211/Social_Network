import math
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
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

try:
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np


# Load edge info
# edges_df = nx.read_edgelist("stormofswords.csv")
#
# # Load node info
# nodes_df = nx.node("tribes.csv", skiprows=1, names=["id", "weight"])
graph_name = "facebook"
nxG = nx.read_graphml(f"graphmls/{graph_name}.graphml")
# print(nxG.number_of_edges())
# # Erdos Renyi graph
# print("Erdos Renyi graph creation")
# # Same edge probability as the original graph
# p = nxG.number_of_edges() / (nxG.number_of_nodes() * (nxG.number_of_nodes() - 1) / 2)
# print(f"p bigger than 1/n^2 {p > 1 / (nxG.number_of_nodes() ** 2)}")
# print(f"p bigger than 1/n {p > 1 / nxG.number_of_nodes()}")
# print(f"p bigger than log(n)/n {p > math.log2(nxG.number_of_nodes()) / nxG.number_of_nodes()}")
# print(f"log(n)/n is {math.log2(nxG.number_of_nodes()) / nxG.number_of_nodes()}")
# print(f"p is {p}")
#
# erg = nx.erdos_renyi_graph(n=nxG.number_of_nodes(), p=p)
# nx.write_graphml(erg, f"graphmls/{graph_name}_erdos_renyi.graphml")
# print("Wrote Erdos Renyi graph")
# #
# # # Gilbert graph
# gnm = nx.gnm_random_graph(n=nxG.number_of_nodes(), m=nxG.number_of_edges())
# nx.write_graphml(gnm, f"graphmls/{graph_name}_gilbert.graphml")
# print("Wrote Gilbert graph")

# # Configuration Model graph
# deg_seq = list(map(lambda x: x[1], nxG.degree))
# configuration_model = nx.configuration_model(deg_seq)
# configuration_model = nx.Graph(configuration_model)
# nx.write_graphml(configuration_model, f"graphmls/{graph_name}_configuration_model.graphml")
# print("Wrote Configuration Model graph")

# Block Model graph
# block_model = nx.stochastic_block_model()
# nx.write_graphml(block_model, f"graphmls/{graph_name}_block_model.graphml")
# print("Wrote Block Model graph")

# Preferential Attachment Model
pa_model = nx.preferential_attachment(nxG
                                      , nxG.edges
                                      )
edge_list = []
weights = []
# for u,v,p in pa_model:
#     print(f"({u}, {v}) -> {p}")
#     edge_list.append((u,v,p))
#     weights.append(p)
pag = nx.Graph()
pag.add_weighted_edges_from(pa_model)
print(pa_model)
nx.write_graphml(pag, f"graphmls/{graph_name}_pa_model.graphml")
print(pag.number_of_edges())
print("Wrote Preferential Attachment Model graph")

#
# # Power Law degree distribution
# degrees = [d for n, d in nxG.degree()]
#
# # Fit the distribution
# fit = powerlaw.Fit(degrees)
#
# # Plotting
# fig = fit.plot_pdf(color='b', linewidth=2)
# fit.power_law.plot_pdf(color='r', linestyle='--', ax=fig)
# plt.title("Power Law Fit")
# plt.xlabel("Degree")
# plt.ylabel("Probability")
# plt.show()
#
# # Compare to other distributions
# R, p = fit.distribution_compare('power_law', 'lognormal')
# print(f"Loglikelihood ratio: {R}, p-value: {p}")
#
# degree_counts = np.bincount(degrees)
# degree_vals = np.nonzero(degree_counts)[0]
# counts = degree_counts[degree_vals]
#
# plt.loglog(degree_vals, counts, 'bo')
# plt.xlabel("Degree (log)")
# plt.ylabel("Count (log)")
# plt.title("Log-Log Degree Distribution")
# plt.grid(True)
# plt.show()


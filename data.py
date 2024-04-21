import os
import networkx as nx
import pandas as pd
data_dir = os.path.expanduser("cora/cora/")

edgelist = pd.read_csv(os.path.join(data_dir, "cora.cites"), sep='\t', header=None, names=["target", "source"])
edgelist["label"] = "cites"

Gnx = nx.from_pandas_edgelist(edgelist, edge_attr="label")
nx.set_node_attributes(Gnx, "paper", "label")

feature_names = ["w_{}".format(ii) for ii in range(1433)]
column_names =  feature_names + ["subject"]
node_data = pd.read_csv(os.path.join(data_dir, "cora.content"), sep='\t', header=None, names=column_names)

set(node_data["subject"])
{'Case_Based',
 'Genetic_Algorithms',
 'Neural_Networks',
 'Probabilistic_Methods',
 'Reinforcement_Learning',
 'Rule_Learning',
 'Theory'}

closeness_centrality = nx.closeness_centrality(Gnx)
degree_centrality = nx.degree_centrality(Gnx)
betweenness_centrality = nx.betweenness_centrality(Gnx)
eigenvector_centrality = nx.eigenvector_centrality(Gnx)

# Sort nodes based on each centrality measure
sorted_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)
sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
sorted_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)

# Select top nodes based on each centrality measure
top_n = 5  # Replace with the desired number of top nodes

top_close_nodes = [node[0] for node in sorted_closeness[:top_n]]
top_degree_nodes = [node[0] for node in sorted_degree[:top_n]]
top_betweenness_nodes = [node[0] for node in sorted_betweenness[:top_n]]
top_eigenvector_nodes = [node[0] for node in sorted_eigenvector[:top_n]]

print("\nTop {} nodes based on Closeness Centrality:".format(top_n))
print(top_close_nodes)

print("\nTop {} nodes based on Degree Centrality:".format(top_n))
print(top_degree_nodes)

print("\nTop {} nodes based on Betweenness Centrality:".format(top_n))
print(top_betweenness_nodes)

print("\nTop {} nodes based on Eigenvector Centrality:".format(top_n))
print(top_eigenvector_nodes)

top_degree_indices = [list(Gnx.nodes).index(node[0]) for node in sorted_degree[:top_n]]
top_betweenness_indices = [list(Gnx.nodes).index(node[0]) for node in sorted_betweenness[:top_n]]
top_eigenvector_indices = [list(Gnx.nodes).index(node[0]) for node in sorted_eigenvector[:top_n]]
top_closeness_indices = [list(Gnx.nodes).index(node[0]) for node in sorted_closeness[:top_n]]

print("Top {} nodes based on Degree Centrality (indices):".format(top_n))
print(top_degree_indices)

print("\nTop {} nodes based on Betweenness Centrality (indices):".format(top_n))
print(top_betweenness_indices)

print("\nTop {} nodes based on Eigenvector Centrality (indices):".format(top_n))
print(top_eigenvector_indices)

print("\nTop {} nodes based on Closeness Centrality (indices):".format(top_n))
print(top_closeness_indices)

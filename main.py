import networkx
import matplotlib.pyplot as plt

from bn.bayesnet import BayesianNetwork

b = BayesianNetwork(["a", "b", "c", "d", "e"], prob=.5)
G = b.random()

print(networkx.to_pandas_adjacency(G, G.nodes()))
print(networkx.is_directed_acyclic_graph(G))
networkx.draw(G, with_labels=True,)
plt.show()

G = b.random(networkx.to_numpy_array(G))
print(networkx.to_pandas_adjacency(G, G.nodes()))
print(networkx.is_directed_acyclic_graph(G))
networkx.draw(G, with_labels=True,)
plt.show()
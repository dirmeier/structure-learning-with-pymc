import networkx
import matplotlib.pyplot as plt
import pandas as pd

from bn.Variable import Variable
from bn.bayesnet import BayesianNetwork

#b = BayesianNetwork(["a", "b", "c", "d", "e"], prob=.5)

#G = b.as_graph(b.random())
#print(networkx.to_pandas_adjacency(G, G.nodes()))
#print(networkx.is_directed_acyclic_graph(G))
#networkx.draw(G, with_labels=True)
#plt.show()

#G = b.as_graph(b.random(networkx.to_numpy_array(G)))
#print(networkx.to_pandas_adjacency(G, G.nodes()))
#print(networkx.is_directed_acyclic_graph(G))
#networkx.draw(G, with_labels=True)
#plt.show()
#b.posterior_sample(networkx.to_numpy_array(G))


difficulty = pd.DataFrame(
  {"difficulty": ["easy", "hard"],
   "probability": [0.6, 0.4]})
difficulty = Variable()

has_studied = pd.DataFrame(
  {"has_studied": ["no", "yes"],
   "probability": [0.7, 0.3]})


sat = pd.DataFrame(
  {"has_studied": ["no", "no", "yes", "yes"],
   "sat": ["no", "yes", "no", "yes"],
   "probability": [0.95, 0.05, 0.2, 0.8]})

letter = pd.DataFrame(
  {"has_studied": ["no", "no", "yes", "yes"],
   "sat": ["no", "yes", "no", "yes"],
   "probability": [0.95, 0.05, 0.2, 0.8]})

grade = pd.DataFrame(
  {"difficulty": ["easy", "easy", "easy", "hard", "hard", "hard", "easy", "easy", "easy",  "hard", "hard", "hard"],
   "has_studied": ["no", "no", "no", "no", "no", "no", "yes", "yes", "yes", "yes", "yes", "yes"],
   "grade": ["good", "ok", "bad", "good", "ok", "bad", "good", "ok", "bad", "good", "ok", "bad"],
   "probability": [0.3, 0.4, 0.3, 0.05, 0.25, 0.7, 0.9, 0.08, 0.02, 0.5, 0.3, 0.2]})

import numpy

import networkx
import matplotlib.pyplot as plt
import pandas as pd

from bn.bayesian_network import BayesianNetwork
from bn.structure_distribution import Structure
from bn.variable import Variable


# b = BayesianNetwork(["a", "b", "c", "d", "e"], prob=.5)

# G = b.as_graph(b.random())
# print(networkx.to_pandas_adjacency(G, G.nodes()))
# print(networkx.is_directed_acyclic_graph(G))
# networkx.draw(G, with_labels=True)
# plt.show()

# G = b.as_graph(b.random(networkx.to_numpy_array(G)))
# print(networkx.to_pandas_adjacency(G, G.nodes()))
# print(networkx.is_directed_acyclic_graph(G))
# networkx.draw(G, with_labels=True)
# plt.show()
# b.posterior_sample(networkx.to_numpy_array(G))


difficulty = Variable(
  "difficulty",
  ["easy", "hard"],
  pd.DataFrame(
    {"difficulty": ["easy", "hard"],
     "probability": [0.6, 0.4]})
)

has_studied = Variable(
  "has_studied",
  ["no", "yes"],
  pd.DataFrame(
    {"has_studied": ["no", "yes"],
     "probability": [0.7, 0.3]})
)

sat = Variable(
  "sat",
  ["low", "high"],
  pd.DataFrame(
    {"has_studied": ["no", "no", "yes", "yes"],
     "sat": ["low", "high", "low", "high"],
     "probability": [0.95, 0.05, 0.2, 0.8]})
)

letter = Variable(
  "letter",
  ["weak", "strong"],
  pd.DataFrame(
    {"grade": ["good", "good", "ok", "ok", "bad", "bad"],
     "letter": ["weak", "strong", "weak", "strong", "weak", "strong"],
     "probability": [0.1, 0.9, 0.4, 0.6, 0.99, 0.01]})
)

grade = Variable(
  "grade",
  ["good", "ok", "bad"],
  pd.DataFrame(
    {"difficulty": ["easy", "easy", "easy", "hard", "hard", "hard", "easy",
                    "easy", "easy", "hard", "hard", "hard"],
     "has_studied": ["no", "no", "no", "no", "no", "no", "yes", "yes", "yes",
                     "yes", "yes", "yes"],
     "grade": ["good", "ok", "bad", "good", "ok", "bad", "good", "ok", "bad",
               "good", "ok", "bad"],
     "probability": [0.3, 0.4, 0.3, 0.05, 0.25, 0.7, 0.9, 0.08, 0.02, 0.5, 0.3,
                     0.2]})
)

adj = numpy.array(
  [
      [0, 1, 0, 0, 0],
      [0, 0, 0, 1, 0],
      [0, 1, 0, 0, 1],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]
  ], dtype=numpy.int8
)

#print(difficulty.lpd)

bn = BayesianNetwork([difficulty, grade, has_studied, letter, sat], adj)
data = bn.sample_data(100)

numpy.random.seed(23)
adj = numpy.zeros_like(adj, dtype=numpy.int8)
distr = Structure([difficulty, grade, has_studied, letter, sat])
best = adj
best_score = -numpy.Inf
for i in range(1000):
    adj, score = distr.posterior_sample(data, adj)
    if best_score < score:
        best = adj
        best_score = score

G = bn.as_graph(adj)
print(networkx.is_directed_acyclic_graph(G))
networkx.draw(G, with_labels=True)
plt.show()


#print(bn.sample_data(1))

#networkx.draw(G, with_labels=True)
#plt.show()

#print(grade.lpd)
#print("------------")
#print(difficulty.lpd)
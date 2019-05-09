import numpy as np
import pydotplus
from sklearn.tree import DecisionTreeRegressor, export_graphviz

X = np.array([5, 7, 12, 23, 25, 28, 29, 34, 35, 40])
y = np.array([82, 80, 103, 118, 172, 127, 204, 189, 99, 166])

F = np.mean(y)
h = 0

iterations = 10

regression = DecisionTreeRegressor(max_depth=1)
regression.fit(X.reshape(-1, 1), (y - F).reshape(-1, 1))

for i in range(iterations):
    h = regression.predict(X.reshape(-1, 1))
    F = F + h
    regression = DecisionTreeRegressor(max_depth=1)
    regression.fit(X.reshape(-1, 1), (y - F).reshape(-1, 1))

dot_data = export_graphviz(regression, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("simple_reg_tree_step_1.png")

plt.plot(X, F)


import pandas as pd
import pydotplus
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, export_graphviz

data = pd.read_csv(
    "tennis.txt", delimiter="\t", header=None, names=["a", "b", "c", "d", "e"]
)

data_encoded = data.apply(preprocessing.LabelEncoder().fit_transform)
print(data_encoded)

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

clf.fit(data_encoded[["a", "b", "c", "d"]], data_encoded["e"])
print(clf.predict([[1, 2, 2, 0]]))
dot_data = export_graphviz(
    clf, out_file=None, feature_names=["Outlook", "Temp.", "Humidity", "Wind"]
)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("tennis_tree.png")

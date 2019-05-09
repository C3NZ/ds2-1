import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

data = pd.DataFrame(
    {
        "sepal length": iris.data[:, 0],
        "sepal width": iris.data[:, 1],
        "petal length": iris.data[:, 2],
        "petal width": iris.data[:, 3],
        "species": iris.target,
    }
)

print(data.head())


X = data[["sepal length", "sepal width", "petal length", "petal width"]]  # Features
y = data["species"]  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)

y_prediction = clf.predict(X_test)

print(y_prediction)

print(accuracy_score(y_test, y_prediction))

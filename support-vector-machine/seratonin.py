import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm

# Obtain SVM dataframe
df = pd.read_csv("SVM_Dataset2.csv")

# slice the the seratonin and dopamine levels from the dataframe
seratonin_levels = df["x1"]
dopamine_levels = df["x2"]

# Zip chemical levels together (X values (matrix))
chemical_levels = np.array(list(zip(seratonin_levels, dopamine_levels)))

# obtain health values (y value (vertex))
health_statuses = df["y"].values

# Color and labels for health statuses
color_ls = []
label = []

# Iterate over the health statuses, assigning blue to healthy (1) and
# non healthy statuses to red (-1)
for status in health_statuses:
    if status == 1:
        label.append("H")
        color_ls.append("b")
    else:
        label.append("NH")
        color_ls.append("r")

# Plot each matrix within a scatter plot (with approriate colors and labels)
for index, (seratonin, dopamine) in enumerate(chemical_levels):
    plt.scatter(seratonin, dopamine, c=color_ls[index])
    plt.text(seratonin + 0.002, dopamine + 0.02, label[index])

# Create the support vector machine classifier with a linear kernel to find the best line
# that fits the data provided.
svm_classifier = svm.SVC(kernel="linear", C=10)
svm_classifier.fit(chemical_levels, health_statuses)


# plot the decision boundary (the line of best fit)
def plot_decision_boundary(clf, X, y):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max), np.arange(x2_min, x2_max))
    Z = clf.decision_function(np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape)
    plt.contour(
        xx1,
        xx2,
        Z,
        colors="b",
        levels=[-1, 0, 1],
        alpha=0.4,
        linestyles=["--", "-", "--"],
    )
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())


# Plot the decision boundary
plot_decision_boundary(svm_classifier, chemical_levels, health_statuses)

# print the number of support vectors
print(svm_classifier.n_support_)

# Predict if you're healthy according to the model
print(svm_classifier.predict([[3, 6]]))

svm_classifier = svm.SVC(kernel="poly", C=10, degree=2)
svm_classifier.fit(chemical_levels, health_statuses)

plot_decision_boundary(svm_classifier, chemical_levels, health_statuses)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import make_blobs


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


# Number of samples that we'd like to generate
n_samples_1 = 1000
n_samples_2 = 100

# Cetners of each sample
centers = [[0.0, 0.0], [2.0, 2.0]]

# THe standard deviation of each sample(?)
clusters_std = [1.5, 0.5]

# Generate our matrix and vector values using sklearn make_blobs to generate the data
X, y = make_blobs(
    n_samples=[n_samples_1, n_samples_2],
    centers=centers,
    cluster_std=clusters_std,
    random_state=0,
    shuffle=False,
)

# List of colors
color_ls = []

# Y values are either 1, -1, or 0(?)
for status in y:
    if status == 1:
        color_ls.append("b")
    else:
        color_ls.append("r")

# Plot each matrix within a scatter plot (with approriate colors and labels)
for index, (seratonin, dopamine) in enumerate(X):
    plt.scatter(seratonin, dopamine, c=color_ls[index])

# Create our unweighted, linear classifier
clf = svm.SVC(kernel="linear", C=1.0)
clf.fit(X, y)

# Create a weighted linear classifier that will give the minority group
# a 10 to 1 weighting
wclf = svm.SVC(kernel="linear", class_weight={1: 10})
wclf.fit(X, y)

# Plot the decision boundary
plot_decision_boundary(wclf, X, y)

# Display the plot
plt.show()

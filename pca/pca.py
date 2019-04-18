import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA

# Load the excel file
df = pd.read_excel("pca_uk.xlsx")

# Convert the features into a list of lists
X = np.array([df[i].values for i in df.columns if i != "Features"])

print(X)

# fit the current features into two dimensions
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

# The amount of variance (information preserverd) within the system
print("Print ing the explained variance (same as the std?)")
print(pca.explained_variance_)

# THe amount of information preserved within the system
print(
    "printing the explained variance ratio (information preserved through each component)"
)
print(pca.explained_variance_ratio_)

# The cumulative sum of the ratios for each ratio
print("Printing the cumsum of the explained variance ratio")
print(pca.explained_variance_ratio_.cumsum())

# X_r is X reduced down to two dimensions
print(X_r)

# Create our scatter plot
sns.scatterplot(X_r[:, 0], X_r[:, 1])

# The columns that we want
wanted_cols = [col for col in df.columns if col != "Features"]

# Add a label to each point
for point in zip(X_r[:, 0], X_r[:, 1], wanted_cols):
    """
        Add text to every single point
    """
    plt.text(point[0] + 5, point[1] + 20, point[2])

print("Plotting our chart")
plt.show()


A = np.array([[2, 0], [1, 5]])
v = np.array([3, 4])
print(np.dot(A, v))

eig_value, eig_vector = np.linalg.eig(A)
print(eig_value)
print(eig_vector)


def PCA_calculation(data, n_components):
    M = np.mean(data, axis=0)
    C = data - M
    covariance_matrix = np.cov(C.T)
    print(covariance_matrix)
    eig_value, eig_vector = np.linalg.eig(covariance_matrix)

    index = np.argsort(eig_value)[::-1]
    eig_vector

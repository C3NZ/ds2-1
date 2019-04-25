import string

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score

documents = [
    "This is the first sentence.",
    "This one is the second sentence.",
    "And this is the third one.",
    "Is this the first sentence?",
]


def calculate_doc_matrix():

    """
        Manually trying to calculate the document term matrix
    """
    unique_words = set(
        [word.lower() for doc in documents for word in doc.split() if word[-1]]
    )

    features = sorted(list(unique_words))
    doc_matrix = []
    doc_matrix.append(features)

    for doc in documents:
        doc_matrix.append([])
        counter = len(doc_matrix) - 1
        for word in doc.split():
            pass
    print(doc_matrix)


# our text vectorizer
vectorizer = CountVectorizer()
print(vectorizer.fit(documents))
X = vectorizer.fit_transform(documents)
print(X)

# Generate our data with 300 total samples and 4 clusters
X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

cluster = KMeans(n_clusters=4)
cluster.fit(X)
print(cluster.cluster_centers_)

vectorizer

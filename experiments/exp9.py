import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris


def plot_dendrogram(model, **kwargs):

    counts = np.zeros(model.children_.shape[0])

    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):

        current_count = 0

        for child_idx in merge:

            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]

        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [
            model.children_,
            model.distances_,
            counts
        ]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)


# Load Iris dataset
iris = load_iris()

X = iris.data

print("========== Hierarchical Clustering ==========\n")

print("Dataset : Iris")

print("Number of Samples :", X.shape[0])

print("Number of Features :", X.shape[1])

print("\nFeature Names:")

for feature in iris.feature_names:
    print("-", feature)


# Create Agglomerative Clustering model
model = AgglomerativeClustering(
    distance_threshold=0,
    n_clusters=None
)

model.fit(X)

print("\nModel trained successfully.")

print("Number of merges :", len(model.children_))


# Plot dendrogram
plt.figure(figsize=(12,7))

plt.title("Hierarchical Clustering Dendrogram")

plot_dendrogram(
    model,
    truncate_mode="level",
    p=3
)

plt.xlabel("Number of points in node")

plt.ylabel("Distance")

plt.grid(True)

plt.tight_layout()

plt.savefig("static/graphs/hierarchical.png")

plt.close()

print("\nDendrogram saved successfully.")

print("\nExperiment Completed Successfully.")
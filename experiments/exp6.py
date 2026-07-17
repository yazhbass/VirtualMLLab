import os
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------------------------------------
# Create graph folder if it doesn't exist
# ----------------------------------------------------

os.makedirs("static/graphs", exist_ok=True)

# ----------------------------------------------------
# Generate Blob Dataset
# ----------------------------------------------------

X, y = make_blobs(
    n_samples=150,
    n_features=2,
    centers=3,
    cluster_std=0.5,
    shuffle=True,
    random_state=0
)

# ----------------------------------------------------
# Create KMeans Model
# ----------------------------------------------------

model = KMeans(
    n_clusters=3,
    init="random",
    n_init=10,
    max_iter=300,
    tol=1e-4,
    random_state=0
)

# ----------------------------------------------------
# Train Model
# ----------------------------------------------------

y_kmeans = model.fit_predict(X)

# ----------------------------------------------------
# Evaluation
# ----------------------------------------------------

sil_score = silhouette_score(X, y_kmeans)

print("=" * 50)
print("K-MEANS CLUSTERING")
print("=" * 50)

print()

print("Number of Samples :", len(X))

print("Number of Features :", X.shape[1])

print("Number of Clusters :", model.n_clusters)

print()

print("Cluster Centers")

print(model.cluster_centers_)

print()

print("Inertia (WCSS) : {:.4f}".format(model.inertia_))

print()

print("Silhouette Score : {:.4f}".format(sil_score))

print()

# ----------------------------------------------------
# Plot Clusters
# ----------------------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    X[y_kmeans==0,0],
    X[y_kmeans==0,1],
    s=50,
    c="lightgreen",
    marker="s",
    edgecolor="black",
    label="Cluster 1"
)

plt.scatter(
    X[y_kmeans==1,0],
    X[y_kmeans==1,1],
    s=50,
    c="orange",
    marker="o",
    edgecolor="black",
    label="Cluster 2"
)

plt.scatter(
    X[y_kmeans==2,0],
    X[y_kmeans==2,1],
    s=50,
    c="skyblue",
    marker="^",
    edgecolor="black",
    label="Cluster 3"
)

# ----------------------------------------------------
# Plot Centroids
# ----------------------------------------------------

plt.scatter(
    model.cluster_centers_[:,0],
    model.cluster_centers_[:,1],
    s=300,
    marker="*",
    c="red",
    edgecolor="black",
    label="Centroids"
)

plt.title("K-Means Clustering on Blob Dataset")

plt.xlabel("Feature 1")

plt.ylabel("Feature 2")

plt.grid(True)

plt.legend()

plt.savefig(
    "static/graphs/kmeans.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Cluster graph generated successfully.")

print()

print("Graph saved as:")

print("static/graphs/kmeans.png")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage


CSV_PATH = "data/cleaned/combined_30_matches.csv"
OUT_IMG = "imagesss"
OUT_DIR = "outputs"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH).dropna(subset=["striker", "runs_off_bat", "ball"])

# -------------------------
# Build a labeled dataset (label = striker)
# -------------------------
g = df.groupby("striker")

player_df = pd.DataFrame({
    "balls_faced": g.size(),
    "total_runs": g["runs_off_bat"].sum(),
    "total_extras_seen": g["extras"].sum(),
})

player_df["avg_runs_per_ball"] = player_df["total_runs"] / player_df["balls_faced"]
player_df["boundary_rate"] = g["runs_off_bat"].apply(lambda x: np.mean(x.isin([4, 6])))
player_df["dot_ball_rate"] = g["runs_off_bat"].apply(lambda x: np.mean(x == 0))

player_df = player_df.dropna()

# Save labels separately (required)
labels = player_df.index.to_series().reset_index(drop=True)
labels.to_csv(f"{OUT_DIR}/clustering_labels_striker.csv", index=False)

X = player_df.values

# -------------------------
# Standardize (required)
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional PCA to 3D for easier plotting (recommended)
pca = PCA(n_components=3, random_state=42)
X_3d = pca.fit_transform(X_scaled)

# -------------------------
# Silhouette scores to choose k (required)
# -------------------------
k_values = list(range(2, 11))
sil_scores = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    preds = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, preds))

plt.figure()
plt.plot(k_values, sil_scores, marker="o")
plt.title("Silhouette Scores (KMeans) for Player Clustering")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/silhouette_scores.png", dpi=200)
plt.show()

top3_k = [k_values[i] for i in np.argsort(sil_scores)[-3:]][::-1]
print("Top 3 k values (silhouette):", top3_k)

# -------------------------
# KMeans plots for 3 k values (required)
# -------------------------
for idx, k in enumerate(top3_k, start=1):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_id = km.fit_predict(X_scaled)

    centroids_scaled = km.cluster_centers_
    centroids_3d = pca.transform(centroids_scaled)

    plt.figure()
    plt.scatter(X_3d[:, 0], X_3d[:, 1], s=25, c=cluster_id)
    plt.scatter(centroids_3d[:, 0], centroids_3d[:, 1], s=250, marker="X")
    plt.title(f"KMeans Player Clusters (k={k}) with Centroids (PCA 2D view)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"{OUT_IMG}/kmeans_k{idx}.png", dpi=200)
    plt.show()

# -------------------------
# Hierarchical clustering dendrogram (required)
# -------------------------
Z = linkage(X_scaled, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode="lastp", p=25)
plt.title("Hierarchical Clustering Dendrogram (Players, truncated)")
plt.xlabel("Cluster size")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/hierarchical_dendrogram.png", dpi=200)
plt.show()

# -------------------------
# DBSCAN (required) - tune eps if needed
# -------------------------
db = DBSCAN(eps=0.9, min_samples=3)
db_labels = db.fit_predict(X_scaled)

plt.figure()
plt.scatter(X_3d[:, 0], X_3d[:, 1], s=25, c=db_labels)
plt.title("DBSCAN Player Clusters (PCA 2D view) - noise = -1")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/dbscan_clusters.png", dpi=200)
plt.show()

unique, counts = np.unique(db_labels, return_counts=True)
print("DBSCAN label counts:", dict(zip(unique, counts)))

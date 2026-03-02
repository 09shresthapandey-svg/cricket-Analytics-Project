import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

CSV_PATH = "data/cleaned/combined_30_matches.csv"
OUT_IMG = "imagesss"
OUT_DIR = "outputs"
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH).dropna(subset=["striker", "runs_off_bat"])

# Build labeled dataset (label=striker)
g = df.groupby("striker")

player_df = pd.DataFrame({
    "balls_faced": g.size(),
    "total_runs": g["runs_off_bat"].sum(),
    "total_extras_seen": g["extras"].sum(),
})
player_df["avg_runs_per_ball"] = player_df["total_runs"] / player_df["balls_faced"]
player_df["boundary_rate"] = g["runs_off_bat"].apply(lambda x: np.mean(x.isin([4,6])))
player_df["dot_ball_rate"] = g["runs_off_bat"].apply(lambda x: np.mean(x == 0))
player_df = player_df.dropna()

# Save labels separately (required)
player_df.index.to_series().to_csv(f"{OUT_DIR}/cluster_labels_striker.csv", index=False)

X = player_df.values

# Normalize (required)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 3D for plotting
pca = PCA(n_components=3, random_state=42)
X3 = pca.fit_transform(X_scaled)

# Silhouette to choose k
k_values = list(range(2, 11))
scores = []
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    preds = km.fit_predict(X_scaled)
    scores.append(silhouette_score(X_scaled, preds))

plt.figure()
plt.plot(k_values, scores, marker="o")
plt.title("Silhouette Scores (KMeans)")
plt.xlabel("k")
plt.ylabel("silhouette")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/silhouette_scores.png", dpi=200)
plt.show()

top3_k = [k_values[i] for i in np.argsort(scores)[-3:]][::-1]
print("Top 3 k values:", top3_k)

# 3 KMeans plots with centroids
for idx, k in enumerate(top3_k, start=1):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cid = km.fit_predict(X_scaled)
    cent3 = pca.transform(km.cluster_centers_)

    plt.figure()
    plt.scatter(X3[:,0], X3[:,1], c=cid, s=25)
    plt.scatter(cent3[:,0], cent3[:,1], s=250, marker="X")
    plt.title(f"KMeans Clusters (k={k})")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"{OUT_IMG}/kmeans_k{idx}.png", dpi=200)
    plt.show()

# Hierarchical dendrogram
Z = linkage(X_scaled, method="ward")
plt.figure(figsize=(10,5))
dendrogram(Z, truncate_mode="lastp", p=25)
plt.title("Hierarchical Dendrogram (Players)")
plt.xlabel("cluster size"); plt.ylabel("distance")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/hierarchical_dendrogram.png", dpi=200)
plt.show()

# DBSCAN
db = DBSCAN(eps=0.9, min_samples=3)
dbl = db.fit_predict(X_scaled)

plt.figure()
plt.scatter(X3[:,0], X3[:,1], c=dbl, s=25)
plt.title("DBSCAN Clusters (noise=-1)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/dbscan_clusters.png", dpi=200)
plt.show()

print("DBSCAN counts:", dict(zip(*np.unique(dbl, return_counts=True))))

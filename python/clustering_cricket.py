import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

CSV_PATH = "data/cleaned/combined_30_matches.csv"
OUT_IMG = "imagesss"
OUT_DIR = "outputs"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Load cleaned ball-by-ball data
df = pd.read_csv(CSV_PATH).dropna(subset=["striker", "runs_off_bat", "extras"])

# Build labeled player dataset (label = striker)
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

# Save labeled dataset
labeled_df = player_df.copy()
labeled_df.insert(0, "striker", labeled_df.index)
labeled_df.to_csv(f"{OUT_DIR}/cluster_labeled_dataset.csv", index=False)

# Save labels separately
labels_series = player_df.index.to_series(name="striker")
labels_series.to_csv(f"{OUT_DIR}/cluster_labels_striker.csv", index=False)

# Image of labeled dataset
sample_labeled = labeled_df.head(20)
plt.figure(figsize=(12, 4))
plt.axis("off")
tbl = plt.table(
    cellText=sample_labeled.values,
    colLabels=sample_labeled.columns,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.2)
plt.title("Labeled Player Dataset Before Clustering (First 20 Rows)", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/cluster_labeled_data.png", dpi=200, bbox_inches="tight")
plt.close()

# Remove labels for clustering
X_df = player_df.copy()
X_df.to_csv(f"{OUT_DIR}/cluster_unlabeled_data.csv", index=False)

# Image of unlabeled dataset
sample_unlabeled = X_df.head(20)
plt.figure(figsize=(11, 4))
plt.axis("off")
tbl = plt.table(
    cellText=sample_unlabeled.values,
    colLabels=sample_unlabeled.columns,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.2)
plt.title("Unlabeled Quantitative Dataset for Clustering (First 20 Rows)", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/cluster_unlabeled_data.png", dpi=200, bbox_inches="tight")
plt.close()

# Normalize with StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)
scaled_df.to_csv(f"{OUT_DIR}/cluster_scaled_data.csv", index=False)

# Image of scaled dataset
sample_scaled = scaled_df.head(20)
plt.figure(figsize=(11, 4))
plt.axis("off")
tbl = plt.table(
    cellText=np.round(sample_scaled.values, 3),
    colLabels=sample_scaled.columns,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.2)
plt.title("Normalized Clustering Dataset (First 20 Rows)", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/cluster_scaled_data.png", dpi=200, bbox_inches="tight")
plt.close()

# Optional PCA to 3D for visualization
pca = PCA(n_components=3)
X3 = pca.fit_transform(X_scaled)
var3 = pca.explained_variance_ratio_.sum() * 100
print(f"Variance retained after PCA to 3D: {var3:.2f}%")

pca3_df = pd.DataFrame(X3, columns=["PC1", "PC2", "PC3"])
pca3_df.to_csv(f"{OUT_DIR}/cluster_pca_3d_data.csv", index=False)

# Image of PCA 3D dataframe
sample_pca3 = pca3_df.head(20)
plt.figure(figsize=(9, 4))
plt.axis("off")
tbl = plt.table(
    cellText=np.round(sample_pca3.values, 3),
    colLabels=sample_pca3.columns,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.2)
plt.title("PCA-Reduced 3D Clustering Data (First 20 Rows)", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/cluster_pca_3d_data.png", dpi=200, bbox_inches="tight")
plt.close()

# Encode original labels only for plot coloring
le = LabelEncoder()
label_colors = le.fit_transform(labels_series.values)

# Silhouette scores for KMeans
k_values = list(range(2, 11))
scores = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    preds = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, preds)
    scores.append(score)

sil_df = pd.DataFrame({"k": k_values, "silhouette_score": scores})
sil_df.to_csv(f"{OUT_DIR}/cluster_silhouette_scores.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(k_values, scores, marker="o")
plt.title("Silhouette Scores (KMeans)")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/silhouette_scores.png", dpi=200)
plt.close()

top3_k = [k_values[i] for i in np.argsort(scores)[-3:]][::-1]
print("Top 3 k values:", top3_k)

# KMeans plots with centroids and label coloring
for idx, k in enumerate(top3_k, start=1):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(X_scaled)
    cent3 = pca.transform(km.cluster_centers_)

    result_df = pca3_df.copy()
    result_df["cluster"] = cluster_ids
    result_df["original_label"] = labels_series.values
    result_df.to_csv(f"{OUT_DIR}/kmeans_k{k}_results.csv", index=False)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X3[:, 0], X3[:, 1], c=label_colors, s=28, alpha=0.8)
    plt.scatter(cent3[:, 0], cent3[:, 1], s=250, marker="X")
    plt.title(f"KMeans Clusters (k={k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"{OUT_IMG}/kmeans_k{idx}.png", dpi=200)
    plt.close()

# Hierarchical clustering with cosine distance
cosine_dist = pdist(X_scaled, metric="cosine")
Z = linkage(cosine_dist, method="average")

plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode="lastp", p=25)
plt.title("Hierarchical Dendrogram (Cosine Distance, Average Linkage)")
plt.xlabel("Cluster Size")
plt.ylabel("Cosine Distance")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/hierarchical_dendrogram.png", dpi=200)
plt.close()

# DBSCAN
db = DBSCAN(eps=0.9, min_samples=3)
db_labels = db.fit_predict(X_scaled)

dbscan_df = pca3_df.copy()
dbscan_df["dbscan_cluster"] = db_labels
dbscan_df["original_label"] = labels_series.values
dbscan_df.to_csv(f"{OUT_DIR}/dbscan_results.csv", index=False)

plt.figure(figsize=(8, 6))
plt.scatter(X3[:, 0], X3[:, 1], c=db_labels, s=28)
plt.title("DBSCAN Clusters (noise = -1)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/dbscan_clusters.png", dpi=200)
plt.close()

# Save clustering summary
summary_df = pd.DataFrame({
    "metric": [
        "Variance retained after PCA to 3D (%)",
        "Top silhouette k values"
    ],
    "value": [
        round(var3, 2),
        str(top3_k)
    ]
})
summary_df.to_csv(f"{OUT_DIR}/clustering_summary.csv", index=False)

print("DBSCAN counts:", dict(zip(*np.unique(db_labels, return_counts=True))))
print("Clustering outputs saved successfully.")

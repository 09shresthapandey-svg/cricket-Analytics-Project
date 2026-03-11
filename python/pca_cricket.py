import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

CSV_PATH = "data/cleaned/combined_30_matches.csv"
OUT_IMG = "imagesss"
OUT_DIR = "outputs"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(CSV_PATH)

# Keep only quantitative variables for PCA
numeric_cols = ["innings", "ball", "runs_off_bat", "extras"]
X = df[numeric_cols].dropna().copy()

# Save prepared numeric dataset
prepared_csv = f"{OUT_DIR}/pca_prepared_data.csv"
X.to_csv(prepared_csv, index=False)
print(f"Saved cleaned/prepared PCA data: {prepared_csv}")

# Image of prepared dataset
sample_prepared = X.head(20)
plt.figure(figsize=(10, 4))
plt.axis("off")
tbl = plt.table(
    cellText=sample_prepared.values,
    colLabels=sample_prepared.columns,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.2)
plt.title("Prepared Data for PCA (First 20 Rows)", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/pca_prepared_data.png", dpi=200, bbox_inches="tight")
plt.close()

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)
scaled_df.to_csv(f"{OUT_DIR}/pca_standardized_data.csv", index=False)

# Full PCA for explained variance
pca_full = PCA()
pca_full.fit(X_scaled)

explained_var = pca_full.explained_variance_ratio_
cum = np.cumsum(explained_var)
eigenvalues = pca_full.explained_variance_

# Scree plot / explained variance bar chart
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_var) + 1), explained_var)
plt.title("Explained Variance by Principal Component")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/pca_explained_variance_bar.png", dpi=200)
plt.close()

# Cumulative explained variance plot
k95 = int(np.argmax(cum >= 0.95) + 1)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum) + 1), cum, marker="o")
plt.axhline(0.95, linestyle="--")
plt.axvline(k95, linestyle="--")
plt.title("Cumulative Explained Variance")
plt.xlabel("Components")
plt.ylabel("Cumulative Variance")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/pca_cumulative_variance.png", dpi=200)
plt.close()

# PCA 2D
pca2 = PCA(n_components=2)
X2 = pca2.fit_transform(X_scaled)
var2 = pca2.explained_variance_ratio_.sum() * 100

print(f"Variance retained (2D): {var2:.2f}%")
print("Explained variance ratio 2D:", pca2.explained_variance_ratio_)

pca2_df = pd.DataFrame(X2, columns=["PC1", "PC2"])
pca2_df.to_csv(f"{OUT_DIR}/pca_2d_coordinates.csv", index=False)

# Image of 2D PCA dataframe
sample_2d = pca2_df.head(20)
plt.figure(figsize=(8, 4))
plt.axis("off")
tbl = plt.table(
    cellText=sample_2d.values,
    colLabels=sample_2d.columns,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.2)
plt.title("2D PCA Transformed Data (First 20 Rows)", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/pca_2d_dataframe.png", dpi=200, bbox_inches="tight")
plt.close()

# 2D scatter
plt.figure(figsize=(8, 6))
plt.scatter(X2[:, 0], X2[:, 1], s=10)
plt.title("PCA 2D Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/pca_2d.png", dpi=200)
plt.close()

# PCA 3D
pca3 = PCA(n_components=3)
X3 = pca3.fit_transform(X_scaled)
var3 = pca3.explained_variance_ratio_.sum() * 100

print(f"Variance retained (3D): {var3:.2f}%")
print("Explained variance ratio 3D:", pca3.explained_variance_ratio_)

pca3_df = pd.DataFrame(X3, columns=["PC1", "PC2", "PC3"])
pca3_df.to_csv(f"{OUT_DIR}/pca_3d_coordinates.csv", index=False)

# Image of 3D PCA dataframe
sample_3d = pca3_df.head(20)
plt.figure(figsize=(9, 4))
plt.axis("off")
tbl = plt.table(
    cellText=sample_3d.values,
    colLabels=sample_3d.columns,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.2)
plt.title("3D PCA Transformed Data (First 20 Rows)", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/pca_3d_dataframe.png", dpi=200, bbox_inches="tight")
plt.close()

# 3D scatter
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], s=8)
ax.set_title("PCA 3D Projection")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/pca_3d.png", dpi=200)
plt.close()

# Top 3 eigenvalues
top3 = eigenvalues[:3]
print("Top 3 eigenvalues:", top3)

eig_df = pd.DataFrame({
    "principal_component": ["PC1", "PC2", "PC3"],
    "eigenvalue": top3
})
eig_df.to_csv(f"{OUT_DIR}/top3_eigenvalues.csv", index=False)

plt.figure(figsize=(6, 2.5))
plt.axis("off")
tbl = plt.table(
    cellText=eig_df.values,
    colLabels=eig_df.columns,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.4)
plt.title("Top 3 Eigenvalues from PCA Output", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/pca_top3_eigenvalues.png", dpi=200, bbox_inches="tight")
plt.close()

# Summary results
summary_df = pd.DataFrame({
    "metric": [
        "Variance retained in 2D (%)",
        "Variance retained in 3D (%)",
        "Components needed for >=95% variance"
    ],
    "value": [
        round(var2, 2),
        round(var3, 2),
        k95
    ]
})
summary_df.to_csv(f"{OUT_DIR}/pca_summary_results.csv", index=False)

plt.figure(figsize=(7, 2.5))
plt.axis("off")
tbl = plt.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.4)
plt.title("PCA Summary Results", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/pca_summary_results.png", dpi=200, bbox_inches="tight")
plt.close()

# Variance summary csv
variance_df = pd.DataFrame({
    "component": range(1, len(explained_var) + 1),
    "explained_variance_ratio": explained_var,
    "cumulative_variance": cum,
    "eigenvalue": eigenvalues
})
variance_df.to_csv(f"{OUT_DIR}/pca_variance_summary.csv", index=False)

print("PCA outputs saved successfully:")
print(f" - {OUT_IMG}/pca_prepared_data.png")
print(f" - {OUT_IMG}/pca_2d_dataframe.png")
print(f" - {OUT_IMG}/pca_3d_dataframe.png")
print(f" - {OUT_IMG}/pca_explained_variance_bar.png")
print(f" - {OUT_IMG}/pca_cumulative_variance.png")
print(f" - {OUT_IMG}/pca_2d.png")
print(f" - {OUT_IMG}/pca_3d.png")
print(f" - {OUT_IMG}/pca_top3_eigenvalues.png")
print(f" - {OUT_IMG}/pca_summary_results.png")
print(f" - {OUT_DIR}/pca_prepared_data.csv")
print(f" - {OUT_DIR}/pca_standardized_data.csv")
print(f" - {OUT_DIR}/pca_2d_coordinates.csv")
print(f" - {OUT_DIR}/pca_3d_coordinates.csv")
print(f" - {OUT_DIR}/top3_eigenvalues.csv")
print(f" - {OUT_DIR}/pca_summary_results.csv")
print(f" - {OUT_DIR}/pca_variance_summary.csv")

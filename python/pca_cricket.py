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

df = pd.read_csv(CSV_PATH)

# PCA requires numeric only, drop labels (teams, players, wicket_type, match_id)
numeric_cols = ["innings", "ball", "runs_off_bat", "extras"]
X = df[numeric_cols].dropna()

# Standardize (required)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 2D
pca2 = PCA(n_components=2, random_state=42)
X2 = pca2.fit_transform(X_scaled)
var2 = pca2.explained_variance_ratio_.sum() * 100
print(f"Variance retained (2D): {var2:.2f}%")
print("Explained variance ratio 2D:", pca2.explained_variance_ratio_)

plt.figure()
plt.scatter(X2[:,0], X2[:,1], s=10)
plt.title("PCA 2D (innings, ball, runs_off_bat, extras)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/pca_2d.png", dpi=200)
plt.show()

# PCA 3D
pca3 = PCA(n_components=3, random_state=42)
X3 = pca3.fit_transform(X_scaled)
var3 = pca3.explained_variance_ratio_.sum() * 100
print(f"Variance retained (3D): {var3:.2f}%")
print("Explained variance ratio 3D:", pca3.explained_variance_ratio_)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X3[:,0], X3[:,1], X3[:,2], s=8)
ax.set_title("PCA 3D")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/pca_3d.png", dpi=200)
plt.show()

# 95% variance components
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)
cum = np.cumsum(pca_full.explained_variance_ratio_)
k95 = int(np.argmax(cum >= 0.95) + 1)
print(f"Components for >=95% variance: {k95} (cum={cum[k95-1]:.4f})")

plt.figure()
plt.plot(range(1, len(cum)+1), cum, marker="o")
plt.axhline(0.95)
plt.title("Cumulative Explained Variance")
plt.xlabel("Components")
plt.ylabel("Cumulative variance")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/pca_cumulative_variance.png", dpi=200)
plt.show()

# Top 3 eigenvalues
top3 = pca_full.explained_variance_[:3]
print("Top 3 eigenvalues:", top3)
pd.DataFrame({"top3_eigenvalues": top3}).to_csv(f"{OUT_DIR}/top3_eigenvalues.csv", index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, classification_report)

# ── LOAD AND PREPARE DATA ─────────────────────────────────────────────────────
df = pd.read_csv("cluster_labeled_dataset.csv")
# Create label FIRST using avg_runs_per_ball before removing it from features
all_cols = ["balls_faced", "total_runs", "total_extras_seen",
            "avg_runs_per_ball", "boundary_rate", "dot_ball_rate"]
df_clean = df[all_cols].dropna()

# Create label from avg_runs_per_ball
median_val = df_clean["avg_runs_per_ball"].median()
y = (df_clean["avg_runs_per_ball"] >= median_val).astype(int)

# Now drop avg_runs_per_ball from features to avoid data leakage
features = ["balls_faced", "total_runs", "total_extras_seen",
            "boundary_rate", "dot_ball_rate"]
X = df_clean[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ── OVERVIEW IMAGE ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")
ax.set_title("Ensemble Learning — Random Forest (100 Trees)",
             fontsize=13, fontweight="bold", pad=15)
text = (
    "Random Forest is an ensemble of Decision Trees.\n"
    "Each tree is trained on a random subset of data (bagging).\n"
    "Each tree votes on the class → majority vote wins.\n"
    "Result: lower variance, higher accuracy than a single tree.\n\n"
    "Bagging → Bootstrap samples → 100 trees → Majority Vote → Final Prediction"
)
ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=11,
        ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="#f7fff8",
                  edgecolor="#0b3d2e", linewidth=2))
plt.tight_layout()
plt.savefig("imagesss/ensemble_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: imagesss/ensemble_overview.png")

# ── TRAIN RANDOM FOREST ───────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, criterion="gini",
                            random_state=42)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Random Forest Accuracy: {acc:.4f}")

# ── CONFUSION MATRIX ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Low Scorer", "High Scorer"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Greens")
ax.set_title(f"Random Forest (100 Trees)\nAccuracy: {acc:.4f}")
plt.tight_layout()
plt.savefig("imagesss/ensemble_cm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: imagesss/ensemble_cm.png")

# ── CLASSIFICATION REPORT ─────────────────────────────────────────────────────
report = classification_report(y_test, preds,
                               target_names=["Low Scorer", "High Scorer"])
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")
ax.set_title("Classification Report — Random Forest",
             fontsize=11, fontweight="bold", pad=10)
ax.text(0.01, 0.95, report, transform=ax.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
plt.tight_layout()
plt.savefig("imagesss/ensemble_cr.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: imagesss/ensemble_cr.png")

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [features[i] for i in indices]
sorted_importances = importances[indices]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(sorted_features[::-1], sorted_importances[::-1],
               color="#0b3d2e", edgecolor="white")
ax.set_xlabel("Feature Importance Score")
ax.set_title("Random Forest — Feature Importance\n(Which batting features matter most?)",
             fontweight="bold")
for bar, val in zip(bars, sorted_importances[::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("imagesss/ensemble_feature_importance.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: imagesss/ensemble_feature_importance.png")

print(f"\nFinal Accuracy: {acc:.4f}")

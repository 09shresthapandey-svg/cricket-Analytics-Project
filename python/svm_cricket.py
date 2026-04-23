import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.decomposition import PCA

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/cluster_labeled_dataset.csv")

# Keep only numeric feature columns
features = ["balls_faced", "total_runs", "total_extras_seen",
            "avg_runs_per_ball", "boundary_rate", "dot_ball_rate"]

X = df[features].dropna()
df = df.loc[X.index]  # align index after dropping NAs

# ── 2. CREATE LABEL ───────────────────────────────────────────────────────────
# Binary label: High Scorer (1) if avg_runs_per_ball >= median, else Low Scorer (0)
median_val = X["avg_runs_per_ball"].median()
y = (X["avg_runs_per_ball"] >= median_val).astype(int)

# ── 3. SCALE FEATURES ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. TRAIN-TEST SPLIT (80/20) ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Save labeled dataset for website reference
labeled_df = X.copy()
labeled_df["label"] = y.values
labeled_df.to_csv("data/svm_dataset.csv", index=False)
print("Saved: data/svm_dataset.csv")

# ── 5. SAVE TRAIN/TEST PREVIEW IMAGES ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].set_title("Training Set Preview (first 10 rows)")
axes[0].axis("off")
train_df = pd.DataFrame(X_train[:10], columns=features)
train_df["label"] = y_train.values[:10]
table0 = axes[0].table(cellText=train_df.round(2).values,
                        colLabels=train_df.columns, loc="center", cellLoc="center")
table0.auto_set_font_size(False)
table0.set_fontsize(7)

axes[1].set_title("Testing Set Preview (first 10 rows)")
axes[1].axis("off")
test_df = pd.DataFrame(X_test[:10], columns=features)
test_df["label"] = y_test.values[:10]
table1 = axes[1].table(cellText=test_df.round(2).values,
                        colLabels=test_df.columns, loc="center", cellLoc="center")
table1.auto_set_font_size(False)
table1.set_fontsize(7)

plt.tight_layout()
plt.savefig("imagesss/svm_dataset.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: imagesss/svm_dataset.png")

# ── 6. SVM OVERVIEW IMAGE (linear separator concept) ──────────────────────────
# Use PCA to 2D for visualization
pca2 = PCA(n_components=2)
X_2d = pca2.fit_transform(X_scaled)
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)

fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#0b3d2e", "#f4a261"]
for cls, color, label in zip([0, 1], colors, ["Low Scorer", "High Scorer"]):
    mask = y.values == cls
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=label,
               alpha=0.6, edgecolors="k", linewidths=0.4, s=40)
ax.set_title("SVM — Data in 2D PCA Space (Linear Separator Concept)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
plt.tight_layout()
plt.savefig("imagesss/svm_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: imagesss/svm_overview.png")

# ── 7. KERNEL TRICK ILLUSTRATION (polynomial casting) ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: original 2D points — not linearly separable
np.random.seed(42)
n = 60
theta = np.linspace(0, 2 * np.pi, n)
r_inner = 0.5 + 0.1 * np.random.randn(n // 2)
r_outer = 1.5 + 0.1 * np.random.randn(n // 2)
x_inner = r_inner * np.cos(theta[:n // 2])
y_inner = r_inner * np.sin(theta[:n // 2])
x_outer = r_outer * np.cos(theta[n // 2:])
y_outer = r_outer * np.sin(theta[n // 2:])

axes[0].scatter(x_inner, y_inner, c="#0b3d2e", label="Class 0", s=40, edgecolors="k", lw=0.4)
axes[0].scatter(x_outer, y_outer, c="#f4a261", label="Class 1", s=40, edgecolors="k", lw=0.4)
axes[0].set_title("Original 2D Space\n(Not linearly separable)")
axes[0].legend()
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")

# Right: transformed space using z = x1^2 + x2^2 (polynomial kernel r=1, d=2)
z_inner = x_inner**2 + y_inner**2
z_outer = x_outer**2 + y_outer**2
axes[1].scatter(x_inner, z_inner, c="#0b3d2e", label="Class 0", s=40, edgecolors="k", lw=0.4)
axes[1].scatter(x_outer, z_outer, c="#f4a261", label="Class 1", s=40, edgecolors="k", lw=0.4)
axes[1].axhline(y=1.0, color="red", linestyle="--", linewidth=2, label="Linear separator in new space")
axes[1].set_title("Polynomial Kernel (r=1, d=2)\nCast to higher dimension z = x1^2 + x2^2")
axes[1].legend()
axes[1].set_xlabel("x1")
axes[1].set_ylabel("z = x1^2 + x2^2")

plt.suptitle("Kernel Trick: Casting 2D Points into Higher Dimensions", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("imagesss/svm_kernel.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: imagesss/svm_kernel.png")

# ── 8. HELPER: plot confusion matrix ──────────────────────────────────────────
def save_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Low Scorer", "High Scorer"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Greens")
    ax.set_title(f"{title}\nAccuracy: {acc:.4f}")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}  |  Accuracy: {acc:.4f}")
    return acc

# ── 9. HELPER: plot decision boundary (2D PCA) ────────────────────────────────
def save_boundary(clf, X_2d_train, y_2d_train, title, filename):
    clf_2d = SVC(kernel=clf.kernel, C=clf.C,
                 degree=getattr(clf, "degree", 3),
                 gamma=getattr(clf, "gamma", "scale"))
    clf_2d.fit(X_2d_train, y_2d_train)

    h = 0.05
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlGn")
    colors = ["#0b3d2e", "#f4a261"]
    for cls, color, lbl in zip([0, 1], colors, ["Low Scorer", "High Scorer"]):
        mask = y.values == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=lbl,
                   alpha=0.7, edgecolors="k", linewidths=0.4, s=35)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")

# ── 10. LINEAR KERNEL ─────────────────────────────────────────────────────────
print("\n--- LINEAR KERNEL ---")
best_acc_linear, best_C_linear, best_clf_linear = 0, 1, None
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    clf = SVC(kernel="linear", C=C, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  C={C:>6} -> Accuracy: {acc:.4f}")
    if acc > best_acc_linear:
        best_acc_linear, best_C_linear, best_clf_linear = acc, C, clf

print(f"Best Linear C={best_C_linear}, Accuracy={best_acc_linear:.4f}")
save_cm(y_test, best_clf_linear.predict(X_test),
        f"Linear Kernel (C={best_C_linear})",
        "imagesss/svm_cm_linear.png")
save_boundary(best_clf_linear, X_train_2d, y_train_2d,
              f"Decision Boundary — Linear Kernel (C={best_C_linear})",
              "imagesss/svm_viz_linear.png")

# ── 11. POLYNOMIAL KERNEL ─────────────────────────────────────────────────────
print("\n--- POLYNOMIAL KERNEL (degree=2, r=1) ---")
best_acc_poly, best_C_poly, best_clf_poly = 0, 1, None
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    clf = SVC(kernel="poly", degree=2, coef0=1, C=C, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  C={C:>6} -> Accuracy: {acc:.4f}")
    if acc > best_acc_poly:
        best_acc_poly, best_C_poly, best_clf_poly = acc, C, clf

print(f"Best Poly C={best_C_poly}, Accuracy={best_acc_poly:.4f}")
save_cm(y_test, best_clf_poly.predict(X_test),
        f"Polynomial Kernel (C={best_C_poly}, d=2, r=1)",
        "imagesss/svm_cm_poly.png")
save_boundary(best_clf_poly, X_train_2d, y_train_2d,
              f"Decision Boundary — Polynomial Kernel (C={best_C_poly})",
              "imagesss/svm_viz_poly.png")

# ── 12. RBF KERNEL ────────────────────────────────────────────────────────────
print("\n--- RBF KERNEL ---")
best_acc_rbf, best_C_rbf, best_clf_rbf = 0, 1, None
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    clf = SVC(kernel="rbf", C=C, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  C={C:>6} -> Accuracy: {acc:.4f}")
    if acc > best_acc_rbf:
        best_acc_rbf, best_C_rbf, best_clf_rbf = acc, C, clf

print(f"Best RBF C={best_C_rbf}, Accuracy={best_acc_rbf:.4f}")
save_cm(y_test, best_clf_rbf.predict(X_test),
        f"RBF Kernel (C={best_C_rbf})",
        "imagesss/svm_cm_rbf.png")
save_boundary(best_clf_rbf, X_train_2d, y_train_2d,
              f"Decision Boundary — RBF Kernel (C={best_C_rbf})",
              "imagesss/svm_viz_rbf.png")

# ── 13. FINAL SUMMARY ─────────────────────────────────────────────────────────
print("\n========== FINAL SUMMARY ==========")
print(f"Linear  Kernel — Best C: {best_C_linear:<6} Accuracy: {best_acc_linear:.4f}")
print(f"Poly    Kernel — Best C: {best_C_poly:<6} Accuracy: {best_acc_poly:.4f}")
print(f"RBF     Kernel — Best C: {best_C_rbf:<6} Accuracy: {best_acc_rbf:.4f}")
best_overall = max(
    ("Linear", best_acc_linear, best_C_linear),
    ("Poly",   best_acc_poly,   best_C_poly),
    ("RBF",    best_acc_rbf,    best_C_rbf),
    key=lambda x: x[1]
)
print(f"BEST OVERALL: {best_overall[0]} kernel, C={best_overall[2]}, Accuracy={best_overall[1]:.4f}")

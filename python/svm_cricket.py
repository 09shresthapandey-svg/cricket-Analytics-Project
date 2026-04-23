import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, classification_report)
from sklearn.decomposition import PCA

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/cluster_labeled_dataset.csv")

features = ["balls_faced", "total_runs", "total_extras_seen",
            "avg_runs_per_ball", "boundary_rate", "dot_ball_rate"]

X = df[features].dropna()
df = df.loc[X.index]

# ── 2. CREATE LABEL ───────────────────────────────────────────────────────────
median_val = X["avg_runs_per_ball"].median()
y = (X["avg_runs_per_ball"] >= median_val).astype(int)

# ── 3. SCALE FEATURES ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. TRAIN-TEST SPLIT (80/20) ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Save labeled dataset
labeled_df = X.copy()
labeled_df["label"] = y.values
labeled_df.to_csv("data/svm_dataset.csv", index=False)
print("Saved: data/svm_dataset.csv")

# ── 5. PCA for 2D visualization ───────────────────────────────────────────────
pca2 = PCA(n_components=2)
X_2d = pca2.fit_transform(X_scaled)
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)

# ── 6. OVERVIEW IMAGE ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#0b3d2e", "#f4a261"]
for cls, color, label in zip([0, 1], colors, ["Low Scorer", "High Scorer"]):
    mask = y.values == cls
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=label,
               alpha=0.6, edgecolors="k", linewidths=0.4, s=40)
ax.set_title("SVM — Data in 2D PCA Space (Linear Separator Concept)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
plt.tight_layout()
plt.savefig("imagesss/svm_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: imagesss/svm_overview.png")

# ── 7. KERNEL TRICK IMAGE ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
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
axes[0].legend(); axes[0].set_xlabel("x1"); axes[0].set_ylabel("x2")

z_inner = x_inner**2 + y_inner**2
z_outer = x_outer**2 + y_outer**2
axes[1].scatter(x_inner, z_inner, c="#0b3d2e", label="Class 0", s=40, edgecolors="k", lw=0.4)
axes[1].scatter(x_outer, z_outer, c="#f4a261", label="Class 1", s=40, edgecolors="k", lw=0.4)
axes[1].axhline(y=1.0, color="red", linestyle="--", linewidth=2, label="Linear separator in new space")
axes[1].set_title("Polynomial Kernel (r=1, d=2)\nz = x1^2 + x2^2")
axes[1].legend(); axes[1].set_xlabel("x1"); axes[1].set_ylabel("z = x1^2 + x2^2")
plt.suptitle("Kernel Trick: Casting 2D Points into Higher Dimensions", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("imagesss/svm_kernel.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: imagesss/svm_kernel.png")

# ── 8. HELPERS ────────────────────────────────────────────────────────────────
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
    return acc

def save_cr(y_true, y_pred, title, filename):
    report = classification_report(y_true, y_pred,
                                   target_names=["Low Scorer", "High Scorer"])
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.text(0.01, 0.95, report, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

def save_boundary(kernel, C, degree=2, filename="temp.png", title=""):
    clf_2d = SVC(kernel=kernel, C=C, degree=degree, gamma="scale")
    clf_2d.fit(X_train_2d, y_train_2d)
    h = 0.05
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlGn")
    for cls, color, lbl in zip([0, 1], ["#0b3d2e", "#f4a261"],
                                ["Low Scorer", "High Scorer"]):
        mask = y.values == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=lbl,
                   alpha=0.7, edgecolors="k", linewidths=0.4, s=35)
    ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

# ── 9. THREE C VALUES PER KERNEL (9 CMs + 9 CRs total) ───────────────────────
C_values = [0.1, 1, 10]
results = {}

# ── LINEAR ────────────────────────────────────────────────────────────────────
print("\n--- LINEAR KERNEL ---")
best_acc_linear, best_C_linear = 0, C_values[0]
for C in C_values:
    clf = SVC(kernel="linear", C=C, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = save_cm(y_test, preds,
                  f"Linear Kernel (C={C})",
                  f"imagesss/svm_cm_linear_C{str(C).replace('.','p')}.png")
    save_cr(y_test, preds,
            f"Classification Report — Linear Kernel (C={C})",
            f"imagesss/svm_cr_linear_C{str(C).replace('.','p')}.png")
    print(f"  Linear C={C} -> Accuracy: {acc:.4f}")
    results[f"Linear C={C}"] = acc
    if acc > best_acc_linear:
        best_acc_linear, best_C_linear = acc, C

save_boundary("linear", best_C_linear,
              filename="imagesss/svm_viz_linear.png",
              title=f"Decision Boundary — Linear Kernel (Best C={best_C_linear})")
print(f"Best Linear: C={best_C_linear}, Acc={best_acc_linear:.4f}")

# ── POLYNOMIAL ────────────────────────────────────────────────────────────────
print("\n--- POLYNOMIAL KERNEL (degree=2, r=1) ---")
best_acc_poly, best_C_poly = 0, C_values[0]
for C in C_values:
    clf = SVC(kernel="poly", degree=2, coef0=1, C=C, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = save_cm(y_test, preds,
                  f"Polynomial Kernel (C={C}, d=2, r=1)",
                  f"imagesss/svm_cm_poly_C{str(C).replace('.','p')}.png")
    save_cr(y_test, preds,
            f"Classification Report — Polynomial Kernel (C={C})",
            f"imagesss/svm_cr_poly_C{str(C).replace('.','p')}.png")
    print(f"  Poly C={C} -> Accuracy: {acc:.4f}")
    results[f"Poly C={C}"] = acc
    if acc > best_acc_poly:
        best_acc_poly, best_C_poly = acc, C

save_boundary("poly", best_C_poly, degree=2,
              filename="imagesss/svm_viz_poly.png",
              title=f"Decision Boundary — Polynomial Kernel (Best C={best_C_poly})")
print(f"Best Poly: C={best_C_poly}, Acc={best_acc_poly:.4f}")

# ── RBF ───────────────────────────────────────────────────────────────────────
print("\n--- RBF KERNEL ---")
best_acc_rbf, best_C_rbf = 0, C_values[0]
for C in C_values:
    clf = SVC(kernel="rbf", C=C, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = save_cm(y_test, preds,
                  f"RBF Kernel (C={C})",
                  f"imagesss/svm_cm_rbf_C{str(C).replace('.','p')}.png")
    save_cr(y_test, preds,
            f"Classification Report — RBF Kernel (C={C})",
            f"imagesss/svm_cr_rbf_C{str(C).replace('.','p')}.png")
    print(f"  RBF C={C} -> Accuracy: {acc:.4f}")
    results[f"RBF C={C}"] = acc
    if acc > best_acc_rbf:
        best_acc_rbf, best_C_rbf = acc, C

save_boundary("rbf", best_C_rbf,
              filename="imagesss/svm_viz_rbf.png",
              title=f"Decision Boundary — RBF Kernel (Best C={best_C_rbf})")
print(f"Best RBF: C={best_C_rbf}, Acc={best_acc_rbf:.4f}")

# ── 10. COMPARISON TABLE IMAGE ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.axis("off")
ax.set_title("Kernel & Cost Comparison — All 9 Combinations",
             fontsize=12, fontweight="bold")
rows = [[k, f"{v:.4f}"] for k, v in results.items()]
table = ax.table(cellText=rows, colLabels=["Kernel / C Value", "Accuracy"],
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.4, 1.8)
plt.tight_layout()
plt.savefig("imagesss/svm_comparison_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: imagesss/svm_comparison_table.png")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n========== FINAL SUMMARY ==========")
for k, v in results.items():
    print(f"  {k:<20} -> {v:.4f}")
best = max(results, key=results.get)
print(f"\nBEST OVERALL: {best} -> {results[best]:.4f}")
print(f"\nBest per kernel:")
print(f"  Linear  -> C={best_C_linear}, Acc={best_acc_linear:.4f}")
print(f"  Poly    -> C={best_C_poly},   Acc={best_acc_poly:.4f}")
print(f"  RBF     -> C={best_C_rbf},    Acc={best_acc_rbf:.4f}")

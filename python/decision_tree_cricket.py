import os
os.makedirs("imagesss", exist_ok=True)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/cleaned/combined_30_matches.csv")

# Create label
df["label"] = df["runs_off_bat"].apply(lambda x: "LOW" if x <= 1 else "HIGH")

X = df[["ball", "innings", "extras"]]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Save dataset
df.to_csv("data/dt_dataset.csv", index=False)

# -------- TREE 1 --------
dt1 = DecisionTreeClassifier(max_depth=3)
dt1.fit(X_train, y_train)

# -------- TREE 2 --------
dt2 = DecisionTreeClassifier(max_depth=5)
dt2.fit(X_train, y_train)

# -------- TREE 3 --------
dt3 = DecisionTreeClassifier(criterion="entropy", max_depth=4)
dt3.fit(X_train, y_train)

# Plot trees
def save_tree(model, filename):
    plt.figure(figsize=(10,6))
    plot_tree(model, filled=True)
    plt.savefig(filename)
    plt.close()

save_tree(dt1, "imagesss/dt_tree1.png")
save_tree(dt2, "imagesss/dt_tree2.png")
save_tree(dt3, "imagesss/dt_tree3.png")

# Predictions
y_pred = dt1.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.savefig("imagesss/dt_confusion_matrix.png")
plt.close()

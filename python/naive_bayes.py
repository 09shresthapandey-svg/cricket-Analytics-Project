import pandas as pd

df = pd.read_csv("data/cleaned/combined_30_matches.csv")

# Create label (VERY IMPORTANT)
df["label"] = df["runs_off_bat"].apply(lambda x: "LOW" if x <= 1 else "HIGH")

# Features
X = df[["ball", "innings", "runs_off_bat", "extras"]]
y = df["label"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

# Multinomial NB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)

# Gaussian NB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

# Bernoulli NB
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred_bnb = bnb.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cm(y_test, y_pred, title, filename):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(title)
    plt.savefig(filename)
    plt.clf()

# Accuracy
print("MNB:", accuracy_score(y_test, y_pred_mnb))
print("GNB:", accuracy_score(y_test, y_pred_gnb))
print("BNB:", accuracy_score(y_test, y_pred_bnb))

# Save images
plot_cm(y_test, y_pred_mnb, "MNB", "imagesss/nb_confusion_matrix_mnb.png")
plot_cm(y_test, y_pred_gnb, "GNB", "imagesss/nb_confusion_matrix_gnb.png")
plot_cm(y_test, y_pred_bnb, "BNB", "imagesss/nb_confusion_matrix_bnb.png")

df.to_csv("data/nb_dataset.csv", index=False)

X_train.to_csv("data/train_data.csv", index=False)
X_test.to_csv("data/test_data.csv", index=False)

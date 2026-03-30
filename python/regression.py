import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your local dataset
data = pd.read_csv('/Users/shresthapandey/python/cluster_labeled_dataset.csv')

# Features and target
X = data[['balls_faced', 'avg_runs_per_ball', 'boundary_rate', 'dot_ball_rate']]
y = (data['total_runs'] > data['total_runs'].median()).astype(int)  # Example binary label

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

# Confusion matrices
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_nb = confusion_matrix(y_test, y_pred_nb)

# Function to save confusion matrix as image
def save_cm_image(cm, title, filename):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(filename)
    plt.close()

# Save images
save_cm_image(cm_lr, 'Logistic Regression Confusion Matrix', 'cm_lr.png')
save_cm_image(cm_nb, 'Naive Bayes Confusion Matrix', 'cm_nb.png')

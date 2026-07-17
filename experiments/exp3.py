import matplotlib
matplotlib.use("Agg")   # Required for Flask

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay
)

# -----------------------------
# Load Dataset
# -----------------------------
dataset = pd.read_csv("Social_Network_Ads.csv")

# Features (Age, Estimated Salary)
X = dataset.iloc[:, [2, 3]].values

# Target
y = dataset.iloc[:, 4].values

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0
)

# -----------------------------
# Feature Scaling
# -----------------------------
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -----------------------------
# Train SVM
# -----------------------------
classifier = SVC(
    kernel="linear",
    random_state=0
)

classifier.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = classifier.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("========== SVM CLASSIFIER ==========\n")

print("Confusion Matrix:\n")
print(cm)

print("\nAccuracy :", round(accuracy,4))
print("Precision:", round(precision,4))
print("Recall   :", round(recall,4))

# -----------------------------
# Plot Decision Boundary
# -----------------------------
from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(
    np.arange(
        start=X_set[:,0].min()-1,
        stop=X_set[:,0].max()+1,
        step=0.01
    ),
    np.arange(
        start=X_set[:,1].min()-1,
        stop=X_set[:,1].max()+1,
        step=0.01
    )
)

plt.figure(figsize=(8,6))

plt.contourf(
    X1,
    X2,
    classifier.predict(
        np.array([X1.ravel(), X2.ravel()]).T
    ).reshape(X1.shape),
    alpha=0.3,
    cmap=ListedColormap(("red","green"))
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set==j,0],
        X_set[y_set==j,1],
        c=ListedColormap(("red","green"))(i),
        label=f"Class {j}",
        edgecolors="black"
    )

plt.title("Support Vector Machine (Linear Kernel)")
plt.xlabel("Age (Scaled)")
plt.ylabel("Estimated Salary (Scaled)")
plt.legend()
plt.grid(True)

plt.savefig(
    "static/graphs/svm.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nDecision boundary graph saved successfully.")
import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Create graph folder if it doesn't exist
os.makedirs("static/graphs", exist_ok=True)

# -------------------------
# Load Dataset
# -------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

# -------------------------
# Split Dataset
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=1,
    stratify=y
)

# -------------------------
# Train Model
# -------------------------

model = DecisionTreeClassifier(
    criterion="gini",
    random_state=1
)

model.fit(X_train, y_train)

# -------------------------
# Prediction
# -------------------------

y_pred = model.predict(X_test)

# -------------------------
# Accuracy
# -------------------------

accuracy = accuracy_score(y_test, y_pred)

# -------------------------
# Confusion Matrix
# -------------------------

cm = confusion_matrix(y_test, y_pred)

print("=" * 50)
print("Decision Tree Classification")
print("=" * 50)
print()

print("Accuracy : {:.2f}%".format(accuracy * 100))

print()

print("Confusion Matrix")

print(cm)

print()

# -------------------------
# Plot Confusion Matrix
# -------------------------

fig, ax = plt.subplots(figsize=(6, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot(ax=ax)

plt.title("Confusion Matrix")

plt.savefig(
    "static/graphs/iris_confusion_matrix.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# -------------------------
# Plot Decision Tree
# -------------------------

plt.figure(figsize=(15, 10))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree Classifier")

plt.savefig(
    "static/graphs/iris_decision_tree.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Graphs Generated Successfully.")

print()

print("Confusion Matrix saved as")
print("static/graphs/iris_confusion_matrix.png")

print()

print("Decision Tree saved as")
print("static/graphs/iris_decision_tree.png")
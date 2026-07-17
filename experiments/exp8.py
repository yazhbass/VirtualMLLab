import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

# ----------------------------
# Load Iris Dataset
# ----------------------------

iris = datasets.load_iris()

X = iris.data
y = iris.target

print("========== ADABOOST CLASSIFIER ==========\n")

print("Input Features:")
print(iris.feature_names)

print("\nTarget Classes:")
print(iris.target_names)

# ----------------------------
# Train-Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ----------------------------
# Create AdaBoost Model
# ----------------------------

classifier = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1,
    random_state=42
)

classifier.fit(X_train, y_train)

# ----------------------------
# Prediction
# ----------------------------

y_pred = classifier.predict(X_test)

# ----------------------------
# Evaluation
# ----------------------------

accuracy = metrics.accuracy_score(y_test, y_pred)

print("\nAccuracy : {:.2f}%".format(accuracy*100))

print("\nClassification Report\n")
print(
    metrics.classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

# ----------------------------
# Confusion Matrix
# ----------------------------

disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=iris.target_names,
    cmap="Blues"
)

plt.title("AdaBoost Confusion Matrix")

plt.savefig(
    "static/graphs/adaboost.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion Matrix saved successfully.")
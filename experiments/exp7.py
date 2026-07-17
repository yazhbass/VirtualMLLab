import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -------------------------------------------------
# Create Graph Folder
# -------------------------------------------------

os.makedirs("static/graphs", exist_ok=True)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

dataset = pd.read_csv("heart_v2.csv")

# Remove leading/trailing spaces in column names
dataset.columns = dataset.columns.str.strip()

print("=" * 60)
print("HEART DISEASE PREDICTION USING RANDOM FOREST")
print("=" * 60)

print("\nFirst Five Records\n")
print(dataset.head())

print("\nDataset Shape :", dataset.shape)

print("\nColumn Names")
print(dataset.columns.tolist())

print("\nMissing Values\n")
print(dataset.isnull().sum())

# -------------------------------------------------
# Feature and Target
# -------------------------------------------------

X = dataset.drop("target", axis=1)
y = dataset["target"]

# -------------------------------------------------
# Train Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", X_train.shape)
print("Testing Samples  :", X_test.shape)

# -------------------------------------------------
# Random Forest Model
# -------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# -------------------------------------------------
# Train Model
# -------------------------------------------------

model.fit(X_train, y_train)

# -------------------------------------------------
# Prediction
# -------------------------------------------------

y_pred = model.predict(X_test)

# -------------------------------------------------
# Accuracy
# -------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy : {:.2f}%".format(accuracy * 100))

# -------------------------------------------------
# Classification Report
# -------------------------------------------------

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix\n")
print(cm)

fig, ax = plt.subplots(figsize=(6,5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(ax=ax, cmap="Blues")

plt.title("Random Forest Confusion Matrix")

plt.savefig(
    "static/graphs/random_forest.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion Matrix Graph Saved Successfully.")

# -------------------------------------------------
# Feature Importance
# -------------------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance\n")
print(importance)

plt.figure(figsize=(8,5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xticks(rotation=45)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")

plt.tight_layout()

plt.savefig(
    "static/graphs/feature_importance.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nFeature Importance Graph Saved Successfully.")

# -------------------------------------------------
# Final Message
# -------------------------------------------------

print("\nExperiment Completed Successfully.")
print("Graph 1 : static/graphs/random_forest.png")
print("Graph 2 : static/graphs/feature_importance.png")
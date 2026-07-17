import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Create graphs folder if it doesn't exist
os.makedirs("static/graphs", exist_ok=True)

# Load the diabetes dataset
diabetes = load_diabetes()

# Input features
X = diabetes.data

# Continuous target
y = diabetes.target

# Convert continuous target into binary classification
# Values above the median -> 1, otherwise -> 0
y_binary = np.where(y > np.median(y), 1, 0)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_binary,
    test_size=0.2,
    random_state=42
)

# Standardize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print("========== Logistic Regression Output ==========\n")

print(f"Accuracy : {accuracy:.4f}\n")

print("Confusion Matrix")
print(cm)

print("\nClassification Report")
print(report)

# Plot Confusion Matrix using Matplotlib
plt.figure(figsize=(6,5))

plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()

classes = ['Low', 'High']
tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Add values inside the matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha='center',
            va='center',
            color='black',
            fontsize=12
        )

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

# Save graph
plt.savefig("static/graphs/logistic.png")

plt.close()
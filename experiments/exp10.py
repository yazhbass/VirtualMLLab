import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)


# ---------------------------------
# Read Dataset
# ---------------------------------

def prepare_person_dataset(filename):

    dataset = []

    with open(filename, "r") as file:

        for line in file:

            person = line.strip().split()

            height = float(person[2])
            weight = float(person[3])
            gender = person[4]

            dataset.append(((height, weight), gender))

    return dataset


# ---------------------------------
# Load datasets
# ---------------------------------

learnset = prepare_person_dataset("person_data.txt")
testset = prepare_person_dataset("person_testset.txt")


# ---------------------------------
# Training Data
# ---------------------------------

X_train, y_train = zip(*learnset)

X_train = np.array(X_train)
y_train = np.array(y_train)


# ---------------------------------
# Testing Data
# ---------------------------------

X_test, y_test = zip(*testset)

X_test = np.array(X_test)
y_test = np.array(y_test)


# ---------------------------------
# Gaussian Naive Bayes
# ---------------------------------

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# ---------------------------------
# Display Results
# ---------------------------------

print("=" * 50)

print("Gaussian Naive Bayes Classifier")

print("=" * 50)

print("\nTraining Samples :", len(X_train))

print("Testing Samples  :", len(X_test))

print("\nPredicted Labels")

print(y_pred)

print("\nActual Labels")

print(y_test)

print("\nAccuracy : %.2f%%" % (accuracy_score(y_test, y_pred) * 100))

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))


# ---------------------------------
# Confusion Matrix
# ---------------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()

plt.title("Gaussian Naive Bayes Confusion Matrix")

plt.tight_layout()

plt.savefig("static/graphs/naive_bayes.png")

plt.close()

print("\nConfusion Matrix saved successfully.")

print("\nExperiment Completed Successfully.")
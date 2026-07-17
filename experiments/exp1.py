import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Create graphs folder if it doesn't exist
os.makedirs("static/graphs", exist_ok=True)

# Load the diabetes dataset
data = load_diabetes()

# Use only BMI feature
X = data.data[:, [2]]
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print output
print("========== Linear Regression Output ==========")
print(f"Slope (Coefficient): {model.coef_[0]:.4f}")
print(f"Intercept          : {model.intercept_:.4f}")
print(f"Mean Squared Error : {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score           : {r2_score(y_test, y_pred):.2f}")

# Plot
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color="black", label="Actual")
plt.plot(X_test, y_pred, color="blue", linewidth=2, label="Prediction")

plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Linear Regression on Diabetes Dataset")
plt.legend()
plt.grid(True)

# Save graph
plt.savefig("static/graphs/regression.png")
plt.close()
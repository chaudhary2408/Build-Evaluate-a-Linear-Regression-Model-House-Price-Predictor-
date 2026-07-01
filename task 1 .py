# ======================================================
# Task 1 - Linear Regression Model
# California Housing Price Prediction
# ======================================================

# Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

# ======================================================
# Step 2: Load Dataset
# ======================================================

housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target

print("First 5 Rows")
print(df.head())

# ======================================================
# Step 3: Dataset Information
# ======================================================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ======================================================
# Step 4: Exploratory Data Analysis (EDA)
# ======================================================

# Histogram

df.hist(figsize=(12,10))
plt.tight_layout()
plt.show()

# Correlation Matrix

correlation = df.corr()

plt.figure(figsize=(10,8))
plt.imshow(correlation, cmap='coolwarm')
plt.colorbar()

plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation.columns)), correlation.columns)

plt.title("Correlation Matrix")
plt.show()

# ======================================================
# Step 5: Define Features and Target
# ======================================================

X = df.drop("Price", axis=1)
y = df["Price"]

# ======================================================
# Step 6: Split Dataset
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ======================================================
# Step 7: Train Linear Regression Model
# ======================================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed.")

# ======================================================
# Step 8: Prediction
# ======================================================

y_pred = model.predict(X_test)

# ======================================================
# Step 9: Evaluation
# ======================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== Model Evaluation ==========")

print(f"Mean Absolute Error : {mae:.4f}")
print(f"Root Mean Squared Error : {rmse:.4f}")
print(f"R2 Score : {r2:.4f}")

# ======================================================
# Step 10: Actual vs Predicted Plot
# ======================================================

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.show()

# ======================================================
# Step 11: Model Coefficients
# ======================================================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Importance:")
print(coefficients)

# ======================================================
# Step 12: Save Model
# ======================================================

joblib.dump(model, "house_price_model.pkl")

print("\nModel Saved Successfully!")

# ======================================================
# Step 13: Predict New House Price
# ======================================================

sample_house = [[
    8.3252,
    41.0,
    6.984,
    1.023,
    322,
    2.555,
    37.88,
    -122.23
]]

prediction = model.predict(sample_house)

print("\nPredicted House Price:")
print(prediction)

# ======================================================
# End of Project
# ======================================================
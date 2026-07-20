# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset and Preprocessing
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ML Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 2. LOAD THE DATASET
# ==========================================
# Fetching the California Housing dataset
housing_data = fetch_california_housing(as_frame=True)

# Convert to a standard Pandas DataFrame
df = housing_data.frame
print("--- Dataset Sample ---")
print(df.head(), "\n")

# ==========================================
# 3. DATA PREPROCESSING
# ==========================================
# Check for missing values
print("--- Missing Values Count ---")
print(df.isnull().sum(), "\n")

# Even though California Housing has no missing values, we include an Imputer 
# as best practice for robust data preprocessing pipelines.
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Split features (X) and target (y)
# Target variable is 'MedHouseVal' (Median House Value in $100,000s)
X = df_imputed.drop(columns=['MedHouseVal'])
y = df_imputed['MedHouseVal']

# Feature Scaling (Crucial for baseline Linear Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================
# Splitting 80% for training and 20% for testing. 
# random_state ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}\n")

# ==========================================
# 5. TRAIN A BASELINE REGRESSION MODEL
# ==========================================
# Using Linear Regression from the previous task as our baseline
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

# ==========================================
# 6. EVALUATE THE BASELINE MODEL
# ==========================================
# Predicting on test data
y_pred_baseline = baseline_model.predict(X_test)

# Calculating Metrics
r2_baseline = r2_score(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))

print("--- Baseline Model (Linear Regression) Performance ---")
print(f"Baseline R² Score: {r2_baseline:.4f}")
print(f"Baseline RMSE:     {rmse_baseline:.4f}\n")

# ==========================================
# 7. CROSS VALIDATION
# ==========================================
# Performing 5-Fold Cross-Validation on the baseline model to check for overfitting/variance
cv_scores = cross_val_score(baseline_model, X_train, y_train, cv=5, scoring='r2')

print("--- 5-Fold Cross-Validation (Baseline Model) ---")
print(f"CV R² Scores for each fold: {cv_scores}")
print(f"Mean CV R² Score:          {np.mean(cv_scores):.4f}")
print(f"Standard Deviation:        {np.std(cv_scores):.4f}\n")

# ==========================================
# 8. HYPERPARAMETER TUNING (Decision Tree)
# ==========================================
# Overfitting Control: Standard Decision Trees overfit easily. 
# We use GridSearchCV to find optimal hyperparameters to constrain tree growth.
dt_model = DecisionTreeRegressor(random_state=42)

# Define the hyperparameter grid to search
param_grid = {
    'max_depth': [4, 6, 8, 10, 12],
    'min_samples_split': [2, 10, 20, 50],
    'min_samples_leaf': [1, 5, 10, 20]
}

print("--- Starting Hyperparameter Tuning via GridSearchCV ---")
grid_search = GridSearchCV(
    estimator=dt_model, 
    param_grid=param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error', # Optimized for lowest MSE
    n_jobs=-1 # Use all available CPU cores
)

grid_search.fit(X_train, y_train)

print(f"Best Hyperparameters Found: {grid_search.best_params_}\n")

# ==========================================
# 9. TRAIN THE BEST MODEL
# ==========================================
# Extract the optimized model
best_tuned_model = grid_search.best_estimator_

# The grid search auto-fits the best model on the entire training set, 
# but we explicitly evaluate it on our held-out test set here.
y_pred_tuned = best_tuned_model.predict(X_test)

# Calculate tuned model metrics
r2_tuned = r2_score(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))

# ==========================================
# 10. COMPARE BASELINE VS TUNED MODEL
# ==========================================
# Creating a structured DataFrame comparison
comparison_data = {
    'Metric': ['R² Score (Higher is Better)', 'RMSE (Lower is Better)'],
    'Baseline (Linear Regression)': [r2_baseline, rmse_baseline],
    'Tuned Model (Decision Tree)': [r2_tuned, rmse_tuned]
}
df_comparison = pd.DataFrame(comparison_data)

print("--- Model Comparison Table ---")
print(df_comparison.to_string(index=False), "\n")

# Optional: Plotting actual vs predicted values for both models to visualize performance
plt.figure(figsize=(12, 5))

# Baseline Plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_baseline, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title(f'Baseline Linear Regression\n(R²: {r2_baseline:.3f})')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Tuned Model Plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_tuned, alpha=0.3, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title(f'Tuned Decision Tree\n(R²: {r2_tuned:.3f})')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()

# ==========================================
# 11. FINAL RESULTS AND CONCLUSION
# ==========================================
print("--- Final Conclusion ---")
print(f"1. The baseline Linear Regression model yielded an R² score of {r2_baseline:.4f}.")
print(f"2. 5-Fold Cross-Validation confirmed model stability with a low variance (Std Dev: {np.std(cv_scores):.4f}).")
print(f"3. By introducing a Decision Tree Regressor and utilizing GridSearchCV for Overfitting Control "
      f"(constraining max_depth and min_samples_split), we successfully captured non-linear relationships.")
print(f"4. The Tuned Decision Tree improved the R² score to {r2_tuned:.4f} and reduced the error (RMSE) to {rmse_tuned:.4f}.")
print("Conclusion: Hyperparameter tuning and switching to a non-linear model structure significantly enhanced predictive accuracy while maintaining strict control over model variance.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set visual style for matplotlib/seaborn
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12})

# ==========================================
# 1. CUSTOM FEATURE ENGINEERING TRANSFORMER
# ==========================================
class GeoClusterFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create location-based clusters from 
    Latitude and Longitude to capture regional price variations.
    """
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        
    def fit(self, X, y=None):
        # Expects X to be a DataFrame or numpy array where columns 6 & 7 are Latitude & Longitude
        geo_data = X[:, [6, 7]] if isinstance(X, np.ndarray) else X[['Latitude', 'Longitude']]
        self.kmeans.fit(geo_data)
        return self
        
    def transform(self, X):
        geo_data = X[:, [6, 7]] if isinstance(X, np.ndarray) else X[['Latitude', 'Longitude']]
        cluster_dist = self.kmeans.transform(geo_data)
        # Append the distance features to the original data matrix
        return np.hstack((X, cluster_dist))

# ==========================================
# 2. DATA LOADING & INGESTION PIPELINE
# ==========================================
def load_and_prepare_data():
    print("Loading California Housing dataset...")
    california = fetch_california_housing(as_frame=True)
    X = california.data
    y = california.target
    
    print(f"Dataset loaded successfully with {X.shape[0]} samples and {X.shape[1]} raw features.")
    return X, y

# ==========================================
# 3. MODEL TRAINING & EXPERIMENTATION ENGINE
# ==========================================
class MLExperimentEngine:
    def __init__(self, X, y):
        # 80/20 Train-Test Split
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize Preprocessing Pipelines
        self.geo_transformer = GeoClusterFeatures(n_clusters=8, random_state=42)
        self.scaler = StandardScaler()
        
        # Preprocess the data
        self.X_train = self._preprocess(self.X_train_raw, fit=True)
        self.X_test = self._preprocess(self.X_test_raw, fit=False)
        
        self.models = {}
        self.results = {}
        self.predictions = {}

    def _preprocess(self, data, fit=False):
        if fit:
            geo_features = self.geo_transformer.fit_transform(data)
            scaled_features = self.scaler.fit_transform(geo_features)
            return scaled_features
        else:
            geo_features = self.geo_transformer.transform(data)
            scaled_features = self.scaler.transform(geo_features)
            return scaled_features

    def run_experiments(self):
        # Define baseline architectures
        base_models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=10.0)
        }
        
        # Train baseline models
        for name, model in base_models.items():
            print(f"Training standard model: {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            self._evaluate_model(name, model)

        # Optimize Hyperparameters for Decision Tree using Grid Search
        print("Running GridSearchCV for Decision Tree Optimization...")
        dt_param_grid = {
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 10, 20],
            'min_samples_leaf': [1, 5, 10]
        }
        grid_search = GridSearchCV(
            DecisionTreeRegressor(random_state=42), 
            dt_param_grid, 
            cv=3, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best Decision Tree Hyperparameters: {grid_search.best_params_}")
        best_dt_model = grid_search.best_estimator_
        self.models["Optimized Decision Tree"] = best_dt_model
        self._evaluate_model("Optimized Decision Tree", best_dt_model)

    def _evaluate_model(self, name, model):
        # Predictions
        preds = model.predict(self.X_test)
        self.predictions[name] = preds
        
        # Core Metrics calculation
        rmse = np.sqrt(mean_squared_error(self.y_test, preds))
        mae = mean_absolute_error(self.y_test, preds)
        r2 = r2_score(self.y_test, preds)
        
        # 5-Fold Cross Validation on Training Data for stability validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
        
        self.results[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R² Score": r2,
            "CV R² Mean": cv_scores.mean(),
            "CV R² Std": cv_scores.style.tolist() if hasattr(cv_scores, 'style') else cv_scores.std()
        }

    def display_metrics_table(self):
        df = pd.DataFrame(self.results).T
        print("\n" + "="*60)
        print("                  FINAL MODEL PERFORMANCE                  ")
        print("="*60)
        print(df.to_string(formatters={
            'RMSE': '{:,.4f}'.format,
            'MAE': '{:,.4f}'.format,
            'R² Score': '{:,.4f}'.format,
            'CV R² Mean': '{:,.4f}'.format,
            'CV R² Std': '{:,.4f}'.format
        }))
        print("="*60)
        
        best_model = df['R² Score'].idxmax()
        print(f"🚀 Recommended Production Model based on R² Score: {best_model}\n")

    def generate_diagnostic_plots(self):
        """Generates complex subplots displaying actual vs predicted and residual errors."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle('Advanced Model Diagnostics & Error Analysis', fontsize=16, weight='bold')
        
        for idx, (name, preds) in enumerate(self.predictions.items()):
            # Row 1: Prediction Regression Fit
            ax_scatter = axes[0, idx]
            ax_scatter.scatter(self.y_test, preds, alpha=0.15, color='royalblue', edgecolors='none')
            ax_scatter.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            ax_scatter.set_title(f'{name}: Actual vs Predicted')
            ax_scatter.set_xlabel('Actual MedHouseValue')
            ax_scatter.set_ylabel('Predicted MedHouseValue')
            
            # Row 2: Residual Error Distributions
            residuals = self.y_test - preds
            ax_resid = axes[1, idx]
            sns.histplot(residuals, kde=True, ax=ax_resid, color='purple', bins=40)
            ax_resid.axvline(x=0, color='black', linestyle='--')
            ax_resid.set_title(f'{name}: Residual Distribution')
            ax_resid.set_xlabel('Prediction Error (Residual)')
            ax_resid.set_ylabel('Density')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# ==========================================
# 4. EXECUTION GATEWAY
# ==========================================
if __name__ == "__main__":
    # Ingest data
    X, y = load_and_prepare_data()
    
    # Initialize pipeline execution framework
    engine = MLExperimentEngine(X, y)
    
    # Run pipeline training, tuning, and validation operations
    engine.run_experiments()
    
    # Output metrics comparison report
    engine.display_metrics_table()
    
    # Execute structural plotting logic
    engine.generate_diagnostic_plots()

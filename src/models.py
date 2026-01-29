"""
Model definitions and training functions.

Implements three ML models for demand forecasting:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
"""

import numpy as np
import random
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import json

# Try to import XGBoost (optional dependency)
# Handle both ImportError and runtime errors (e.g., missing OpenMP on macOS)
XGBOOST_AVAILABLE = False
xgb = None
try:
    import xgboost as xgb
    # Test if XGBoost actually works by trying to access its core module
    # This will fail if OpenMP is missing
    _ = xgb.__version__
    XGBOOST_AVAILABLE = True
except Exception:
    # XGBoost not available (not installed or missing dependencies like OpenMP)
    XGBOOST_AVAILABLE = False
    xgb = None


RANDOM_STATE = 42

# Ensure deterministic behaviour across Python and NumPy
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Clean features to prevent numerical warnings.
    
    Handles NaN, Inf, and extreme values that can cause numerical instability.
    
    Args:
        X: Feature DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    X = X.copy()
    
    # Replace infinite values with NaN first, then fill
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with 0 (for features, 0 is a safe default)
    X = X.fillna(0)
    
    # Clip extreme values to prevent overflow (use reasonable bounds)
    # For most features, values beyond ±1e6 are likely errors
    X = X.clip(lower=-1e6, upper=1e6)
    
    return X


def scale_features(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], StandardScaler]:
    """
    Scale features using StandardScaler, fit on train only.

    Prevents data leakage by fitting scaler only on training data.
    Test data is transformed using the scaler fitted on training data.
    Includes data cleaning to prevent numerical warnings.

    Args:
        X_train: Training features
        X_test: Test features (optional)
        scaler: Pre-fitted scaler (optional, if None creates new one)

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
        X_test_scaled will be None if X_test is None
    """
    # Clean features before scaling
    X_train = clean_features(X_train)
    if X_test is not None:
        X_test = clean_features(X_test)
    
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
    else:
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

    # Clean scaled features (handle any Inf/NaN from scaling)
    X_train_scaled = clean_features(X_train_scaled)
    if X_test_scaled is not None:
        X_test_scaled = clean_features(X_test_scaled)

    return X_train_scaled, X_test_scaled, scaler


def train_naive_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_df: pd.DataFrame = None,
) -> Dict[str, Any]:
    """
    Train naive baseline models for comparison.

    Creates three baseline models:
    - last_value: Uses last known value per product
    - mean: Uses mean historical demand per product
    - seasonal_naive: Uses value from 7 days ago (same day of week)

    Args:
        X_train: Training features (not used, kept for API consistency)
        y_train: Training target values
        train_df: Training DataFrame with product_line and date columns

    Returns:
        Dictionary with baseline model names as keys and model objects as values
    """
    baselines = {}
    
    # Last value baseline
    class LastValueBaseline:
        def __init__(self, y_train, train_df):
            self.y_train = y_train
            self.train_df = train_df
            self.last_values = {}
            if train_df is not None and "product_line" in train_df.columns:
                for product in train_df["product_line"].unique():
                    product_mask = train_df["product_line"] == product
                    # Use train_df index to get corresponding y_train values
                    product_indices = train_df[product_mask].index
                    product_values = y_train.loc[product_indices] if hasattr(y_train, 'loc') else y_train[product_mask]
                    self.last_values[product] = product_values.iloc[-1] if len(product_values) > 0 else 0.0
                self.global_last = y_train.iloc[-1] if len(y_train) > 0 else 0.0
            else:
                self.global_last = y_train.iloc[-1] if len(y_train) > 0 else 0.0
        
        def predict(self, X_test, test_df=None):
            if test_df is not None and "product_line" in test_df.columns:
                return np.array([self.last_values.get(p, self.global_last) for p in test_df["product_line"]])
            return np.full(len(X_test), self.global_last)
    
    # Mean baseline
    class MeanBaseline:
        def __init__(self, y_train, train_df):
            self.y_train = y_train
            self.train_df = train_df
            self.means = {}
            if train_df is not None and "product_line" in train_df.columns:
                for product in train_df["product_line"].unique():
                    product_mask = train_df["product_line"] == product
                    # Use train_df index to get corresponding y_train values
                    product_indices = train_df[product_mask].index
                    product_values = y_train.loc[product_indices] if hasattr(y_train, 'loc') else y_train[product_mask]
                    self.means[product] = product_values.mean() if len(product_values) > 0 else 0.0
                self.global_mean = y_train.mean() if len(y_train) > 0 else 0.0
            else:
                self.global_mean = y_train.mean() if len(y_train) > 0 else 0.0
        
        def predict(self, X_test, test_df=None):
            if test_df is not None and "product_line" in test_df.columns:
                return np.array([self.means.get(p, self.global_mean) for p in test_df["product_line"]])
            return np.full(len(X_test), self.global_mean)
    
    # Seasonal naive baseline (7-day)
    class SeasonalNaiveBaseline:
        def __init__(self, y_train, train_df):
            self.y_train = y_train
            self.train_df = train_df
            # Store historical values by (product, day_of_week) for seasonal matching
            self.seasonal_values = {}  # {(product, day_of_week): value}
            self.last_values = {}  # Fallback: last value per product
            self.global_last = y_train.iloc[-1] if len(y_train) > 0 else 0.0
            
            if train_df is not None:
                # Get date information from index or date column
                if isinstance(train_df.index, pd.DatetimeIndex):
                    dates = train_df.index
                elif "date" in train_df.columns:
                    dates = pd.to_datetime(train_df["date"])
                elif "datetime" in train_df.columns:
                    dates = pd.to_datetime(train_df["datetime"])
                else:
                    dates = None
                
                # Get product identifier
                if "product_line" in train_df.columns:
                    product_col = "product_line"
                elif "Product ID" in train_df.columns and "Category" in train_df.columns:
                    # Use composite key
                    product_col = None
                else:
                    product_col = None
                
                if dates is not None and product_col is not None:
                    # Build seasonal lookup: (product, day_of_week) -> most recent value
                    train_df_copy = train_df.copy()
                    train_df_copy["_date"] = dates
                    train_df_copy["_day_of_week"] = train_df_copy["_date"].dt.dayofweek
                    train_df_copy["_y"] = y_train.values if hasattr(y_train, 'values') else y_train
                    
                    for product in train_df_copy[product_col].unique():
                        product_mask = train_df_copy[product_col] == product
                        product_data = train_df_copy[product_mask].copy()
                        product_data = product_data.sort_values("_date")
                        
                        # Store last value per product (fallback)
                        if len(product_data) > 0:
                            self.last_values[product] = product_data["_y"].iloc[-1]
                        
                        # Store most recent value for each day of week
                        for day_of_week in range(7):
                            day_mask = product_data["_day_of_week"] == day_of_week
                            day_data = product_data[day_mask]
                            if len(day_data) > 0:
                                # Use most recent value for this day of week
                                self.seasonal_values[(product, day_of_week)] = day_data["_y"].iloc[-1]
                elif product_col is not None:
                    # No date info, fall back to last value per product
                    for product in train_df[product_col].unique():
                        product_mask = train_df[product_col] == product
                        product_indices = train_df[product_mask].index
                        product_values = y_train.loc[product_indices] if hasattr(y_train, 'loc') else y_train[product_mask]
                        self.last_values[product] = product_values.iloc[-1] if len(product_values) > 0 else 0.0
        
        def predict(self, X_test, test_df=None):
            """Predict using seasonal naive: value from 7 days ago, same day of week."""
            if test_df is not None:
                # Get date information from test_df
                if isinstance(test_df.index, pd.DatetimeIndex):
                    test_dates = test_df.index
                elif "date" in test_df.columns:
                    test_dates = pd.to_datetime(test_df["date"])
                elif "datetime" in test_df.columns:
                    test_dates = pd.to_datetime(test_df["datetime"])
                else:
                    test_dates = None
                
                # Get product identifier
                if "product_line" in test_df.columns:
                    product_col = "product_line"
                    products = test_df[product_col].values
                elif "Product ID" in test_df.columns and "Category" in test_df.columns:
                    # Use composite key
                    products = (test_df["Product ID"].astype(str) + "_" + test_df["Category"].astype(str)).values
                    product_col = None
                else:
                    products = None
                    product_col = None
                
                if test_dates is not None and products is not None:
                    # Use seasonal matching: same day of week
                    test_day_of_week = test_dates.dayofweek.values
                    predictions = []
                    for i, (product, day_of_week) in enumerate(zip(products, test_day_of_week)):
                        # Try seasonal match first
                        key = (product, day_of_week)
                        if key in self.seasonal_values:
                            predictions.append(self.seasonal_values[key])
                        elif product in self.last_values:
                            # Fallback to last value for this product
                            predictions.append(self.last_values[product])
                        else:
                            # Final fallback to global last
                            predictions.append(self.global_last)
                    return np.array(predictions)
                elif products is not None:
                    # No date info, use last value per product
                    return np.array([self.last_values.get(p, self.global_last) for p in products])
            
            # No test_df or product info, return global last
            return np.full(len(X_test), self.global_last)
    
    baselines["baseline_last_value"] = LastValueBaseline(y_train, train_df)
    baselines["baseline_mean"] = MeanBaseline(y_train, train_df)
    baselines["baseline_seasonal_naive"] = SeasonalNaiveBaseline(y_train, train_df)
    
    return baselines


def prepare_features_and_target(
    df: pd.DataFrame, target_col: str = "quantity"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and target vector from DataFrame.

    Separates features from target variable and handles missing values.
    Validates that Weather, Seasonality, and Region features are included.
    
    Note: Demand Forecast is explicitly excluded to prevent data leakage -
    it's a pre-computed target variable and should not be used as a feature.

    Args:
        df: DataFrame with features and target
        target_col: Name of target column

    Returns:
        Tuple of (X, y) where X is features and y is target

    Raises:
        ValueError: If Weather, Seasonality, or Region features are missing
    """
    # Drop non-feature columns (target, product identifiers, current_stock, and Demand Forecast)
    # current_stock is used for recommendations, not for model features
    # Demand Forecast is excluded to prevent data leakage (it's a pre-computed target)
    exclude_cols = [
        target_col, "revenue", "current_stock", "Product ID", "Category", 
        "Store ID", "Region", "Weather Condition", "Seasonality", "Demand Forecast"
    ]
    
    # Also exclude original categorical columns (we use one-hot encoded versions)
    feature_cols = [
        col
        for col in df.columns
        if col not in exclude_cols
    ]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Validate that Weather, Seasonality, Region features are present
    weather_cols = [col for col in X.columns if "weather_condition" in col.lower()]
    seasonality_cols = [col for col in X.columns if "seasonality" in col.lower()]
    region_cols = [col for col in X.columns if "region" in col.lower()]
    
    if not weather_cols:
        raise ValueError(
            "Weather Condition features not found in feature matrix. "
            "Ensure Weather Condition is one-hot encoded."
        )
    if not seasonality_cols:
        raise ValueError(
            "Seasonality features not found in feature matrix. "
            "Ensure Seasonality is one-hot encoded."
        )
    if not region_cols:
        raise ValueError(
            "Region features not found in feature matrix. "
            "Ensure Region is one-hot encoded."
        )

    # Clean features: handle NaN, Inf, and extreme values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    # Clip extreme values to prevent numerical overflow
    X = X.clip(lower=-1e6, upper=1e6)

    # Ensure y doesn't have NaN or Inf values
    y = y.replace([np.inf, -np.inf], np.nan)
    if y.isna().any():
        # Fill NaN values with 0 (for quantity, 0 is reasonable)
        y = y.fillna(0)
    # Clip y to reasonable bounds (negative quantities don't make sense)
    y = y.clip(lower=0, upper=1e6)

    return X, y


def prepare_features_and_target_revenue(
    df: pd.DataFrame, target_col: str = "revenue"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and revenue target vector from DataFrame.

    Separates features from revenue target variable and handles missing values.
    Validates that Weather, Seasonality, and Region features are included.

    Args:
        df: DataFrame with features and revenue target
        target_col: Name of revenue target column (default: "revenue")

    Returns:
        Tuple of (X, y) where X is features and y is revenue target

    Raises:
        ValueError: If Weather, Seasonality, or Region features are missing
        ValueError: If revenue column doesn't exist
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Revenue target column '{target_col}' not found in DataFrame. "
            "Ensure revenue is calculated (Price × Units Sold)."
        )
    
    # Use same feature preparation as quantity (validates Weather/Seasonality/Region)
    X, y = prepare_features_and_target(df, target_col=target_col)

    return X, y


def train_linear_regression(
    X_train: pd.DataFrame, y_train: pd.Series, use_ridge: bool = True
) -> Tuple[Any, StandardScaler]:
    """
    Train Linear Regression model for demand forecasting.

    Uses Ridge regression by default for numerical stability.
    Features are automatically scaled before training.

    Args:
        X_train: Training features (will be scaled)
        y_train: Training target
        use_ridge: If True, use Ridge regression (L2 regularization)

    Returns:
        Tuple of (trained model, fitted scaler)
    """
    # Scale features
    X_train_scaled, _, scaler = scale_features(X_train)

    # Use Ridge for numerical stability (L2 regularization)
    if use_ridge:
        model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    else:
        model = LinearRegression()

    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 10,
) -> Tuple[RandomForestRegressor, StandardScaler]:
    """
    Train Random Forest Regressor for demand forecasting.

    Random Forest captures non-linear relationships and interactions
    between features without overfitting. Features are scaled for consistency.

    Args:
        X_train: Training features (will be scaled)
        y_train: Training target
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees

    Returns:
        Tuple of (trained model, fitted scaler)
    """
    # Scale features (RF is scale-invariant but we scale for consistency)
    X_train_scaled, _, scaler = scale_features(X_train)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 5,
) -> Tuple[GradientBoostingRegressor, StandardScaler]:
    """
    Train Gradient Boosting Regressor for demand forecasting.

    Gradient Boosting sequentially improves predictions by learning
    from previous errors, often achieving best performance.
    Features are scaled for consistency.

    Args:
        X_train: Training features (will be scaled)
        y_train: Training target
        n_estimators: Number of boosting stages
        learning_rate: Learning rate for each tree
        max_depth: Maximum depth of trees

    Returns:
        Tuple of (trained model, fitted scaler)
    """
    # Scale features (GB is scale-invariant but we scale for consistency)
    X_train_scaled, _, scaler = scale_features(X_train)

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    cv_splits: int = 3,
    verbose: bool = False,
) -> Tuple[RandomForestRegressor, StandardScaler, dict]:
    """
    Tune Random Forest hyperparameters using RandomizedSearchCV.
    
    Uses time-series cross-validation to find optimal hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_iter: Number of parameter settings sampled
        cv_splits: Number of cross-validation splits
        verbose: Whether to print progress
        
    Returns:
        Tuple of (best_model, scaler, best_parameters)
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    
    # Parameter distribution for RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5, 0.7]
    }
    
    # Base model
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    # Randomized search
    search = RandomizedSearchCV(
        rf,
        param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1 if verbose else 0
    )
    
    if verbose:
        print(f"  Tuning Random Forest with {n_iter} iterations...")
    search.fit(X_train_scaled, y_train)
    
    if verbose:
        print(f"  Best score: {-search.best_score_:.4f} MAE")
        print(f"  Best parameters: {search.best_params_}")
    
    return search.best_estimator_, scaler, search.best_params_


def tune_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    cv_splits: int = 3,
    verbose: bool = False,
) -> Tuple[GradientBoostingRegressor, StandardScaler, dict]:
    """
    Tune Gradient Boosting hyperparameters using RandomizedSearchCV.
    
    Uses time-series cross-validation to find optimal hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_iter: Number of parameter settings sampled
        cv_splits: Number of cross-validation splits
        verbose: Whether to print progress
        
    Returns:
        Tuple of (best_model, scaler, best_parameters)
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    
    # Parameter distribution for RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Base model
    gb = GradientBoostingRegressor(random_state=RANDOM_STATE)
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    # Randomized search
    search = RandomizedSearchCV(
        gb,
        param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1 if verbose else 0
    )
    
    if verbose:
        print(f"  Tuning Gradient Boosting with {n_iter} iterations...")
    search.fit(X_train_scaled, y_train)
    
    if verbose:
        print(f"  Best score: {-search.best_score_:.4f} MAE")
        print(f"  Best parameters: {search.best_params_}")
    
    return search.best_estimator_, scaler, search.best_params_


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    cv_splits: int = 3,
    verbose: bool = False,
) -> Tuple[Any, StandardScaler, dict]:
    """
    Tune XGBoost hyperparameters using RandomizedSearchCV.
    
    Uses time-series cross-validation to find optimal hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_iter: Number of parameter settings sampled
        cv_splits: Number of cross-validation splits
        verbose: Whether to print progress
        
    Returns:
        Tuple of (best_model, scaler, best_parameters)
        
    Raises:
        ImportError: If xgboost is not installed
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    
    # Parameter distribution for RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Base model
    xgb_model = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    # Randomized search
    search = RandomizedSearchCV(
        xgb_model,
        param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1 if verbose else 0
    )
    
    if verbose:
        print(f"  Tuning XGBoost with {n_iter} iterations...")
    search.fit(X_train_scaled, y_train)
    
    if verbose:
        print(f"  Best score: {-search.best_score_:.4f} MAE")
        print(f"  Best parameters: {search.best_params_}")
    
    return search.best_estimator_, scaler, search.best_params_


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 6,
) -> Tuple[Any, StandardScaler]:
    """
    Train XGBoost model for demand forecasting.
    
    XGBoost is often the best-performing model for tabular data.
    Features are automatically scaled before training.
    
    Args:
        X_train: Training features (will be scaled)
        y_train: Training target
        n_estimators: Number of boosting rounds
        learning_rate: Learning rate for each tree
        max_depth: Maximum depth of trees
        
    Returns:
        Tuple of (trained model, fitted scaler)
        
    Raises:
        ImportError: If xgboost is not installed
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    return model, scaler


def save_hyperparameters(
    hyperparameters: Dict[str, dict],
    save_path: str = "results/metrics/best_hyperparameters.json"
):
    """
    Save best hyperparameters to JSON file for reproducibility.
    
    Args:
        hyperparameters: Dictionary mapping model names to their best parameters
        save_path: Path to save the JSON file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(hyperparameters, f, indent=2)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scaler: Optional[StandardScaler] = None,
    cost_config=None,
) -> Dict[str, float]:
    """
    Evaluate model performance using MAE and RMSE.

    MAE provides average prediction error, RMSE penalizes larger errors.
    Optionally includes economic costs.

    Args:
        model: Trained model
        X_test: Test features (will be scaled if scaler provided)
        y_test: Test target
        scaler: Fitted scaler to transform test features
        cost_config: CostConfig instance for economic cost calculation (optional)

    Returns:
        Dictionary with MAE, RMSE, and optionally economic costs
    """
    if scaler is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
        X_test = X_test_scaled

    y_pred = model.predict(X_test)
    y_pred_series = pd.Series(y_pred, index=y_test.index).clip(lower=0)

    # Use calculate_metrics to get MAE and RMSE
    from src.evaluation import calculate_metrics
    metrics = calculate_metrics(y_test, y_pred_series)
    
    # Add economic costs if cost_config provided
    if cost_config is not None:
        from src.evaluation import calculate_economic_cost
        cost_metrics = calculate_economic_cost(y_test, y_pred_series, cost_config)
        metrics.update(cost_metrics)

    return metrics


def save_model(model, model_name: str, save_dir: str = "models/trained/", verbose: bool = False):
    """
    Save trained model to disk for future use.

    Args:
        model: Trained model to save
        model_name: Name for the saved model file
        save_dir: Directory to save model
        verbose: Whether to print save confirmation
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_file = save_path / f"{model_name}.joblib"
    joblib.dump(model, model_file)
    if verbose:
        print(f"Model saved to {model_file}")


def load_model(model_name: str, model_dir: str = "models/trained/"):
    """
    Load a saved model from disk.

    Args:
        model_name: Name of the model file
        model_dir: Directory containing saved models

    Returns:
        Loaded model
    """
    model_path = Path(model_dir) / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    return model


def train_revenue_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    save_models: bool = True,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, StandardScaler]]:
    """
    Train all three models for revenue forecasting.

    Trains Linear Regression, Random Forest, and Gradient Boosting
    models for revenue prediction. All models use scaled features.

    Args:
        X_train: Training features (must include Weather/Seasonality/Region)
        y_train: Training revenue target
        save_models: Whether to save models to disk
        verbose: Whether to print progress

    Returns:
        Tuple of (models_dict, scalers_dict) where models_dict has model names as keys
        and trained models as values, and scalers_dict has model names as keys and
        fitted scalers as values
    """
    models = {}
    scalers = {}

    if verbose:
        print("Training Linear Regression for revenue...")
    model, scaler = train_linear_regression(X_train, y_train)
    models["linear_regression_revenue"] = model
    scalers["linear_regression_revenue"] = scaler
    if save_models:
        save_model(model, "linear_regression_revenue", verbose=verbose)
        scaler_path = Path("models/trained/linear_regression_revenue_scaler.joblib")
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    if verbose:
        print("Training Random Forest for revenue...")
    model, scaler = train_random_forest(X_train, y_train)
    models["random_forest_revenue"] = model
    scalers["random_forest_revenue"] = scaler
    if save_models:
        save_model(model, "random_forest_revenue", verbose=verbose)
        scaler_path = Path("models/trained/random_forest_revenue_scaler.joblib")
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    if verbose:
        print("Training Gradient Boosting for revenue...")
    model, scaler = train_gradient_boosting(X_train, y_train)
    models["gradient_boosting_revenue"] = model
    scalers["gradient_boosting_revenue"] = scaler
    if save_models:
        save_model(model, "gradient_boosting_revenue", verbose=verbose)
        scaler_path = Path("models/trained/gradient_boosting_revenue_scaler.joblib")
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return models, scalers


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_df: pd.DataFrame = None,
    save_models: bool = True,
    include_baselines: bool = True,
    use_tuning: bool = False,
    include_xgboost: bool = True,
    tuning_n_iter: int = 30,
    tuning_cv_splits: int = 3,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, StandardScaler]]:
    """
    Train all models and return them in a dictionary.

    Trains Linear Regression, Random Forest, Gradient Boosting, and optionally XGBoost
    models for comparison. All models use scaled features.
    Optionally includes naive baselines and hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training target
        train_df: Training DataFrame with product_line column (for baselines)
        save_models: Whether to save models to disk
        include_baselines: Whether to include naive baseline models
        use_tuning: If True, use hyperparameter tuning for tree-based models
        include_xgboost: If True, include XGBoost model (requires xgboost package)
        tuning_n_iter: Number of iterations for RandomizedSearchCV (if use_tuning=True)
        tuning_cv_splits: Number of CV splits for tuning (if use_tuning=True)
        verbose: Whether to print progress

    Returns:
        Tuple of (models_dict, scalers_dict) where models_dict has model names as keys
        and trained models as values, and scalers_dict has model names as keys and
        fitted scalers as values (None for baselines)
    """
    models = {}
    scalers = {}
    best_hyperparameters = {}
    
    # Train naive baselines first (no scaling needed)
    if include_baselines:
        if verbose:
            print("Training naive baselines...")
        baseline_models = train_naive_baselines(X_train, y_train, train_df)
        models.update(baseline_models)
        # Baselines don't need scalers
        for baseline_name in baseline_models.keys():
            scalers[baseline_name] = None

    # Linear Regression (no tuning - uses Ridge with fixed alpha)
    if verbose:
        print("Training Linear Regression...")
    model, scaler = train_linear_regression(X_train, y_train)
    models["linear_regression"] = model
    scalers["linear_regression"] = scaler
    if save_models:
        save_model(model, "linear_regression", verbose=verbose)
        scaler_path = Path("models/trained/linear_regression_scaler.joblib")
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    # Random Forest (with or without tuning)
    if use_tuning:
        if verbose:
            print("Training Random Forest (with hyperparameter tuning)...")
        model, scaler, best_params = tune_random_forest(
            X_train, y_train, n_iter=tuning_n_iter, cv_splits=tuning_cv_splits, verbose=verbose
        )
        best_hyperparameters["random_forest"] = best_params
    else:
        if verbose:
            print("Training Random Forest...")
        model, scaler = train_random_forest(X_train, y_train)
    
    models["random_forest"] = model
    scalers["random_forest"] = scaler
    if save_models:
        save_model(model, "random_forest", verbose=verbose)
        scaler_path = Path("models/trained/random_forest_scaler.joblib")
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    # Gradient Boosting (with or without tuning)
    if use_tuning:
        if verbose:
            print("Training Gradient Boosting (with hyperparameter tuning)...")
        model, scaler, best_params = tune_gradient_boosting(
            X_train, y_train, n_iter=tuning_n_iter, cv_splits=tuning_cv_splits, verbose=verbose
        )
        best_hyperparameters["gradient_boosting"] = best_params
    else:
        if verbose:
            print("Training Gradient Boosting...")
        model, scaler = train_gradient_boosting(X_train, y_train)
    
    models["gradient_boosting"] = model
    scalers["gradient_boosting"] = scaler
    if save_models:
        save_model(model, "gradient_boosting", verbose=verbose)
        scaler_path = Path("models/trained/gradient_boosting_scaler.joblib")
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    # XGBoost (optional, with or without tuning)
    if include_xgboost:
        if not XGBOOST_AVAILABLE:
            if verbose:
                print("Warning: XGBoost not available. Install with: pip install xgboost")
        else:
            if use_tuning:
                if verbose:
                    print("Training XGBoost (with hyperparameter tuning)...")
                try:
                    model, scaler, best_params = tune_xgboost(
                        X_train, y_train, n_iter=tuning_n_iter, cv_splits=tuning_cv_splits, verbose=verbose
                    )
                    best_hyperparameters["xgboost"] = best_params
                except Exception as e:
                    if verbose:
                        print(f"Warning: XGBoost tuning failed: {e}. Using default parameters.")
                    model, scaler = train_xgboost(X_train, y_train)
            else:
                if verbose:
                    print("Training XGBoost...")
                model, scaler = train_xgboost(X_train, y_train)
            
            models["xgboost"] = model
            scalers["xgboost"] = scaler
            if save_models:
                save_model(model, "xgboost", verbose=verbose)
                scaler_path = Path("models/trained/xgboost_scaler.joblib")
                scaler_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(scaler, scaler_path)

    # Save hyperparameters if tuning was used
    if best_hyperparameters and save_models:
        save_hyperparameters(best_hyperparameters)

    return models, scalers


def compare_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scalers: Optional[Dict[str, StandardScaler]] = None,
    test_df: pd.DataFrame = None,
    cost_config=None,
) -> pd.DataFrame:
    """
    Compare performance of all models.

    Evaluates all models on test set and returns comparison metrics.
    Handles both ML models (with scalers) and baseline models (without scalers).
    Optionally includes economic costs.

    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test target
        scalers: Dictionary of fitted scalers (optional, keys should match models)
        test_df: Test DataFrame with product_line column (for baselines)
        cost_config: CostConfig instance for economic cost calculation (optional)

    Returns:
        DataFrame with model comparison metrics
    """
    results = []

    for model_name, model in models.items():
        # Baselines don't use scalers
        if model_name.startswith("baseline_"):
            y_pred = model.predict(X_test, test_df)
            y_pred_series = pd.Series(y_pred, index=y_test.index).clip(lower=0)
            # Use calculate_metrics to get MAE and RMSE
            from src.evaluation import calculate_metrics
            metrics = calculate_metrics(y_test, y_pred_series)
            
            # Add economic costs if cost_config provided
            if cost_config is not None:
                from src.evaluation import calculate_economic_cost
                cost_metrics = calculate_economic_cost(y_test, y_pred_series, cost_config)
                metrics.update(cost_metrics)
        else:
            scaler = scalers.get(model_name) if scalers else None
            metrics = evaluate_model(model, X_test, y_test, scaler=scaler, cost_config=cost_config)
        
        metrics["Model"] = model_name
        results.append(metrics)

    comparison_df = pd.DataFrame(results)
    # Include economic costs in output if available
    base_cols = ["Model", "MAE", "RMSE"]
    if cost_config is not None:
        cost_cols = ["total_cost", "holding_cost", "stockout_cost"]
        comparison_df = comparison_df[base_cols + cost_cols]
    else:
        comparison_df = comparison_df[base_cols]

    return comparison_df


def cross_validate_model(
    model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> Dict[str, float]:
    """
    Perform time-series cross-validation on a model.

    Uses TimeSeriesSplit to respect temporal order and avoid data leakage.
    This provides more robust performance estimates than a single train/test split.

    Args:
        model: Model instance to cross-validate (not yet trained)
        X: Feature matrix
        y: Target vector
        n_splits: Number of cross-validation folds

    Returns:
        Dictionary with mean and std of MAE, RMSE, and R2 across folds
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for cross-validation.")
    if len(X) <= n_splits:
        raise ValueError(
            f"Not enough samples ({len(X)}) for {n_splits} TimeSeriesSplit folds."
        )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    mae_scores = []
    rmse_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Scale features: fit scaler on training fold, transform validation fold
        X_train_fold_scaled, X_val_fold_scaled, _ = scale_features(X_train_fold, X_val_fold)

        # Create new model instance for this fold
        if isinstance(model, (LinearRegression, Ridge)):
            # Use Ridge for numerical stability
            model_fold = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        elif isinstance(model, RandomForestRegressor):
            model_fold = RandomForestRegressor(
                n_estimators=model.n_estimators,
                max_depth=model.max_depth,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        elif isinstance(model, GradientBoostingRegressor):
            model_fold = GradientBoostingRegressor(
                n_estimators=model.n_estimators,
                learning_rate=model.learning_rate,
                max_depth=model.max_depth,
                random_state=RANDOM_STATE,
            )
        elif XGBOOST_AVAILABLE and isinstance(model, xgb.XGBRegressor):
            model_fold = xgb.XGBRegressor(
                n_estimators=model.n_estimators,
                learning_rate=model.learning_rate,
                max_depth=model.max_depth,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        else:
            # Fallback: try to clone with same parameters
            from sklearn.base import clone
            model_fold = clone(model)
            if hasattr(model_fold, "random_state"):
                model_fold.random_state = RANDOM_STATE

        # Train model on scaled training fold
        model_fold.fit(X_train_fold_scaled, y_train_fold)

        # Evaluate on scaled validation fold
        y_pred_fold = model_fold.predict(X_val_fold_scaled)
        mae_scores.append(mean_absolute_error(y_val_fold, y_pred_fold))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred_fold)))

    return {
        "MAE_mean": np.mean(mae_scores),
        "MAE_std": np.std(mae_scores),
        "RMSE_mean": np.mean(rmse_scores),
        "RMSE_std": np.std(rmse_scores),
    }


def walk_forward_validate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: float = 0.2,
) -> Dict[str, float]:
    """
    Perform walk-forward validation on a model.

    Uses expanding window: each fold adds more training data.
    Test set always comes after training set chronologically.
    This simulates real-world forecasting where we predict future using past.

    Args:
        model: Model instance to validate (not yet trained)
        X: Feature matrix (should be sorted chronologically)
        y: Target vector (should be sorted chronologically)
        n_splits: Number of validation folds
        test_size: Proportion of data to use for final test (from end)

    Returns:
        Dictionary with mean and std of MAE, RMSE, and R2 across folds
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for walk-forward validation.")
    
    # Reserve final test_size for final test, use rest for walk-forward
    total_size = len(X)
    final_test_start = int(total_size * (1 - test_size))
    X_walk = X.iloc[:final_test_start]
    y_walk = y.iloc[:final_test_start]
    
    if len(X_walk) <= n_splits:
        raise ValueError(
            f"Not enough samples ({len(X_walk)}) for {n_splits} walk-forward folds."
        )
    
    # Calculate fold sizes
    fold_size = len(X_walk) // (n_splits + 1)  # +1 to ensure we have enough data
    
    mae_scores = []
    rmse_scores = []
    
    for i in range(n_splits):
        # Expanding window: each fold uses more training data
        train_end = fold_size * (i + 1)
        test_start = train_end
        test_end = min(train_end + fold_size, len(X_walk))
        
        if test_start >= len(X_walk):
            break
        
        X_train_fold = X_walk.iloc[:train_end]
        X_test_fold = X_walk.iloc[test_start:test_end]
        y_train_fold = y_walk.iloc[:train_end]
        y_test_fold = y_walk.iloc[test_start:test_end]
        
        # Scale features: fit on training fold, transform test fold
        X_train_fold_scaled, X_test_fold_scaled, _ = scale_features(X_train_fold, X_test_fold)
        
        # Create new model instance for this fold
        if isinstance(model, (LinearRegression, Ridge)):
            model_fold = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        elif isinstance(model, RandomForestRegressor):
            model_fold = RandomForestRegressor(
                n_estimators=model.n_estimators,
                max_depth=model.max_depth,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        elif isinstance(model, GradientBoostingRegressor):
            model_fold = GradientBoostingRegressor(
                n_estimators=model.n_estimators,
                learning_rate=model.learning_rate,
                max_depth=model.max_depth,
                random_state=RANDOM_STATE,
            )
        elif XGBOOST_AVAILABLE and isinstance(model, xgb.XGBRegressor):
            model_fold = xgb.XGBRegressor(
                n_estimators=model.n_estimators,
                learning_rate=model.learning_rate,
                max_depth=model.max_depth,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        else:
            from sklearn.base import clone
            model_fold = clone(model)
            if hasattr(model_fold, "random_state"):
                model_fold.random_state = RANDOM_STATE
        
        # Train model on scaled training fold
        model_fold.fit(X_train_fold_scaled, y_train_fold)
        
        # Evaluate on scaled test fold
        y_pred_fold = model_fold.predict(X_test_fold_scaled)
        mae_scores.append(mean_absolute_error(y_test_fold, y_pred_fold))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test_fold, y_pred_fold)))
    
    return {
        "MAE_mean": np.mean(mae_scores),
        "MAE_std": np.std(mae_scores),
        "RMSE_mean": np.mean(rmse_scores),
        "RMSE_std": np.std(rmse_scores),
    }


def cross_validate_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    include_xgboost: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Perform cross-validation for all models.

    Provides robust performance estimates by evaluating models across
    multiple time-series splits, helping identify overfitting and
    model stability.

    Args:
        X_train: Training features
        y_train: Training target
        n_splits: Number of cross-validation folds
        include_xgboost: Whether to include XGBoost in cross-validation
        verbose: Whether to print progress

    Returns:
        DataFrame with cross-validation results for all models
    """
    results = []

    # Linear Regression (using Ridge for numerical stability)
    if verbose:
        print("Cross-validating Linear Regression...")
    lr_model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    lr_cv = cross_validate_model(lr_model, X_train, y_train, n_splits)
    lr_cv["Model"] = "linear_regression"
    results.append(lr_cv)

    # Random Forest
    if verbose:
        print("Cross-validating Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_cv = cross_validate_model(rf_model, X_train, y_train, n_splits)
    rf_cv["Model"] = "random_forest"
    results.append(rf_cv)

    # Gradient Boosting
    if verbose:
        print("Cross-validating Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
    )
    gb_cv = cross_validate_model(gb_model, X_train, y_train, n_splits)
    gb_cv["Model"] = "gradient_boosting"
    results.append(gb_cv)

    # XGBoost (optional)
    if include_xgboost and XGBOOST_AVAILABLE:
        if verbose:
            print("Cross-validating XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        xgb_cv = cross_validate_model(xgb_model, X_train, y_train, n_splits)
        xgb_cv["Model"] = "xgboost"
        results.append(xgb_cv)

    cv_df = pd.DataFrame(results)
    cv_df = cv_df[
        [
            "Model",
            "MAE_mean",
            "MAE_std",
            "RMSE_mean",
            "RMSE_std",
        ]
    ]

    # Save cross-validation results
    save_path = Path("results/metrics/cross_validation_comparison.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(save_path, index=False)
    if verbose:
        print(f"Cross-validation results saved to {save_path}")

    return cv_df

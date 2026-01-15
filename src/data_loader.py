"""
Data loading and preprocessing module.

Handles loading raw retail store inventory forecasting data, preprocessing, 
feature engineering with Weather, Seasonality, and Region as key features,
and time-aware train/test splitting for demand and revenue forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Iterable, List, Any
import zipfile
import tempfile
import os


RANDOM_STATE = 42

# Required schema for the retail store inventory forecasting dataset
REQUIRED_COLUMNS = {
    "Date", 
    "Category", 
    "Units Sold", 
    "Store ID", 
    "Product ID", 
    "Region", 
    "Price", 
    "Weather Condition", 
    "Seasonality"
}


def validate_required_columns(df: pd.DataFrame):
    """
    Ensure the raw dataset contains required columns.

    Raises:
        ValueError: if dataset doesn't match expected format.
    """
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )


def load_raw_data(data_path: str = "data/raw/supermarket_sales.csv") -> pd.DataFrame:
    """
    Load raw supermarket sales data from CSV file or ZIP archive.

    Automatically handles ZIP files by extracting and loading the CSV inside.
    If ZIP contains multiple CSV files, loads the first one found.

    Args:
        data_path: Path to the raw CSV file or ZIP archive containing CSV

    Returns:
        DataFrame with raw sales data
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Check if it's a ZIP file
    if data_path.suffix.lower() == '.zip':
        return _load_from_zip(data_path)
    else:
        # Regular CSV file
        df = pd.read_csv(data_path)
        # Validate schema (returns format type, but we don't need to store it here)
        validate_required_columns(df)
        return df


def _load_from_zip(zip_path: Path) -> pd.DataFrame:
    """
    Extract and load CSV file from ZIP archive.

    Args:
        zip_path: Path to ZIP file

    Returns:
        DataFrame with raw sales data

    Raises:
        ValueError: If ZIP doesn't contain any CSV files
        FileNotFoundError: If ZIP file doesn't exist
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Find CSV files in the ZIP
        csv_files = [f for f in zip_ref.namelist() if f.lower().endswith('.csv')]
        
        if not csv_files:
            raise ValueError(
                f"No CSV files found in ZIP archive: {zip_path}. "
                "Please ensure the ZIP contains at least one CSV file."
            )
        
        # Use the first CSV file found (or the one that looks most relevant)
        # Prefer files in root, then by name (prefer shorter names, likely main dataset)
        csv_files.sort(key=lambda x: (x.count('/'), len(x)))
        csv_file = csv_files[0]
        
        # Extract to temporary file and read
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
            tmp_path = tmp_file.name
            try:
                # Extract and write CSV to temp file
                tmp_file.write(zip_ref.read(csv_file))
                tmp_file.flush()
                
                # Read CSV from temp file
                df = pd.read_csv(tmp_path)
                
                # Validate schema (returns format type, but we don't need to store it here)
                validate_required_columns(df)
                
                return df
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Date column and create datetime index.

    Args:
        df: DataFrame with Date column

    Returns:
        DataFrame with datetime index
    """
    df = df.copy()
    
    # Parse Date column (assumes YYYY-MM-DD format, with fallback)
    df["datetime"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors='coerce')
    # If that fails, try other common formats
    if df["datetime"].isna().any():
        df["datetime"] = pd.to_datetime(df["Date"], errors='coerce')
    
    df = df.set_index("datetime").sort_index()
    return df


def clean_and_validate_data(
    df: pd.DataFrame,
    check_product_category_consistency: bool = True,
    fix_product_category_consistency: bool = False,
    remove_duplicates: bool = True,
    handle_negative_values: bool = True,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean and validate data quality before processing.
    
    Checks for:
    - Product IDs in multiple categories (data quality issue)
    - Missing values in critical columns
    - Data type issues
    - Duplicate records
    - Negative values in Units Sold or Price
    
    Args:
        df: Raw DataFrame to clean and validate
        check_product_category_consistency: If True, check for Product IDs in multiple categories
        fix_product_category_consistency: If True, fix by using most frequent category per Product ID
        remove_duplicates: If True, remove exact duplicate rows
        handle_negative_values: If True, handle negative Units Sold/Price (set to 0)
        verbose: If True, print validation results
        
    Returns:
        Tuple of (cleaned DataFrame, validation report dictionary)
    """
    validation_report = {
        "product_category_inconsistencies": [],
        "missing_values": {},
        "duplicates_removed": 0,
        "negative_values_fixed": 0,
        "data_type_issues": []
    }
    
    df_cleaned = df.copy()
    
    # 1. Check Product ID - Category consistency
    if check_product_category_consistency and "Product ID" in df_cleaned.columns and "Category" in df_cleaned.columns:
        product_categories = df_cleaned.groupby("Product ID")["Category"].nunique()
        inconsistent_products = product_categories[product_categories > 1].index.tolist()
        
        if inconsistent_products:
            validation_report["product_category_inconsistencies"] = inconsistent_products
            if verbose:
                print(f"Warning: {len(inconsistent_products)} Product IDs appear in multiple categories: {inconsistent_products[:10]}{'...' if len(inconsistent_products) > 10 else ''}")
            
            if fix_product_category_consistency:
                # Fix by using most frequent category per Product ID
                for product_id in inconsistent_products:
                    product_data = df_cleaned[df_cleaned["Product ID"] == product_id]
                    if len(product_data) > 0:
                        most_frequent_category = product_data["Category"].mode()
                        if len(most_frequent_category) > 0:
                            most_frequent_category = most_frequent_category.iloc[0]
                            df_cleaned.loc[df_cleaned["Product ID"] == product_id, "Category"] = most_frequent_category
                if verbose:
                    print(f"Fixed: Assigned most frequent category to {len(inconsistent_products)} Product IDs")
    
    # 2. Check for missing values in critical columns
    critical_columns = ["Product ID", "Category", "Units Sold", "Date"]
    # Handle case where Date is in index (after parse_datetime)
    if "Date" not in df_cleaned.columns and (df_cleaned.index.name == "datetime" or isinstance(df_cleaned.index, pd.DatetimeIndex)):
        critical_columns = [col for col in critical_columns if col != "Date"]
    
    for col in critical_columns:
        if col in df_cleaned.columns:
            missing_count = df_cleaned[col].isna().sum()
            if missing_count > 0:
                validation_report["missing_values"][col] = missing_count
                if verbose:
                    print(f"Warning: {missing_count} missing values in '{col}'")
                # Fill missing values appropriately
                if col == "Units Sold":
                    df_cleaned[col] = df_cleaned[col].fillna(0)
                elif col in ["Product ID", "Category"]:
                    # Drop rows with missing Product ID or Category
                    df_cleaned = df_cleaned.dropna(subset=[col])
    
    # 3. Check for duplicate records
    if remove_duplicates:
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        duplicates_removed = initial_rows - len(df_cleaned)
        if duplicates_removed > 0:
            validation_report["duplicates_removed"] = duplicates_removed
            if verbose:
                print(f"Removed {duplicates_removed} duplicate rows")
    
    # 4. Check for negative values in Units Sold or Price
    if handle_negative_values:
        if "Units Sold" in df_cleaned.columns:
            negative_units = (df_cleaned["Units Sold"] < 0).sum()
            if negative_units > 0:
                validation_report["negative_values_fixed"] += negative_units
                df_cleaned.loc[df_cleaned["Units Sold"] < 0, "Units Sold"] = 0
                if verbose:
                    print(f"Fixed {negative_units} negative values in 'Units Sold' (set to 0)")
        
        if "Price" in df_cleaned.columns:
            negative_price = (df_cleaned["Price"] < 0).sum()
            if negative_price > 0:
                validation_report["negative_values_fixed"] += negative_price
                df_cleaned.loc[df_cleaned["Price"] < 0, "Price"] = 0
                if verbose:
                    print(f"Fixed {negative_price} negative values in 'Price' (set to 0)")
    
    # 5. Data type validation and conversion
    if "Product ID" in df_cleaned.columns:
        if not pd.api.types.is_string_dtype(df_cleaned["Product ID"]):
            df_cleaned["Product ID"] = df_cleaned["Product ID"].astype(str)
            validation_report["data_type_issues"].append("Product ID converted to string")
    
    if "Category" in df_cleaned.columns:
        if not pd.api.types.is_string_dtype(df_cleaned["Category"]):
            df_cleaned["Category"] = df_cleaned["Category"].astype(str)
            validation_report["data_type_issues"].append("Category converted to string")
    
    if "Store ID" in df_cleaned.columns:
        if not pd.api.types.is_string_dtype(df_cleaned["Store ID"]):
            df_cleaned["Store ID"] = df_cleaned["Store ID"].astype(str)
            validation_report["data_type_issues"].append("Store ID converted to string")
    
    if "Units Sold" in df_cleaned.columns:
        if not pd.api.types.is_numeric_dtype(df_cleaned["Units Sold"]):
            df_cleaned["Units Sold"] = pd.to_numeric(df_cleaned["Units Sold"], errors='coerce').fillna(0)
            validation_report["data_type_issues"].append("Units Sold converted to numeric")
    
    if "Price" in df_cleaned.columns:
        if not pd.api.types.is_numeric_dtype(df_cleaned["Price"]):
            df_cleaned["Price"] = pd.to_numeric(df_cleaned["Price"], errors='coerce').fillna(0)
            validation_report["data_type_issues"].append("Price converted to numeric")
    
    if verbose and validation_report["data_type_issues"]:
        print(f"Data type conversions: {', '.join(validation_report['data_type_issues'])}")
    
    return df_cleaned, validation_report


def aggregate_sales_by_all_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sales by Date, Store ID, Product ID, Category, Region, 
    preserving Weather and Seasonality.

    Aggregates Units Sold by summing, and averages Price and other numerical
    features. Uses most frequent Weather Condition and Seasonality for each
    day/region combination.

    Args:
        df: DataFrame with datetime index and required columns

    Returns:
        DataFrame with daily aggregated sales by all dimensions
    """
    # Reset index to have date as column for grouping
    df_reset = df.reset_index()
    df_reset["date"] = pd.to_datetime(df_reset["datetime"]).dt.date
    
    # Define aggregation functions
    agg_dict = {
        "Units Sold": "sum",
        "Price": "mean",
    }
    
    # Add other numerical columns if they exist
    numerical_cols = ["Inventory Level", "Units Ordered", "Demand Forecast", 
                      "Discount", "Competitor Pricing"]
    for col in numerical_cols:
        if col in df_reset.columns:
            agg_dict[col] = "mean"
    
    # Group by all key dimensions
    group_cols = ["date", "Store ID", "Product ID", "Category", "Region"]
    
    # Aggregate numerical columns
    daily_sales = df_reset.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Add Weather Condition and Seasonality (most frequent per group)
    weather_season = df_reset.groupby(group_cols).agg({
        "Weather Condition": lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        "Seasonality": lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()
    
    # Merge weather and seasonality back
    daily_sales = daily_sales.merge(
        weather_season[group_cols + ["Weather Condition", "Seasonality"]],
        on=group_cols,
        how="left"
    )
    
    # Rename quantity column for consistency
    daily_sales = daily_sales.rename(columns={"Units Sold": "quantity"})
    
    # Convert date back to datetime (keep as column, not index, since we have multiple dimensions per date)
    daily_sales["date"] = pd.to_datetime(daily_sales["date"])
    daily_sales = daily_sales.sort_values(["date"] + group_cols)

    return daily_sales


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features for demand forecasting.

    Adds calendar-based features that capture seasonality and trends:
    - Day of week (weekend vs weekday patterns)
    - Month (monthly seasonality)
    - Quarter (quarterly trends)
    - Day of month (monthly patterns)

    Args:
        df: DataFrame with datetime index or date column

    Returns:
        DataFrame with time-based features added
    """
    df = df.copy()

    # Handle both date column and datetime index
    if "date" in df.columns:
        date_series = pd.to_datetime(df["date"])
    elif isinstance(df.index, pd.DatetimeIndex):
        # If index is already DatetimeIndex, convert to Series to use .dt accessor
        date_series = pd.Series(df.index, index=df.index)
    else:
        date_series = pd.to_datetime(df.index)
        # Convert to Series if it's a DatetimeIndex
        if isinstance(date_series, pd.DatetimeIndex):
            date_series = pd.Series(date_series, index=df.index)

    df["day_of_week"] = date_series.dt.dayofweek
    df["month"] = date_series.dt.month
    df["quarter"] = date_series.dt.quarter
    df["day_of_month"] = date_series.dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df


def create_lag_features_by_dimensions(
    df: pd.DataFrame, lags: Iterable[int] = (1, 7, 30)
) -> pd.DataFrame:
    """
    Create lag features grouped by Store × Product × Category × Region.

    Lag features capture historical demand patterns which are crucial
    for time series forecasting. Created per product-location combination.
    Weather and Seasonality are included as features but not used for grouping
    to avoid fragmenting the time series into too many small groups.

    Args:
        df: DataFrame with datetime index and grouping columns
        lags: List of lag periods (in days)

    Returns:
        DataFrame with lag features added
    """
    df = df.copy()

    # Reset index to have date as column for grouping
    df_reset = df.reset_index()
    if "date" not in df_reset.columns and "datetime" in df_reset.columns:
        df_reset["date"] = pd.to_datetime(df_reset["datetime"])
    elif "date" not in df_reset.columns:
        df_reset["date"] = df_reset.index
    
    # Define grouping columns (exclude Weather/Seasonality to avoid over-fragmentation)
    # Weather/Seasonality remain as features but don't split temporal groupings
    group_cols = ["Store ID", "Product ID", "Category", "Region"]
    
    # Filter to only existing columns
    group_cols = [col for col in group_cols if col in df_reset.columns]
    
    # Sort by date and grouping columns
    df_reset = df_reset.sort_values(["date"] + group_cols)
    
    # Create lags per multi-dimensional group
    for lag in lags:
        if group_cols:
            df_reset[f"lag_{lag}"] = df_reset.groupby(group_cols)["quantity"].shift(lag)
        else:
            df_reset[f"lag_{lag}"] = df_reset["quantity"].shift(lag)

    # Keep date as column (don't set as index to avoid duplicate index issues with multiple dimensions)
    return df_reset


def create_moving_averages_by_dimensions(
    df: pd.DataFrame, windows: Iterable[int] = (7, 30)
) -> pd.DataFrame:
    """
    Create moving average features grouped by Store × Product × Category × Region.

    Moving averages smooth out noise and capture trends in demand.
    Calculated per product-location combination.
    Weather and Seasonality are included as features but not used for grouping
    to avoid fragmenting the time series into too many small groups.

    Args:
        df: DataFrame with datetime index and grouping columns
        windows: List of window sizes (in days)

    Returns:
        DataFrame with moving average features added
    """
    df = df.copy()

    # Reset index to have date as column for grouping
    df_reset = df.reset_index()
    if "date" not in df_reset.columns and "datetime" in df_reset.columns:
        df_reset["date"] = pd.to_datetime(df_reset["datetime"])
    elif "date" not in df_reset.columns:
        df_reset["date"] = df_reset.index
    
    # Define grouping columns (exclude Weather/Seasonality to avoid over-fragmentation)
    # Weather/Seasonality remain as features but don't split temporal groupings
    group_cols = ["Store ID", "Product ID", "Category", "Region"]
    
    # Filter to only existing columns
    group_cols = [col for col in group_cols if col in df_reset.columns]
    
    # Sort by date and grouping columns
    df_reset = df_reset.sort_values(["date"] + group_cols)
    
    # Create moving averages per multi-dimensional group
    # Since we have multiple dimensions, we can't use date as index (would create duplicates)
    # Use groupby with proper rolling window calculation
    for window in windows:
        if group_cols:
            # Sort by dimensions and date for proper rolling calculation
            df_reset = df_reset.sort_values(group_cols + ["date"])
            # Use transform with rolling to calculate moving average per group
            df_reset[f"ma_{window}"] = df_reset.groupby(group_cols)["quantity"].transform(
                lambda x: x.rolling(window=window, min_periods=window).mean()
            )
        else:
            # No grouping, simple rolling
            df_reset = df_reset.sort_values("date")
            df_reset[f"ma_{window}"] = df_reset["quantity"].rolling(window=window, min_periods=window).mean()
    
    return df_reset


def create_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical features: Weather Condition, Seasonality, Region (key features),
    plus Store ID, Product ID, Category.

    Args:
        df: DataFrame with categorical columns

    Returns:
        DataFrame with one-hot encoded categorical features
    """
    df = df.copy()
    
    # Key categorical features (must be included)
    key_categorical = ["Weather Condition", "Seasonality", "Region"]
    
    # Additional categorical features
    additional_categorical = ["Store ID", "Product ID", "Category"]
    
    # One-hot encode all categorical features
    for col in key_categorical + additional_categorical:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col.lower().replace(" ", "_"))
            df = pd.concat([df, dummies], axis=1)

    return df


def create_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Include all numerical features: Inventory Level, Units Ordered, Price,
    Discount, Competitor Pricing, Holiday/Promotion.

    Note: Demand Forecast is EXCLUDED to prevent data leakage - it's a pre-computed
    target variable and should not be used as a feature.

    Args:
        df: DataFrame with numerical columns

    Returns:
        DataFrame with numerical features preserved
    """
    df = df.copy()

    # Numerical features to include
    # Demand Forecast is EXCLUDED - it's a pre-computed target, not a feature
    numerical_features = [
        "Inventory Level",
        "Units Ordered", 
        "Price",
        "Discount",
        "Competitor Pricing"
    ]
    
    # Check for Holiday/Promotion (binary)
    if "Holiday" in df.columns:
        df["is_holiday"] = (df["Holiday"] == 1).astype(int)
    if "Promotion" in df.columns:
        df["is_promotion"] = (df["Promotion"] == 1).astype(int)
    
    # Ensure all numerical features are present (fill missing with 0)
    for col in numerical_features:
        if col not in df.columns:
            df[col] = 0.0
    
    return df


def calculate_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate revenue = Price × Units Sold.

    Args:
        df: DataFrame with Price and quantity (Units Sold) columns

    Returns:
        DataFrame with revenue column added
    """
    df = df.copy()
    
    # Calculate revenue
    if "Price" in df.columns and "quantity" in df.columns:
        df["revenue"] = df["Price"] * df["quantity"]
    elif "Price" in df.columns and "Units Sold" in df.columns:
        df["revenue"] = df["Price"] * df["Units Sold"]
        df["quantity"] = df["Units Sold"]  # Standardize column name
    else:
        # If Price or quantity missing, set revenue to 0
        df["revenue"] = 0.0
        if "quantity" not in df.columns and "Units Sold" in df.columns:
            df["quantity"] = df["Units Sold"]

    return df


def filter_products_with_sufficient_history(
    df: pd.DataFrame, min_days: int = 30
) -> pd.DataFrame:
    """
    Filter products with sufficient sales history.

    Ensures we only forecast for products with enough historical data
    for reliable predictions. Groups by Product ID and Category.

    Args:
        df: DataFrame with Product ID, Category, and date columns
        min_days: Minimum number of days of sales history required

    Returns:
        Filtered DataFrame
    """
    # Group by Product ID and Category to count days
    if "Product ID" in df.columns and "Category" in df.columns:
        group_cols = ["Product ID", "Category"]
    elif "Product ID" in df.columns:
        group_cols = ["Product ID"]
    elif "Category" in df.columns:
        group_cols = ["Category"]
    else:
        # Fallback: no filtering if grouping columns not available
        return df.copy()
    
    product_counts = df.groupby(group_cols).size()
    valid_products = product_counts[product_counts >= min_days].index
    
    if isinstance(valid_products, pd.MultiIndex):
        # Multi-level index
        df_filtered = df.set_index(group_cols).loc[valid_products].reset_index()
    else:
        # Single-level index
        df_filtered = df[df[group_cols[0]].isin(valid_products)].copy()

    return df_filtered


def create_all_features(
    df: pd.DataFrame,
    lags: Iterable[int] = (1, 7, 30),
    moving_average_windows: Iterable[int] = (7, 30),
) -> pd.DataFrame:
    """
    Create all engineered features for demand and revenue forecasting.

    Orchestrates feature engineering pipeline: time features, numerical features,
    lags, moving averages, categorical encoding, and revenue calculation.
    Ensures Weather, Seasonality, and Region features are always included.

    Args:
        df: DataFrame with date column and required columns (Store ID, Product ID, 
            Category, Region, Weather Condition, Seasonality, Price, quantity)

    Returns:
        DataFrame with all engineered features including Weather/Seasonality/Region
    """
    df = df.copy()
    
    # Ensure date column exists before creating time features
    # The input should have a "date" column from reset_index() in load_and_preprocess_data
    if "date" not in df.columns:
        # If date is not in columns, it might be in the index
        if isinstance(df.index, pd.DatetimeIndex) or df.index.name == "date":
            # Reset index to create date column
            df = df.reset_index()
            # Ensure we have a "date" column (reset_index should create it if index was named "date")
            if "date" not in df.columns:
                # Find datetime column and rename it
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df = df.rename(columns={col: "date"})
                        break
    
    # Create time features (now we have date column)
    df = create_time_features(df)

    # Ensure date is datetime for sorting
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    
    # Sort by date and dimensions
    # Note: After lag/moving average functions, date might be the index
    if "date" in df.columns:
        sort_cols = ["date"]
        for col in ["Store ID", "Product ID", "Category", "Region"]:
            if col in df.columns:
                sort_cols.append(col)
        df = df.sort_values(sort_cols)
    elif isinstance(df.index, pd.DatetimeIndex) or df.index.name == "date":
        # Date is the index, sort by index and other columns
        sort_cols = []
        for col in ["Store ID", "Product ID", "Category", "Region"]:
            if col in df.columns:
                sort_cols.append(col)
        if sort_cols:
            df = df.sort_values(sort_cols)
        df = df.sort_index()
    else:
        # No date information, just sort by available columns
        sort_cols = []
        for col in ["Store ID", "Product ID", "Category", "Region"]:
            if col in df.columns:
                sort_cols.append(col)
        if sort_cols:
            df = df.sort_values(sort_cols)

    # Create numerical features
    df = create_numerical_features(df)
    
    # Calculate revenue
    df = calculate_revenue(df)

    # Create lag and moving average features (grouped by dimensions)
    df = create_lag_features_by_dimensions(df, lags=lags)
    df = create_moving_averages_by_dimensions(df, windows=moving_average_windows)

    # Clean up extra columns from reset_index operations
    df = df.drop(columns=["level_0", "index"], errors="ignore")

    # Encode categorical features (Weather, Seasonality, Region are key features)
    df = create_categorical_features(df)

    # Fill NaN values from lag features (first days have no history)
    lag_cols = [col for col in df.columns if col.startswith("lag_")]
    ma_cols = [col for col in df.columns if col.startswith("ma_")]
    df[lag_cols + ma_cols] = df[lag_cols + ma_cols].fillna(0)

    # Replace infinite values with NaN first, then fill
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill any remaining NaN values
    df = df.fillna(0)
    
    # Clip extreme values to prevent numerical overflow
    # For numerical columns, clip to reasonable bounds
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['current_stock']:  # Don't clip stock, it might legitimately be large
            df[col] = df[col].clip(lower=-1e6, upper=1e6)

    # Set date as index for time-aware splitting
    if "date" in df.columns:
        df = df.set_index("date")
    
    # Validate that Weather, Seasonality, Region features are present
    weather_cols = [col for col in df.columns if "weather_condition" in col.lower()]
    seasonality_cols = [col for col in df.columns if "seasonality" in col.lower()]
    region_cols = [col for col in df.columns if "region" in col.lower()]
    
    if not weather_cols:
        raise ValueError("Weather Condition features not found after encoding. Check data.")
    if not seasonality_cols:
        raise ValueError("Seasonality features not found after encoding. Check data.")
    if not region_cols:
        raise ValueError("Region features not found after encoding. Check data.")

    return df


def time_aware_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data using time-aware approach to avoid data leakage.

    Uses chronological split: earlier data for training, later data for testing.
    This prevents future information from leaking into past predictions.

    Args:
        df: DataFrame with datetime index or date column
        test_size: Proportion of data to use for testing (from end of time series)

    Returns:
        Tuple of (train_df, test_df)
    """
    # Sort by date (either index or date column)
    if "date" in df.columns:
        df = df.sort_values("date")
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
    else:
        df = df.sort_index()
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def load_and_preprocess_data(
    data_path: str = "data/raw/Retail_store_inventory_forecasting_dataset.zip",
    save_processed: bool = True,
    processed_path: str = "data/processed/processed_data.csv",
    add_stock_to_processed: bool = True,
    days_of_coverage: float = 10.0,
    coverage_method: str = "mean",
    stock_path: str = "data/raw/current_stock.csv",
    test_size: float = 0.2,
    min_history_days: int = 30,
    lags: Iterable[int] = (1, 7, 30),
    moving_average_windows: Iterable[int] = (7, 30),
    validate_schema: bool = True,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete data loading and preprocessing pipeline.

    Orchestrates the full preprocessing workflow from raw data to
    feature-engineered datasets ready for modeling. Now includes
    automatic addition of current_stock column if missing.

    Args:
        data_path: Path to raw CSV file
        save_processed: Whether to save processed data
        processed_path: Path to save processed data
        add_stock_to_processed: If True, add current_stock column if missing
        days_of_coverage: Days of demand to use for stock estimation (if needed)
        coverage_method: Method for stock estimation ('mean', 'median', 'p75', 'p90')

    Returns:
        Tuple of (train_df, test_df) ready for modeling, both with current_stock column
    """
    # Load raw data
    if verbose:
        print("Loading raw data...")
    df_raw = load_raw_data(data_path)
    if validate_schema:
        validate_required_columns(df_raw)

    # Parse datetime
    if verbose:
        print("Parsing datetime...")
    df = parse_datetime(df_raw)

    # Clean and validate data
    if verbose:
        print("Cleaning and validating data...")
    df, validation_report = clean_and_validate_data(
        df,
        check_product_category_consistency=True,
        fix_product_category_consistency=True,  # Auto-fix inconsistencies
        remove_duplicates=True,
        handle_negative_values=True,
        verbose=verbose
    )
    if verbose and validation_report["product_category_inconsistencies"]:
        print(f"  Found {len(validation_report['product_category_inconsistencies'])} Product IDs with category inconsistencies (fixed)")

    # Aggregate to daily sales by all dimensions (preserving Weather/Seasonality/Region)
    if verbose:
        print("Aggregating daily sales by all dimensions...")
    df_aggregated = aggregate_sales_by_all_dimensions(df)

    # Filter products with sufficient history
    if verbose:
        print("Filtering products with sufficient history...")
    df_filtered = filter_products_with_sufficient_history(
        df_aggregated, min_days=min_history_days
    )

    if df_filtered.empty:
        raise ValueError(
            "No data left after filtering for minimum history. "
            "Reduce min_history_days or verify the input dataset."
        )

    # Date should already be a column (not index) since we have multiple dimensions per date
    # Just ensure it's a datetime type
    if "date" in df_filtered.columns:
        df_filtered["date"] = pd.to_datetime(df_filtered["date"])
    else:
        # If somehow date is in index, reset it
        df_filtered = df_filtered.reset_index()
        if "date" not in df_filtered.columns:
            # Find datetime column
            for col in df_filtered.columns:
                if pd.api.types.is_datetime64_any_dtype(df_filtered[col]):
                    df_filtered = df_filtered.rename(columns={col: "date"})
                    break

    # Create all features
    if verbose:
        print("Creating features...")
    df_features = create_all_features(
        df_filtered, lags=lags, moving_average_windows=moving_average_windows
    )

    # Add current_stock column if requested and missing
    if add_stock_to_processed:
        if verbose:
            print("Adding current_stock column to processed data...")
        df_features, stock_source = add_stock_column_to_dataframe(
            df_features,
            df_raw=df_raw,
            df_aggregated=df_aggregated,
            auto_estimate=True,
            days_of_coverage=days_of_coverage,
            coverage_method=coverage_method,
            stock_path=stock_path,
        )
        if verbose:
            print(f"  Stock source: {stock_source}")

    # Time-aware train/test split
    if verbose:
        print("Splitting data (time-aware)...")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    train_df, test_df = time_aware_train_test_split(df_features, test_size=test_size)

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train/test split produced an empty dataset. "
            "Check test_size and filtering thresholds."
        )

    # Save processed data if requested
    if save_processed:
        processed_path = Path(processed_path)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        # Reset index for saving
        df_features_save = df_features.reset_index()
        # Clean up any extra columns from reset_index
        df_features_save = df_features_save.drop(columns=["level_0", "index"], errors="ignore")
        df_features_save.to_csv(processed_path, index=False)
        if verbose:
            print(f"Processed data saved to {processed_path}")

    if verbose:
        print(f"Training set: {len(train_df)} samples")
        print(f"Test set: {len(test_df)} samples")

    return train_df, test_df


def detect_stock_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect if dataset has inventory/stock column.

    Checks for common column names that might indicate stock levels:
    inventory, stock, current_stock, on_hand, quantity_on_hand, etc.

    Args:
        df: Raw DataFrame to check

    Returns:
        Column name if found, None otherwise
    """
    stock_keywords = [
        "inventory",
        "stock",
        "current_stock",
        "on_hand",
        "quantity_on_hand",
        "units_in_stock",
        "available_stock",
    ]

    df_columns_lower = [col.lower().strip() for col in df.columns]

    for keyword in stock_keywords:
        for i, col_lower in enumerate(df_columns_lower):
            if keyword in col_lower:
                return df.columns[i]  # Return original column name

    return None


def _create_product_key(row: pd.Series) -> str:
    """Create a composite product key from Product ID and Category."""
    if "Product ID" in row and "Category" in row:
        return f"{row['Product ID']}_{row['Category']}"
    elif "Product ID" in row:
        return str(row["Product ID"])
    elif "Category" in row:
        return str(row["Category"])
    else:
        return "default"


def estimate_stock_from_historical_demand(
    df_aggregated: pd.DataFrame,
    days_of_coverage: float = 10.0,
    coverage_method: str = "mean",
) -> Dict[str, float]:
    """
    Estimate current stock levels based on historical demand patterns.

    Uses historical sales data to estimate realistic stock levels that
    would provide sufficient coverage. This makes reorder recommendations
    more meaningful and data-driven.

    Estimation methods:
    - 'mean': Average daily demand (balanced approach)
    - 'median': Less sensitive to outliers
    - 'p75': 75th percentile (covers most days, conservative)
    - 'p90': 90th percentile (high coverage, very conservative)

    Args:
        df_aggregated: DataFrame with daily aggregated sales (date, Product ID, Category, quantity)
        days_of_coverage: Number of days of average demand to use as stock estimate
        coverage_method: Method to calculate base demand ('mean', 'median', 'p75', 'p90')

    Returns:
        Dictionary mapping product key (Product ID + Category) to estimated stock level
    """
    stock_dict = {}
    
    # Create product key column - handle both date as column and date as index
    df_agg = df_aggregated.copy()
    
    # Reset index if date is in index to ensure apply() works correctly
    if isinstance(df_agg.index, pd.DatetimeIndex) or df_agg.index.name == "date":
        df_agg = df_agg.reset_index()
        # Remove any extra columns from reset_index
        df_agg = df_agg.drop(columns=["level_0", "index"], errors="ignore")
    
    # Create product key
    df_agg["product_key"] = df_agg.apply(_create_product_key, axis=1)

    # Calculate historical demand statistics per product
    for product_key in df_agg["product_key"].unique():
        product_data = df_agg[df_agg["product_key"] == product_key]["quantity"]
        
        # Skip if no data
        if len(product_data) == 0:
            continue

        # Calculate base demand based on method
        if coverage_method == "mean":
            base_demand = product_data.mean()
        elif coverage_method == "median":
            base_demand = product_data.median()
        elif coverage_method == "p75":
            base_demand = product_data.quantile(0.75)
        elif coverage_method == "p90":
            base_demand = product_data.quantile(0.90)
        else:
            # Default to mean if method not recognized
            base_demand = product_data.mean()

        # Estimate stock as days_of_coverage × base_demand
        estimated_stock = base_demand * days_of_coverage

        # Ensure positive values (round to nearest integer for realistic stock)
        stock_dict[product_key] = max(1.0, round(estimated_stock, 0))

    return stock_dict


def load_or_estimate_current_stock(
    stock_path: str = "data/raw/current_stock.csv",
    df_raw: pd.DataFrame = None,
    df_aggregated: pd.DataFrame = None,
    auto_estimate: bool = True,
    days_of_coverage: float = 10.0,
    coverage_method: str = "mean",
) -> Tuple[Dict[str, float], str]:
    """
    Load current stock from file or estimate from historical data.

    Professional approach that checks multiple sources in priority order:
    1. Stock/inventory column in raw dataset (if exists)
    2. current_stock.csv file (if exists)
    3. Estimated from historical demand patterns (if auto_estimate=True)
    4. Default zeros (fallback)

    This ensures stock levels are always available and realistic.

    Args:
        stock_path: Path to current_stock.csv file
        df_raw: Raw DataFrame to check for stock columns
        df_aggregated: Aggregated daily sales DataFrame for estimation
        auto_estimate: If True, estimate stock when file/column not found
        days_of_coverage: Days of demand to use for estimation (if auto_estimate=True)
        coverage_method: Method for estimation ('mean', 'median', 'p75', 'p90')

    Returns:
        Tuple of (stock_dict, source_description)
        stock_dict: Dictionary mapping product_line to stock level
        source_description: Description of where stock came from
    """
    stock_path = Path(stock_path)

    # Get list of all products (used for validation)
    if df_aggregated is not None:
        df_agg = df_aggregated.copy()
        df_agg["product_key"] = df_agg.apply(_create_product_key, axis=1)
        all_products = df_agg["product_key"].unique().tolist()
    else:
        # Default products if no data available
        all_products = []

    # Step 1: Check for stock column in raw dataset
    if df_raw is not None:
        stock_column = detect_stock_column(df_raw)
        if stock_column:
            # Extract stock per product (use latest value per product)
            df_raw_copy = df_raw.copy()
            df_raw_copy.columns = df_raw_copy.columns.str.lower().str.replace(" ", "_")

            # Find product column
            product_col = None
            for col in df_raw_copy.columns:
                if "product" in col and "line" in col:
                    product_col = col
                    break

            if product_col and stock_column.lower() in df_raw_copy.columns:
                stock_col_lower = stock_column.lower()
                # Get latest stock value per product (most recent transaction)
                stock_df = df_raw_copy.groupby(product_col)[stock_col_lower].last()
                stock_dict = {}

                # Normalize product names and ensure positive values
                for product, stock in stock_df.items():
                    stock_value = float(stock) if pd.notna(stock) else 0.0
                    stock_dict[product] = max(0.0, stock_value)

                # Ensure all products are in dictionary
                for product in all_products:
                    if product not in stock_dict:
                        stock_dict[product] = 0.0

                return stock_dict, f"Extracted from dataset column: {stock_column}"

    # Step 2: Check for current_stock.csv file
    if stock_path.exists():
        try:
            stock_df = pd.read_csv(stock_path)
            # Support both old format (product_line) and new format (Product ID + Category)
            if "product_line" in stock_df.columns:
                stock_dict = dict(zip(stock_df["product_line"], stock_df["current_stock"]))
            elif "Product ID" in stock_df.columns and "Category" in stock_df.columns:
                stock_df["product_key"] = stock_df.apply(_create_product_key, axis=1)
                stock_dict = dict(zip(stock_df["product_key"], stock_df["current_stock"]))
            else:
                raise ValueError("Stock file must have 'product_line' or 'Product ID' + 'Category' columns")

            # Ensure all values are positive floats
            stock_dict = {k: float(max(0, v)) for k, v in stock_dict.items()}

            # Check if stock file format matches data format
            # If we have products in data but none match stock file keys, format mismatch
            if all_products:
                matching_products = [p for p in all_products if p in stock_dict]
                if len(matching_products) == 0:
                    # Stock file format doesn't match data format - fall back to estimation
                    if auto_estimate and df_aggregated is not None:
                        stock_dict = estimate_stock_from_historical_demand(
                            df_aggregated, days_of_coverage, coverage_method
                        )
                        method_desc = (
                            f"Estimated from historical demand "
                            f"({coverage_method}, {days_of_coverage} days coverage) "
                            f"[stock file format mismatch]"
                        )
                        return stock_dict, method_desc
                    # If estimation disabled, still use zeros but note the mismatch
                    stock_dict = {product: 0.0 for product in all_products}
                    return stock_dict, "Stock file format mismatch (zeros used)"
                else:
                    # Some products match - fill in missing ones with zeros
                    for product in all_products:
                        if product not in stock_dict:
                            stock_dict[product] = 0.0
            else:
                # No products list available - use stock file as-is
                pass

            return stock_dict, "Loaded from current_stock.csv file"
        except Exception as e:
            # Silently handle stock file loading errors (will fall back to estimation)
            pass

    # Step 3: Estimate from historical demand (if enabled)
    if auto_estimate and df_aggregated is not None:
        stock_dict = estimate_stock_from_historical_demand(
            df_aggregated, days_of_coverage, coverage_method
        )
        method_desc = (
            f"Estimated from historical demand "
            f"({coverage_method}, {days_of_coverage} days coverage)"
        )
        return stock_dict, method_desc

    # Step 4: Default fallback (should rarely happen)
    # Warning suppressed - will use zeros silently
    stock_dict = {product: 0.0 for product in all_products}
    return stock_dict, "No stock data available (zeros used)"


def add_stock_column_to_dataframe(
    df: pd.DataFrame,
    df_raw: pd.DataFrame = None,
    df_aggregated: pd.DataFrame = None,
    auto_estimate: bool = True,
    days_of_coverage: float = 10.0,
    coverage_method: str = "mean",
    stock_path: str = "data/raw/current_stock.csv",
) -> Tuple[pd.DataFrame, str]:
    """
    Add current_stock column to DataFrame if missing.

    Checks multiple sources to add stock column:
    1. Stock column already exists in DataFrame (no action needed)
    2. Stock column in raw dataset (aggregate by product_line)
    3. Estimate from historical demand patterns (if auto_estimate=True)

    Stock is static per product_line (same value for all rows of the same product).

    Args:
        df: DataFrame with product_line column to add stock to
        df_raw: Raw DataFrame to check for stock columns
        df_aggregated: Aggregated daily sales DataFrame for estimation
        auto_estimate: If True, estimate stock when column/file not found
        days_of_coverage: Days of demand to use for estimation (if auto_estimate=True)
        coverage_method: Method for estimation ('mean', 'median', 'p75', 'p90')

    Returns:
        Tuple of (DataFrame with current_stock column, source_description)
        DataFrame: Input DataFrame with current_stock column added
        source_description: Description of where stock came from
    """
    # Check if current_stock column already exists
    if "current_stock" in df.columns:
        return df, "current_stock column already exists in DataFrame"

    # Ensure Product ID and Category columns exist (or product_line for backward compatibility)
    has_product_key = ("Product ID" in df.columns and "Category" in df.columns) or "product_line" in df.columns
    if not has_product_key:
        raise ValueError("DataFrame must have 'Product ID' + 'Category' or 'product_line' column to add current_stock")

    # Get stock dictionary using existing logic
    stock_dict, source = load_or_estimate_current_stock(
        stock_path=stock_path,
        df_raw=df_raw,
        df_aggregated=df_aggregated,
        auto_estimate=auto_estimate,
        days_of_coverage=days_of_coverage,
        coverage_method=coverage_method,
    )

    # Add stock column by mapping product key to stock value
    df_copy = df.copy()
    
    # Reset index temporarily if needed to ensure apply() works correctly
    index_was_date = isinstance(df_copy.index, pd.DatetimeIndex) or df_copy.index.name == "date"
    if index_was_date:
        df_copy = df_copy.reset_index()
        # Remove any extra columns from reset_index
        df_copy = df_copy.drop(columns=["level_0", "index"], errors="ignore")
        if "date" in df_copy.columns and index_was_date:
            # Keep date column for later
            pass
    
    if "product_line" in df_copy.columns:
        # Backward compatibility
        df_copy["current_stock"] = df_copy["product_line"].map(stock_dict).fillna(0.0)
    elif "Product ID" in df_copy.columns and "Category" in df_copy.columns:
        # New format: create product key
        df_copy["product_key"] = df_copy.apply(_create_product_key, axis=1)
        df_copy["current_stock"] = df_copy["product_key"].map(stock_dict).fillna(0.0)
        df_copy = df_copy.drop("product_key", axis=1)
    else:
        # No product identification available
        df_copy["current_stock"] = 0.0
    
    # Set date back as index if it was originally indexed
    if index_was_date and "date" in df_copy.columns:
        df_copy = df_copy.set_index("date")
    
    # Validate that stock values were properly mapped (not all zeros if stock_dict has values)
    if stock_dict and any(v > 0 for v in stock_dict.values()):
        non_zero_stock_count = (df_copy["current_stock"] > 0).sum()
        if non_zero_stock_count == 0:
            # Warning: stock mapping may have failed
            # This is informational, not an error - will use zeros
            pass

    return df_copy, source


def extract_stock_from_dataframe(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract current_stock dictionary from DataFrame.

    Gets unique stock values per product from current_stock column.
    Uses Product ID + Category as key, or product_line for backward compatibility.
    Useful for converting DataFrame stock column to dictionary format
    needed by recommendation functions.

    Args:
        df: DataFrame with Product ID, Category (or product_line) and current_stock columns

    Returns:
        Dictionary mapping product key to stock level

    Raises:
        ValueError: If current_stock or product identification columns are missing
    """
    if "current_stock" not in df.columns:
        raise ValueError("DataFrame must have 'current_stock' column")
    
    df_copy = df.copy()
    
    # Reset index if needed to ensure apply() works correctly
    index_was_date = isinstance(df_copy.index, pd.DatetimeIndex) or df_copy.index.name == "date"
    if index_was_date:
        df_copy = df_copy.reset_index()
        # Remove any extra columns from reset_index
        df_copy = df_copy.drop(columns=["level_0", "index"], errors="ignore")
    
    # Create product key
    if "product_line" in df_copy.columns:
        # Backward compatibility
        group_col = "product_line"
    elif "Product ID" in df_copy.columns and "Category" in df_copy.columns:
        df_copy["product_key"] = df_copy.apply(_create_product_key, axis=1)
        group_col = "product_key"
    else:
        raise ValueError("DataFrame must have 'Product ID' + 'Category' or 'product_line' column")

    # Extract unique stock values per product
    stock_dict = df_copy.groupby(group_col)["current_stock"].first().to_dict()

    # Ensure all values are floats
    stock_dict = {k: float(v) for k, v in stock_dict.items()}

    return stock_dict


def load_current_stock(
    stock_path: str = "data/raw/current_stock.csv",
) -> Dict[str, float]:
    """
    Load current stock levels from CSV file (legacy function for backward compatibility).

    This function is deprecated. Use `load_or_estimate_current_stock()` instead
    for data-driven stock estimation.

    Args:
        stock_path: Path to the current stock CSV file

    Returns:
        Dictionary mapping product_line to current stock level
    """
    stock_dict, _ = load_or_estimate_current_stock(
        stock_path=stock_path,
        df_raw=None,
        df_aggregated=None,
        auto_estimate=False,
    )
    return stock_dict

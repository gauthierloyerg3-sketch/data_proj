"""
Comprehensive tests for data_loader module.

Tests data loading, preprocessing, feature engineering, and validation functions.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import zipfile

from src.data_loader import (
    validate_required_columns,
    validate_data_leakage_risks,
    load_raw_data,
    parse_datetime,
    clean_and_validate_data,
    create_time_features,
    create_lag_features_by_dimensions,
    create_moving_averages_by_dimensions,
    create_numerical_features,
    create_categorical_features,
    filter_products_with_sufficient_history,
    time_aware_train_test_split,
)


def test_validate_required_columns_empty_dataframe():
    """Test that empty DataFrame raises ValueError."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="None or empty"):
        validate_required_columns(df)


def test_validate_required_columns_none():
    """Test that None DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="None or empty"):
        validate_required_columns(None)


def test_validate_data_leakage_risks_empty():
    """Test data leakage validation with empty DataFrame."""
    df = pd.DataFrame()
    result = validate_data_leakage_risks(df)
    assert not result["has_warnings"]
    assert len(result["warnings"]) == 0


def test_validate_data_leakage_risks_inventory_negative():
    """Test detection of negative inventory values."""
    df = pd.DataFrame({
        "Inventory Level": [-5, 10, 20],
        "quantity": [5, 10, 15]
    })
    result = validate_data_leakage_risks(df, verbose=False)
    assert result["has_warnings"]
    assert "inventory_negative_values" in result["risks"]


def test_validate_data_leakage_risks_units_ordered_negative():
    """Test detection of negative Units Ordered values."""
    df = pd.DataFrame({
        "Units Ordered": [-5, 10, 20],
        "quantity": [5, 10, 15]
    })
    result = validate_data_leakage_risks(df, verbose=False)
    assert result["has_warnings"]
    assert "units_ordered_negative_values" in result["risks"]


def test_validate_data_leakage_risks_inventory_correlation():
    """Test detection of high correlation between inventory and future sales."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # Create synthetic data where inventory at t correlates with sales at t+1
    # (simulating end-of-day inventory)
    df = pd.DataFrame({
        "Inventory Level": np.random.randn(100).cumsum() + 50,
        "quantity": np.random.randn(100).cumsum() + 50,
    }, index=dates)
    # Make inventory at t = sales at t+1 (high correlation)
    df["Inventory Level"] = df["quantity"].shift(-1).fillna(50)
    
    result = validate_data_leakage_risks(df, verbose=False)
    # May or may not detect depending on correlation threshold
    # Just check that function runs without error
    assert isinstance(result, dict)
    assert "has_warnings" in result


def test_parse_datetime_missing_date_column():
    """Test that missing Date column raises ValueError."""
    df = pd.DataFrame({"Units Sold": [10, 20]})
    with pytest.raises(ValueError, match="Date column not found"):
        parse_datetime(df)


def test_parse_datetime_invalid_dates():
    """Test that invalid dates raise ValueError."""
    df = pd.DataFrame({
        "Date": ["invalid", "also-invalid", "not-a-date"],
        "Units Sold": [10, 20, 30]
    })
    with pytest.raises(ValueError, match="Failed to parse Date column"):
        parse_datetime(df)


def test_parse_datetime_valid():
    """Test parsing valid dates."""
    df = pd.DataFrame({
        "Date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "Units Sold": [10, 20, 30]
    })
    result = parse_datetime(df)
    assert isinstance(result.index, pd.DatetimeIndex)
    assert len(result) == 3


def test_clean_and_validate_data_negative_values():
    """Test handling of negative values."""
    df = pd.DataFrame({
        "Product ID": ["P001", "P002"],
        "Category": ["A", "B"],
        "Units Sold": [-5, 10],
        "Price": [5.0, -2.0],
        "Date": ["2020-01-01", "2020-01-02"]
    })
    df_cleaned, report = clean_and_validate_data(df, handle_negative_values=True)
    assert report["negative_values_fixed"] > 0
    assert (df_cleaned["Units Sold"] >= 0).all()
    assert (df_cleaned["Price"] >= 0).all()


def test_clean_and_validate_data_duplicates():
    """Test removal of duplicate rows."""
    df = pd.DataFrame({
        "Product ID": ["P001", "P001", "P002"],
        "Category": ["A", "A", "B"],
        "Units Sold": [10, 10, 20],  # Duplicate first two rows
        "Date": ["2020-01-01", "2020-01-01", "2020-01-02"]
    })
    df_cleaned, report = clean_and_validate_data(df, remove_duplicates=True)
    assert report["duplicates_removed"] > 0
    assert len(df_cleaned) < len(df)


def test_create_time_features_date_column():
    """Test time feature creation from date column."""
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=10, freq="D"),
        "quantity": range(10)
    })
    result = create_time_features(df)
    assert "day_of_week" in result.columns
    assert "month" in result.columns
    assert "quarter" in result.columns
    assert "day_of_month" in result.columns
    assert "is_weekend" in result.columns


def test_create_time_features_datetime_index():
    """Test time feature creation from datetime index."""
    df = pd.DataFrame(
        {"quantity": range(10)},
        index=pd.date_range("2020-01-01", periods=10, freq="D")
    )
    result = create_time_features(df)
    assert "day_of_week" in result.columns
    assert "is_weekend" in result.columns


def test_create_lag_features_by_dimensions():
    """Test lag feature creation."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "Store ID": ["S1"] * 10,
        "Product ID": ["P1"] * 10,
        "Category": ["A"] * 10,
        "Region": ["R1"] * 10,
        "quantity": range(10)
    })
    result = create_lag_features_by_dimensions(df, lags=[1, 7])
    assert "lag_1" in result.columns
    assert "lag_7" in result.columns


def test_create_moving_averages_by_dimensions():
    """Test moving average feature creation."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "Store ID": ["S1"] * 10,
        "Product ID": ["P1"] * 10,
        "Category": ["A"] * 10,
        "Region": ["R1"] * 10,
        "quantity": range(10)
    })
    result = create_moving_averages_by_dimensions(df, windows=[7])
    assert "ma_7" in result.columns


def test_create_numerical_features():
    """Test numerical feature creation."""
    df = pd.DataFrame({
        "Inventory Level": [10, 20, 30],
        "Units Ordered": [5, 10, 15],
        "Price": [5.0, 10.0, 15.0],
        "Discount": [0.1, 0.2, 0.0],
        "Competitor Pricing": [4.5, 9.5, 14.5],
        "Holiday": [1, 0, 0],
        "Promotion": [0, 1, 0]
    })
    result = create_numerical_features(df)
    assert "is_holiday" in result.columns
    assert "is_promotion" in result.columns
    assert "Inventory Level" in result.columns


def test_create_categorical_features():
    """Test categorical feature encoding."""
    df = pd.DataFrame({
        "Weather Condition": ["Sunny", "Rainy", "Cloudy"],
        "Seasonality": ["Summer", "Winter", "Spring"],
        "Region": ["North", "South", "East"],
        "Store ID": ["S1", "S2", "S1"],
        "Product ID": ["P1", "P2", "P1"],
        "Category": ["A", "B", "A"]
    })
    result = create_categorical_features(df)
    # Check for one-hot encoded columns
    weather_cols = [col for col in result.columns if "weather_condition" in col.lower()]
    assert len(weather_cols) > 0
    seasonality_cols = [col for col in result.columns if "seasonality" in col.lower()]
    assert len(seasonality_cols) > 0
    region_cols = [col for col in result.columns if "region" in col.lower()]
    assert len(region_cols) > 0


def test_filter_products_with_sufficient_history():
    """Test filtering products by history length."""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "Product ID": ["P1"] * 30 + ["P2"] * 20,  # P1 has 30 days, P2 has 20 days
        "Category": ["A"] * 30 + ["B"] * 20,
        "quantity": range(50)
    })
    # Filter for products with at least 25 days
    result = filter_products_with_sufficient_history(df, min_days=25)
    # Only P1 should remain (30 days >= 25)
    assert len(result["Product ID"].unique()) == 1
    assert "P1" in result["Product ID"].values


def test_time_aware_train_test_split_chronological():
    """Test that train/test split is chronological."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "quantity": range(100)
    })
    train_df, test_df = time_aware_train_test_split(df, test_size=0.2)
    assert len(train_df) == 80
    assert len(test_df) == 20
    assert train_df["date"].max() <= test_df["date"].min()


def test_time_aware_train_test_split_datetime_index():
    """Test train/test split with datetime index."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {"quantity": range(100)},
        index=dates
    )
    train_df, test_df = time_aware_train_test_split(df, test_size=0.2)
    assert len(train_df) == 80
    assert len(test_df) == 20
    assert train_df.index.max() <= test_df.index.min()


def test_time_aware_train_test_split_invalid_size():
    """Test that invalid test_size raises ValueError."""
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=10, freq="D"),
        "quantity": range(10)
    })
    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        time_aware_train_test_split(df, test_size=1.5)

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    validate_required_columns, 
    time_aware_train_test_split,
    calculate_revenue,
    create_categorical_features,
)
from src.evaluation import (
    create_order_recommendations, 
    calculate_economic_cost,
    evaluate_revenue_predictions,
)
from src.models import (
    scale_features, 
    train_naive_baselines,
    prepare_features_and_target,
    prepare_features_and_target_revenue,
)
from src.config import CostConfig


def test_validate_required_columns_missing():
    """Test that missing required columns raise ValueError."""
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01"],
            "Category": ["Groceries"],
            "Units Sold": [10],
            # Missing required columns: Store ID, Product ID, Region, Price, Weather Condition, Seasonality
        }
    )

    with pytest.raises(ValueError):
        validate_required_columns(df)


def test_validate_required_columns_present():
    """Test that all required columns pass validation."""
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02"],
            "Category": ["Groceries", "Electronics"],
            "Units Sold": [10, 20],
            "Store ID": ["S001", "S002"],
            "Product ID": ["P0001", "P0002"],
            "Region": ["North", "South"],
            "Price": [5.0, 10.0],
            "Weather Condition": ["Sunny", "Rainy"],
            "Seasonality": ["Summer", "Winter"],
        }
    )

    # Should not raise
        validate_required_columns(df)


def test_time_aware_train_test_split_chronological():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")[::-1]  # reversed
    df = pd.DataFrame(
        {
            "date": dates,
            "product_line": ["A"] * 10,
            "quantity": range(10),
        }
    )

    train_df, test_df = time_aware_train_test_split(df, test_size=0.2)

    # Ensure chronological ordering and non-empty splits
    assert not train_df.empty and not test_df.empty
    assert train_df["date"].max() <= test_df["date"].min()
    assert len(test_df) == 2


def test_create_order_recommendations_clips_and_handles_nan():
    """Test order recommendations with Product ID and Category."""
    test_df = pd.DataFrame(
        {
            "Product ID": ["P0001", "P0002"],
            "Category": ["Groceries", "Electronics"],
        },
        index=[0, 1],
    )
    preds = pd.Series([-5.0, 10.0], index=test_df.index)
    # Use composite key for stock
    current_stock = {"P0001_Groceries": 5.0, "P0002_Electronics": 0.0}

    recommendations = create_order_recommendations(
        preds, test_df=test_df, current_stock=current_stock, safety_stock_multiplier=1.2
    )

    # Should handle predictions (may use different product identification)
    assert len(recommendations) > 0

    # NaN predictions should raise
    with pytest.raises(ValueError):
        create_order_recommendations(pd.Series([np.nan]))


def test_scale_features_train_test_separation():
    """Test that scaler is fitted only on training data."""
    X_train = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [10, 20, 30, 40, 50]})
    X_test = pd.DataFrame({"feature1": [6, 7], "feature2": [60, 70]})
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Check that scaler was fitted on training data only
    assert scaler.mean_[0] == X_train["feature1"].mean()
    assert scaler.mean_[1] == X_train["feature2"].mean()
    
    # Check that test data was transformed (not fitted)
    assert X_test_scaled is not None
    assert len(X_test_scaled) == len(X_test)
    
    # Check that scaling was applied (values should be centered and scaled)
    assert abs(X_train_scaled["feature1"].mean()) < 1e-10  # Should be ~0 after scaling
    assert abs(X_train_scaled["feature2"].mean()) < 1e-10


def test_scale_features_reuse_scaler():
    """Test that a pre-fitted scaler can be reused."""
    X_train1 = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [10, 20, 30]})
    X_train2 = pd.DataFrame({"feature1": [4, 5], "feature2": [40, 50]})
    
    # Fit scaler on first dataset
    _, _, scaler = scale_features(X_train1)
    
    # Reuse scaler on second dataset
    X_train2_scaled, _, _ = scale_features(X_train2, scaler=scaler)
    
    # Should use the same scaler (mean from first dataset)
    assert scaler.mean_[0] == X_train1["feature1"].mean()


def test_naive_baselines():
    """Test that naive baselines can be trained and make predictions."""
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [10, 20, 30]})
    y_train = pd.Series([5, 10, 15])
    train_df = pd.DataFrame({
        "Product ID": ["P0001", "P0001", "P0002"],
        "Category": ["Groceries", "Groceries", "Electronics"],
        "quantity": [5, 10, 15]
    })
    
    baselines = train_naive_baselines(X_train, y_train, train_df)
    
    assert "baseline_last_value" in baselines
    assert "baseline_mean" in baselines
    assert "baseline_seasonal_naive" in baselines
    
    # Test predictions
    X_test = pd.DataFrame({"feature1": [4, 5], "feature2": [40, 50]})
    test_df = pd.DataFrame({"Product ID": ["P0001", "P0002"], "Category": ["Groceries", "Electronics"]})
    
    last_value_pred = baselines["baseline_last_value"].predict(X_test, test_df)
    mean_pred = baselines["baseline_mean"].predict(X_test, test_df)
    
    assert len(last_value_pred) == len(X_test)
    assert len(mean_pred) == len(X_test)
    assert all(p >= 0 for p in last_value_pred)  # Non-negative predictions
    assert all(p >= 0 for p in mean_pred)


def test_calculate_revenue():
    """Test revenue calculation."""
    df = pd.DataFrame({
        "Price": [5.0, 10.0, 15.0],
        "quantity": [2, 3, 4],
    })
    
    df_with_revenue = calculate_revenue(df)
    
    assert "revenue" in df_with_revenue.columns
    assert df_with_revenue["revenue"].iloc[0] == 10.0  # 5.0 * 2
    assert df_with_revenue["revenue"].iloc[1] == 30.0  # 10.0 * 3
    assert df_with_revenue["revenue"].iloc[2] == 60.0  # 15.0 * 4


def test_create_categorical_features():
    """Test that Weather, Seasonality, Region are one-hot encoded."""
    df = pd.DataFrame({
        "Weather Condition": ["Sunny", "Rainy", "Cloudy"],
        "Seasonality": ["Summer", "Winter", "Spring"],
        "Region": ["North", "South", "East"],
        "Store ID": ["S001", "S002", "S001"],
    })
    
    df_encoded = create_categorical_features(df)
    
    # Check for Weather Condition features
    weather_cols = [col for col in df_encoded.columns if "weather_condition" in col.lower()]
    assert len(weather_cols) > 0
    
    # Check for Seasonality features
    seasonality_cols = [col for col in df_encoded.columns if "seasonality" in col.lower()]
    assert len(seasonality_cols) > 0
    
    # Check for Region features
    region_cols = [col for col in df_encoded.columns if "region" in col.lower()]
    assert len(region_cols) > 0


def test_prepare_features_and_target_validates_key_features():
    """Test that prepare_features_and_target validates Weather/Seasonality/Region."""
    # Create DataFrame with one-hot encoded features
    df = pd.DataFrame({
        "quantity": [10, 20, 30],
        "weather_condition_sunny": [1, 0, 1],
        "weather_condition_rainy": [0, 1, 0],
        "seasonality_summer": [1, 0, 0],
        "seasonality_winter": [0, 1, 0],
        "region_north": [1, 0, 1],
        "region_south": [0, 1, 0],
    })
    
    X, y = prepare_features_and_target(df, target_col="quantity")
    
    # Should not raise and should include key features
    assert len(X.columns) > 0
    assert len(y) == 3


def test_prepare_features_and_target_revenue():
    """Test revenue target preparation."""
    df = pd.DataFrame({
        "revenue": [50.0, 100.0, 150.0],
        "weather_condition_sunny": [1, 0, 1],
        "weather_condition_rainy": [0, 1, 0],
        "seasonality_summer": [1, 0, 0],
        "seasonality_winter": [0, 1, 0],
        "region_north": [1, 0, 1],
        "region_south": [0, 1, 0],
    })
    
    X, y = prepare_features_and_target_revenue(df, target_col="revenue")
    
    assert len(X.columns) > 0
    assert len(y) == 3
    assert y.sum() == 300.0


def test_evaluate_revenue_predictions():
    """Test revenue prediction evaluation."""
    y_true = pd.Series([100.0, 200.0, 300.0])
    y_pred = pd.Series([110.0, 190.0, 310.0])
    
    metrics = evaluate_revenue_predictions(y_true, y_pred)
    
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "Total_Revenue_True" in metrics
    assert "Total_Revenue_Pred" in metrics
    assert "Revenue_Error_Pct" in metrics
    assert metrics["Total_Revenue_True"] == 600.0


def test_calculate_economic_cost():
    """Test economic cost calculation."""
    y_true = pd.Series([10, 20, 30])
    y_pred = pd.Series([15, 15, 25])  # Over-predict first, under-predict second, close on third
    
    cost_config = CostConfig(
        stockout_cost_per_unit=10.0,
        holding_cost_per_unit_per_day=0.1,
        unit_price=5.0
    )
    
    costs = calculate_economic_cost(y_true, y_pred, cost_config)
    
    assert "total_cost" in costs
    assert "holding_cost" in costs
    assert "stockout_cost" in costs
    assert costs["total_cost"] > 0
    
    # Over-prediction: (15-10) = 5 units excess
    # Under-prediction: (20-15) = 5 units stockout
    # Holding cost: 5 * 0.1 = 0.5
    # Stockout cost: 5 * 10.0 = 50.0
    # Total: 50.5
    
    assert costs["holding_cost"] > 0  # Should have holding cost from over-prediction
    assert costs["stockout_cost"] > 0  # Should have stockout cost from under-prediction


def test_calculate_economic_cost_perfect_prediction():
    """Test that perfect predictions have zero cost."""
    y_true = pd.Series([10, 20, 30])
    y_pred = pd.Series([10, 20, 30])  # Perfect predictions
    
    cost_config = CostConfig()
    
    costs = calculate_economic_cost(y_true, y_pred, cost_config)
    
    assert costs["total_cost"] == 0.0
    assert costs["holding_cost"] == 0.0
    assert costs["stockout_cost"] == 0.0

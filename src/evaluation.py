"""
Evaluation and visualization module.

Calculates metrics, generates visualizations, and creates order recommendations
based on demand forecasts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import warnings


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Calculate MAE and RMSE metrics.

    MAE gives average error magnitude, RMSE penalizes larger errors more.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Dictionary with MAE and RMSE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {"MAE": mae, "RMSE": rmse}


def calculate_economic_cost(
    y_true: pd.Series, y_pred: pd.Series, cost_config
) -> Dict[str, float]:
    """
    Calculate dollar cost of prediction errors.

    Converts prediction errors to economic costs:
    - Over-prediction: holding cost for excess inventory
    - Under-prediction: stockout cost for missed sales

    Args:
        y_true: True target values
        y_pred: Predicted values
        cost_config: CostConfig instance with cost parameters

    Returns:
        Dictionary with total_cost, holding_cost, stockout_cost
    """
    # Ensure predictions are non-negative
    y_pred = pd.Series(y_pred).clip(lower=0)
    
    # Calculate over-prediction (excess inventory) and under-prediction (stockouts)
    over_pred = np.maximum(0, y_pred - y_true)
    under_pred = np.maximum(0, y_true - y_pred)
    
    # Calculate costs
    # Holding cost: cost of storing excess inventory (per unit per day)
    holding_cost = over_pred.sum() * cost_config.holding_cost_per_unit_per_day
    
    # Stockout cost: cost of missing sales (lost revenue + customer dissatisfaction)
    stockout_cost = under_pred.sum() * cost_config.stockout_cost_per_unit
    
    total_cost = holding_cost + stockout_cost
    
    return {
        "total_cost": total_cost,
        "holding_cost": holding_cost,
        "stockout_cost": stockout_cost,
        "over_prediction_units": over_pred.sum(),
        "under_prediction_units": under_pred.sum(),
    }


def save_metrics(
    metrics: Dict[str, float], model_name: str, save_dir: str = "results/metrics/"
):
    """
    Save evaluation metrics to a text file.

    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
        save_dir: Directory to save metrics
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    metrics_file = save_path / f"{model_name}_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 50 + "\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
        f.write("\n")
        f.write("MAE: Mean Absolute Error\n")
        f.write("RMSE: Root Mean Squared Error\n")
        # R² removed - not calculated anymore

    # Suppress print - metrics saved silently


def plot_sales_trends(
    df: pd.DataFrame,
    product_line: str = None,
    save_path: str = "results/figures/sales_trends.png",
):
    """
    Plot historical sales trends.

    Visualizes past sales patterns to understand demand behavior.

    Args:
        df: DataFrame with date index and quantity column
        product_line: Specific product to plot (if None, plots all)
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Helper to create product identifier
    def _get_product_identifier(row):
        """Get product identifier for display."""
        if "Product ID" in row and "Category" in row:
            return f"{row['Product ID']} ({row['Category']})"
        elif "product_line" in row:
            return row["product_line"]
        elif "Product ID" in row:
            return str(row["Product ID"])
        elif "Category" in row:
            return str(row["Category"])
        else:
            return "Unknown"

    # Reset index if needed to access columns
    df_plot = df.reset_index() if isinstance(df.index, pd.DatetimeIndex) else df.copy()
    
    # Create product identifier column
    if "Product ID" in df_plot.columns and "Category" in df_plot.columns:
        df_plot["product_identifier"] = df_plot.apply(_get_product_identifier, axis=1)
        product_col = "product_identifier"
    elif "product_line" in df_plot.columns:
        product_col = "product_line"
    else:
        # Fallback: use Category if available
        product_col = "Category" if "Category" in df_plot.columns else None

    if product_col and product_line:
        plot_df = df_plot[df_plot[product_col] == product_line].copy()
        if "date" in plot_df.columns:
            plot_df = plot_df.set_index("date")
        ax.plot(plot_df.index, plot_df["quantity"], label=product_line, linewidth=2)
    elif product_col:
        # Plot all products
        for product in df_plot[product_col].unique():
            product_df = df_plot[df_plot[product_col] == product].copy()
            if "date" in product_df.columns:
                product_df = product_df.set_index("date")
            ax.plot(
                product_df.index,
                product_df["quantity"],
                label=product,
                alpha=0.7,
                linewidth=1.5,
            )
    else:
        # No product identification, plot all data
        if "date" in df_plot.columns:
            df_plot = df_plot.set_index("date")
        ax.plot(df_plot.index, df_plot["quantity"], label="All Products", linewidth=2)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Quantity Sold", fontsize=12)
    ax.set_title("Historical Sales Trends", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Suppress plot save messages - too verbose for business users


def plot_predictions_vs_actual(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    save_path: str = "results/figures/predictions_vs_actual.png",
):
    """
    Plot predicted vs actual demand.

    Visualizes model performance by comparing predictions to actual values.

    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name of the model
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.6, s=50)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction"
    )
    ax1.set_xlabel("Actual Demand", fontsize=12)
    ax1.set_ylabel("Predicted Demand", fontsize=12)
    ax1.set_title(
        f"{model_name}: Predictions vs Actual", fontsize=14, fontweight="bold"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Time series plot
    dates = y_true.index if hasattr(y_true, "index") else range(len(y_true))
    ax2.plot(dates, y_true.values, label="Actual", linewidth=2, alpha=0.7)
    ax2.plot(dates, y_pred, label="Predicted", linewidth=2, alpha=0.7)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Demand", fontsize=12)
    ax2.set_title(
        f"{model_name}: Time Series Comparison", fontsize=14, fontweight="bold"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Suppress plot save messages


def calculate_reorder_urgency(
    current_stock: float, predicted_demand: float
) -> Tuple[str, float]:
    """
    Calculate reorder urgency level based on stock levels and predicted demand.

    Determines how urgent it is to reorder by calculating days until stockout.
    Urgency levels help prioritize which products need immediate attention.

    Args:
        current_stock: Current inventory level for the product
        predicted_demand: Predicted daily demand for the product

    Returns:
        Tuple of (urgency_level, days_until_stockout)
        urgency_level: "URGENT", "NORMAL", "LOW", or "POOR"
        days_until_stockout: Number of days until stock runs out
    """
    # Calculate days until stockout
    if predicted_demand <= 0:
        # No demand predicted, stock is sufficient
        return "POOR", 999.0

    days_until_stockout = current_stock / predicted_demand

    # Determine urgency level based on days until stockout
    if days_until_stockout <= 3:
        urgency = "URGENT"
    elif days_until_stockout <= 7:
        urgency = "NORMAL"
    elif days_until_stockout <= 14:
        urgency = "LOW"
    else:
        urgency = "POOR"

    return urgency, days_until_stockout


def create_order_recommendations(
    predictions: pd.Series,
    test_df: pd.DataFrame = None,
    current_stock: Dict[str, float] = None,
    safety_stock_multiplier: float = 1.2,
    target_days_of_coverage: float = 14.0,
) -> pd.DataFrame:
    """
    Create order recommendations based on predicted demand.

    Recommends ordering when predicted demand exceeds current stock,
    with a safety stock buffer to prevent stockouts. Includes urgency
    metrics to prioritize reordering decisions.

    Args:
        predictions: Predicted demand values
        test_df: Test DataFrame with Product ID and Category columns (if available)
        current_stock: Dictionary mapping product keys (Product ID + Category) to current stock levels
        safety_stock_multiplier: Multiplier for safety stock buffer
        target_days_of_coverage: Target days of inventory coverage

    Returns:
        DataFrame with order recommendations including urgency metrics
    """
    if isinstance(predictions, np.ndarray):
        predictions = pd.Series(predictions)

    if predictions.isna().any():
        raise ValueError("Predictions contain NaN values; cannot create recommendations.")

    predictions = predictions.clip(lower=0)

    recommendations = []

    # Helper function to create product key (import from data_loader)
    def _create_product_key_from_df(row):
        """Create composite product key from Product ID and Category."""
        if "Product ID" in row and "Category" in row:
            return f"{row['Product ID']}_{row['Category']}"
        elif "Product ID" in row:
            return str(row["Product ID"])
        elif "Category" in row:
            return str(row["Category"])
        else:
            return "default"

    # If we have Product ID and Category information in test_df
    if test_df is not None and "Product ID" in test_df.columns and "Category" in test_df.columns:
        # Group predictions by product (Product ID + Category)
        predictions_df = pd.DataFrame(
            {"predicted_demand": predictions.values}, index=test_df.index
        )
        predictions_df["Product ID"] = test_df["Product ID"].values
        predictions_df["Category"] = test_df["Category"].values
        
        # Create product key
        predictions_df["product_key"] = predictions_df.apply(_create_product_key_from_df, axis=1)

        for product_key in predictions_df["product_key"].unique():
            product_preds = predictions_df[
                predictions_df["product_key"] == product_key
            ]["predicted_demand"]
            avg_predicted_demand = product_preds.mean()

            current_stock_level = (
                current_stock.get(product_key, 0) if current_stock else 0
            )
            safety_stock = avg_predicted_demand * safety_stock_multiplier
            
            # Calculate target inventory level (days of coverage * daily demand)
            target_inventory = avg_predicted_demand * target_days_of_coverage
            
            # Order enough to reach target inventory, but at least maintain safety stock
            order_quantity = max(0, target_inventory - current_stock_level)

            # Calculate urgency level
            urgency, days_until_stockout = calculate_reorder_urgency(
                current_stock_level, avg_predicted_demand
            )

            # Get Product ID and Category for display
            product_row = predictions_df[predictions_df["product_key"] == product_key].iloc[0]
            product_display = f"{product_row['Product ID']} ({product_row['Category']})"

            recommendations.append(
                {
                    "product_line": product_display,
                    "product_key": product_key,
                    "current_stock": current_stock_level,
                    "predicted_demand": avg_predicted_demand,
                    "safety_stock": safety_stock,
                    "target_inventory": target_inventory,
                    "recommended_order": order_quantity,
                    "reorder_urgency": urgency,
                    "days_until_stockout": days_until_stockout,
                }
            )
    elif test_df is not None and "product_line" in test_df.columns:
        # Backward compatibility: handle old format
        predictions_df = pd.DataFrame(
            {"predicted_demand": predictions.values}, index=test_df.index
        )
        predictions_df["product_line"] = test_df["product_line"].values

        for product_line in predictions_df["product_line"].unique():
            product_preds = predictions_df[
                predictions_df["product_line"] == product_line
            ]["predicted_demand"]
            avg_predicted_demand = product_preds.mean()

            current_stock_level = (
                current_stock.get(product_line, 0) if current_stock else 0
            )
            safety_stock = avg_predicted_demand * safety_stock_multiplier
            
            # Calculate target inventory level (days of coverage * daily demand)
            target_inventory = avg_predicted_demand * target_days_of_coverage
            
            # Order enough to reach target inventory, but at least maintain safety stock
            order_quantity = max(0, target_inventory - current_stock_level)

            # Calculate urgency level
            urgency, days_until_stockout = calculate_reorder_urgency(
                current_stock_level, avg_predicted_demand
            )

            recommendations.append(
                {
                    "product_line": product_line,
                    "current_stock": current_stock_level,
                    "predicted_demand": avg_predicted_demand,
                    "safety_stock": safety_stock,
                    "target_inventory": target_inventory,
                    "recommended_order": order_quantity,
                    "reorder_urgency": urgency,
                    "days_until_stockout": days_until_stockout,
                }
            )
    else:
        # Simple case: aggregated predictions
        avg_predicted_demand = predictions.mean()
        current_stock_level = current_stock.get("default", 0) if current_stock else 0
        safety_stock = avg_predicted_demand * safety_stock_multiplier
        
        # Calculate target inventory level (days of coverage * daily demand)
        target_inventory = avg_predicted_demand * target_days_of_coverage
        
        # Order enough to reach target inventory, but at least maintain safety stock
        order_quantity = max(0, target_inventory - current_stock_level)

        # Calculate urgency level
        urgency, days_until_stockout = calculate_reorder_urgency(
            current_stock_level, avg_predicted_demand
        )

        recommendations.append(
            {
                "product_line": "All Products",
                "current_stock": current_stock_level,
                "predicted_demand": avg_predicted_demand,
                "safety_stock": safety_stock,
                "target_inventory": target_inventory,
                "recommended_order": order_quantity,
                "reorder_urgency": urgency,
                "days_until_stockout": days_until_stockout,
            }
        )

    recommendations_df = pd.DataFrame(recommendations)

    # Sort by urgency (URGENT first, then NORMAL, LOW, POOR)
    urgency_order = {"URGENT": 0, "NORMAL": 1, "LOW": 2, "POOR": 3}
    recommendations_df["urgency_rank"] = recommendations_df["reorder_urgency"].map(
        urgency_order
    )
    recommendations_df = recommendations_df.sort_values("urgency_rank").drop(
        "urgency_rank", axis=1
    )

    return recommendations_df


def save_order_recommendations(
    recommendations: pd.DataFrame,
    save_path: str = "results/metrics/order_recommendations.csv",
):
    """
    Save order recommendations to CSV file.

    Args:
        recommendations: DataFrame with order recommendations
        save_path: Path to save recommendations
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(save_path, index=False)
    # Suppress save messages


def plot_order_recommendations(
    recommendations: pd.DataFrame,
    save_path: str = "results/figures/order_recommendations.png",
):
    """
    Visualize order recommendations.

    Creates a bar chart showing recommended order quantities per product.

    Args:
        recommendations: DataFrame with order recommendations (must have product_line column)
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use product_line column (now contains Product ID + Category format)
    products = recommendations["product_line"]
    order_qty = recommendations["recommended_order"]

    bars = ax.bar(range(len(products)), order_qty, color="steelblue", alpha=0.7, edgecolor="black")

    ax.set_xlabel("Product", fontsize=12)
    ax.set_ylabel("Recommended Order Quantity", fontsize=12)
    ax.set_title(
        "Order Recommendations by Product", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(range(len(products)))
    ax.set_xticklabels(products, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Suppress plot save messages


def evaluate_and_visualize(
    model,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame = None,
    train_df: pd.DataFrame = None,
    output_config=None,
    is_best_model: bool = False,
    current_stock: Optional[Dict[str, float]] = None,
    safety_stock_multiplier: float = 1.2,
    target_days_of_coverage: float = 14.0,
    scaler=None,
):
    """
    Complete evaluation and visualization pipeline.

    Calculates metrics, creates visualizations, and generates order recommendations.
    Now controlled by output_config to prevent redundant file generation.

    Args:
        model: Trained model
        model_name: Name of the model
        X_test: Test features (will be scaled if scaler provided)
        y_test: Test target
        test_df: Test DataFrame with product_line column (for recommendations)
        train_df: Training data for trend visualization
        output_config: OutputConfig instance controlling what files to save
        is_best_model: Whether this is the best performing model
        current_stock: Dictionary mapping product_line to current stock levels
        scaler: Fitted scaler to transform test features (optional)
    """
    # Lazy import to avoid circular imports
    if output_config is None:
        from src.output_config import OutputConfig
        output_config = OutputConfig()  # Default: minimal files
    
    # Scale test features if scaler provided
    if scaler is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
        X_test = X_test_scaled
    
    # Make predictions
    y_pred = model.predict(X_test)
    if np.isnan(y_pred).any():
        raise ValueError(
            f"Model {model_name} produced NaN predictions; check preprocessing."
        )
    y_pred_series = pd.Series(y_pred, index=y_test.index).clip(lower=0)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred_series)
    # Suppress individual model metrics in main output (shown in comparison table)
    # Only print if verbose mode

    # Save metrics file only if configured (silently)
    should_save = output_config.should_save_model_files(model_name, is_best_model)
    if output_config.should_save_metrics_txt() and should_save:
        save_metrics(metrics, model_name)
        # Suppress print from save_metrics

    # Create visualizations based on configuration
    # Only generate plots that are configured and needed
    if output_config.should_save_plot_type("predictions", is_best_model):
        plot_predictions_vs_actual(
            y_test,
            y_pred_series,
            model_name,
            f"results/figures/{model_name}_predictions.png",
        )

    # Sales trends: Save ONCE (first model only) since it's based on training data (same for all models)
    if train_df is not None and output_config.should_save_plot_type("sales_trends", is_best_model):
        # Check if sales_trends already exists (from previous model)
        sales_trends_path = Path("results/figures/sales_trends.png")
        if not sales_trends_path.exists():
            plot_sales_trends(train_df, save_path="results/figures/sales_trends.png")
        # Silently skip if already exists

    # Generate diagnostic plots if configured
    if output_config.should_save_plot_type("residuals", is_best_model):
        plot_residuals(
            y_test, y_pred_series, model_name, f"results/figures/{model_name}_residuals.png"
        )
    
    if output_config.should_save_plot_type("feature_importance", is_best_model):
        analyze_feature_importance(model, X_test.columns, model_name)

    # Create order recommendations (always create in memory for final report)
    recommendations = create_order_recommendations(
        y_pred_series,
        test_df,
        current_stock,
        safety_stock_multiplier=safety_stock_multiplier,
    )
    
    # Only save individual recommendation files if configured
    if output_config.should_save_individual_recommendations() and should_save:
        save_order_recommendations(
            recommendations, f"results/metrics/{model_name}_recommendations.csv"
        )
    
    # Save recommendation plots if configured
    if output_config.should_save_plot_type("recommendations", is_best_model):
        plot_order_recommendations(
            recommendations, f"results/figures/{model_name}_recommendations.png"
        )

    return metrics, recommendations


def create_final_recommendations_report(
    recommendations: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """
    Create a comprehensive final order recommendations report.

    Formats recommendations into a clear, easy-to-read CSV file with
    all relevant information for decision-making. Includes urgency
    summary statistics.

    Args:
        recommendations: DataFrame with order recommendations from best model
        model_name: Name of the model used for predictions

    Returns:
        Formatted DataFrame ready for saving
    """
    # Create a copy to avoid modifying original
    report = recommendations.copy()

    # Round numerical values for readability
    report["current_stock"] = report["current_stock"].round(2)
    report["predicted_demand"] = report["predicted_demand"].round(2)
    report["safety_stock"] = report["safety_stock"].round(2)
    if "target_inventory" in report.columns:
        report["target_inventory"] = report["target_inventory"].round(2)
    report["recommended_order"] = report["recommended_order"].round(2)
    report["days_until_stockout"] = report["days_until_stockout"].round(2)

    # Reorder columns for clarity
    column_order = [
        "product_line",
        "reorder_urgency",
        "days_until_stockout",
        "current_stock",
        "predicted_demand",
        "safety_stock",
        "target_inventory",
        "recommended_order",
    ]
    report = report[[col for col in column_order if col in report.columns]]

    # Rename columns for better readability
    report.columns = [
        "Product Line",
        "Reorder Urgency",
        "Days Until Stockout",
        "Current Stock",
        "Predicted Daily Demand",
        "Safety Stock Target",
        "Target Inventory",
        "Recommended Order Quantity",
    ]

    # Calculate urgency summary
    urgency_counts = report["Reorder Urgency"].value_counts()
    urgent_count = urgency_counts.get("URGENT", 0)
    normal_count = urgency_counts.get("NORMAL", 0)
    low_count = urgency_counts.get("LOW", 0)
    poor_count = urgency_counts.get("POOR", 0)
    total_order_quantity = report["Recommended Order Quantity"].sum()

    # Save the report
    save_path = Path("results/metrics/final_order_recommendations.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Create summary header
    summary_lines = [
        f"# Order Recommendations Report - {model_name.upper()}",
        f"# Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "#",
        "# URGENCY SUMMARY:",
        f"#   URGENT: {urgent_count} products (stockout within 0-3 days)",
        f"#   NORMAL: {normal_count} products (stockout within 4-7 days)",
        f"#   LOW: {low_count} products (stockout within 8-14 days)",
        f"#   POOR: {poor_count} products (stockout beyond 14 days or sufficient stock)",
        "#",
        f"# Total Recommended Order Quantity: {total_order_quantity:.2f} units",
        "#",
        "#",
    ]

    # Write summary and data to file
    with open(save_path, "w") as f:
        f.write("\n".join(summary_lines))
        report.to_csv(f, index=False)

    # Report is saved to file; display is handled by main.py
    # Return report for main.py to display using formatted output

    return report


def plot_residuals(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    save_path: str = "results/figures/residuals.png",
):
    """
    Generate residual diagnostic plots for model evaluation.

    Creates three diagnostic plots to assess model assumptions:
    - Residuals vs predicted: checks for heteroscedasticity (non-constant variance)
    - Residuals vs actual: identifies systematic prediction errors
    - Q-Q plot: checks for normality of residuals

    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name of the model
        save_path: Path to save the figure
    """
    residuals = y_true.values - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, s=50)
    axes[0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Predicted Values", fontsize=12)
    axes[0].set_ylabel("Residuals", fontsize=12)
    axes[0].set_title("Residuals vs Predicted", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Residuals vs Actual
    axes[1].scatter(y_true.values, residuals, alpha=0.6, s=50)
    axes[1].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Actual Values", fontsize=12)
    axes[1].set_ylabel("Residuals", fontsize=12)
    axes[1].set_title("Residuals vs Actual", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # Q-Q Plot for normality
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot (Normality Check)", fontsize=14, fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"{model_name}: Residual Diagnostics", fontsize=16, fontweight="bold")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Suppress plot save messages


def analyze_feature_importance(
    model, feature_names: pd.Index, model_name: str, top_n: int = 10
):
    """
    Analyze and visualize feature importance for a model.

    Extracts feature importances or coefficients depending on model type:
    - Random Forest & Gradient Boosting: feature_importances_
    - Linear Regression: absolute value of coefficients

    Args:
        model: Trained model
        feature_names: Names of features (from DataFrame columns)
        model_name: Name of the model
        top_n: Number of top features to display
    """
    # Extract feature importance based on model type
    if hasattr(model, "feature_importances_"):
        # Random Forest or Gradient Boosting
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Linear Regression
        importances = np.abs(model.coef_)
    else:
        print(f"Warning: Model {model_name} does not support feature importance")
        return

    # Create DataFrame for easier handling
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    )
    feature_importance_df = feature_importance_df.sort_values(
        "importance", ascending=False
    ).head(top_n)

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        range(len(feature_importance_df)),
        feature_importance_df["importance"],
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )

    ax.set_yticks(range(len(feature_importance_df)))
    ax.set_yticklabels(feature_importance_df["feature"])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(
        f"{model_name}: Top {top_n} Feature Importances",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (idx, row) in enumerate(feature_importance_df.iterrows()):
        ax.text(
            row["importance"],
            i,
            f"{row['importance']:.4f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    plt.tight_layout()

    save_path = Path(f"results/figures/{model_name}_feature_importance.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Suppress plot save messages


def interpret_model_performance(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    comparison_df: pd.DataFrame,
    save_path: str = "results/metrics/model_interpretation.txt",
):
    """
    Analyze and interpret model performance to understand why certain models perform better.

    Investigates feature correlations, data characteristics, and model complexity
    to explain performance differences. Helps understand why Linear Regression
    might outperform ensemble methods.

    Args:
        models: Dictionary of trained models
        X_train: Training features
        y_train: Training target
        comparison_df: DataFrame with model comparison metrics
        save_path: Path to save interpretation report
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("MODEL PERFORMANCE INTERPRETATION")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Best model identification
    best_model_name = comparison_df.loc[comparison_df["RMSE"].idxmin(), "Model"]
    report_lines.append(f"Best Model: {best_model_name}")
    report_lines.append(f"  MAE: {comparison_df.loc[comparison_df['RMSE'].idxmin(), 'MAE']:.4f}")
    report_lines.append(f"  RMSE: {comparison_df.loc[comparison_df['RMSE'].idxmin(), 'RMSE']:.4f}")
    # R² removed - not calculated anymore
    report_lines.append("")

    # Data size analysis
    n_samples = len(X_train)
    n_features = len(X_train.columns)
    samples_per_feature = n_samples / n_features if n_features > 0 else 0

    report_lines.append("DATA CHARACTERISTICS:")
    report_lines.append(f"  Number of samples: {n_samples}")
    report_lines.append(f"  Number of features: {n_features}")
    report_lines.append(f"  Samples per feature: {samples_per_feature:.2f}")
    report_lines.append("")

    # Feature correlation analysis
    report_lines.append("FEATURE CORRELATION ANALYSIS:")
    corr_matrix = X_train.corr().abs()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.7:
                high_corr_pairs.append(
                    (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                )

    if high_corr_pairs:
        report_lines.append(f"  Found {len(high_corr_pairs)} highly correlated feature pairs (>0.7):")
        for feat1, feat2, corr in high_corr_pairs[:5]:  # Show top 5
            report_lines.append(f"    {feat1} <-> {feat2}: {corr:.3f}")
    else:
        report_lines.append("  No highly correlated features found (all < 0.7)")
    report_lines.append("")

    # Model complexity analysis
    report_lines.append("MODEL COMPLEXITY ANALYSIS:")
    report_lines.append("  Linear Regression:")
    report_lines.append(f"    Parameters: {n_features + 1} (coefficients + intercept)")
    report_lines.append(f"    Complexity: Low (linear relationship)")

    report_lines.append("  Random Forest:")
    rf_model = models.get("random_forest")
    if rf_model:
        n_trees = rf_model.n_estimators
        max_depth = rf_model.max_depth
        report_lines.append(f"    Parameters: ~{n_trees * (2**max_depth)} (estimated)")
        report_lines.append(f"    Complexity: High (non-linear, ensemble)")

    report_lines.append("  Gradient Boosting:")
    gb_model = models.get("gradient_boosting")
    if gb_model:
        n_estimators = gb_model.n_estimators
        max_depth = gb_model.max_depth
        report_lines.append(f"    Parameters: ~{n_estimators * (2**max_depth)} (estimated)")
        report_lines.append(f"    Complexity: Very High (sequential learning)")
    report_lines.append("")

    # Why Linear Regression might perform best
    report_lines.append("POSSIBLE REASONS FOR LINEAR REGRESSION PERFORMANCE:")
    if samples_per_feature < 50:
        report_lines.append(
            f"  - Limited data ({samples_per_feature:.1f} samples/feature): "
            "Complex models may overfit"
        )
    if len(high_corr_pairs) == 0:
        report_lines.append(
            "  - Low feature correlation: Linear relationships may be sufficient"
        )
    report_lines.append(
        "  - Ensemble methods may be overfitting on small dataset"
    )
    report_lines.append(
        "  - Linear relationships may capture most of the signal"
    )
    report_lines.append("")

    # Recommendations
    report_lines.append("RECOMMENDATIONS:")
    if best_model_name == "linear_regression":
        report_lines.append(
            "  - Linear Regression performs best, suggesting linear relationships"
        )
        report_lines.append(
            "  - Consider feature engineering to capture non-linear patterns"
        )
        report_lines.append(
            "  - With more data, ensemble methods might improve"
        )
    else:
        report_lines.append(
            f"  - {best_model_name} performs best, capturing non-linear patterns"
        )
    report_lines.append("  - Consider collecting more data for better model performance")
    report_lines.append("  - Feature selection might improve all models")
    report_lines.append("")

    report_lines.append("=" * 70)

    # Save report
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(report_lines))

    # Suppress save messages


def analyze_per_product_performance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
    model_name: str,
    scaler=None,
) -> pd.DataFrame:
    """
    Analyze model performance for each product line separately.

    Identifies which products are predicted best and worst, helping
    target improvements for specific product categories.

    Args:
        model: Trained model
        X_test: Test features (will be scaled if scaler provided)
        y_test: Test target
        test_df: Test DataFrame with product_line column
        model_name: Name of the model
        scaler: Fitted scaler to transform test features (optional)

    Returns:
        DataFrame with per-product performance metrics
    """
    # Scale test features if scaler provided
    if scaler is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
        X_test = X_test_scaled
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_series = pd.Series(y_pred, index=y_test.index)

    # Helper function to create product key
    def _create_product_key_from_df(row):
        """Create composite product key from Product ID and Category."""
        if "Product ID" in row and "Category" in row:
            return f"{row['Product ID']}_{row['Category']}"
        elif "Product ID" in row:
            return str(row["Product ID"])
        elif "Category" in row:
            return str(row["Category"])
        else:
            return "default"

    # Group by product if Product ID and Category available
    if test_df is not None and "Product ID" in test_df.columns and "Category" in test_df.columns:
        # Create DataFrame with predictions and actuals
        results_df = pd.DataFrame(
            {
                "Product ID": test_df["Product ID"].values,
                "Category": test_df["Category"].values,
                "actual": y_test.values,
                "predicted": y_pred_series.values,
            }
        )
        
        # Create product key
        results_df["product_key"] = results_df.apply(_create_product_key_from_df, axis=1)
        results_df["product_line"] = results_df["Product ID"] + " (" + results_df["Category"] + ")"

        # Calculate metrics per product
        per_product_metrics = []
        for product_key in results_df["product_key"].unique():
            product_data = results_df[results_df["product_key"] == product_key]
            product_actual = product_data["actual"]
            product_pred = product_data["predicted"]

            mae = mean_absolute_error(product_actual, product_pred)
            rmse = np.sqrt(mean_squared_error(product_actual, product_pred))
            
            # Get display name
            product_display = product_data["product_line"].iloc[0]

            per_product_metrics.append(
                {
                    "product_line": product_display,
                    "MAE": mae,
                    "RMSE": rmse,
                    "n_samples": len(product_data),
                }
            )

        performance_df = pd.DataFrame(per_product_metrics)
        performance_df = performance_df.sort_values("RMSE")

        # Save to CSV
        save_path = Path("results/metrics/per_product_performance.csv")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        performance_df.to_csv(save_path, index=False)
        # Suppress save messages

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        products = performance_df["product_line"]
        rmse_values = performance_df["RMSE"]

        # Color-code by performance (lower is better)
        colors = [
            "green" if rmse < rmse_values.quantile(0.33)
            else "orange" if rmse < rmse_values.quantile(0.67)
            else "red"
            for rmse in rmse_values
        ]

        bars = ax.barh(products, rmse_values, color=colors, alpha=0.7, edgecolor="black")

        ax.set_xlabel("RMSE", fontsize=12)
        ax.set_ylabel("Product Line", fontsize=12)
        ax.set_title(
            f"{model_name}: Per-Product Performance (RMSE)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for i, (idx, row) in enumerate(performance_df.iterrows()):
            ax.text(
                row["RMSE"],
                i,
                f"{row['RMSE']:.2f}",
                va="center",
                ha="left",
                fontsize=10,
            )

        plt.tight_layout()

        fig_path = Path("results/figures/per_product_performance.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Suppress plot save messages

        return performance_df
    elif test_df is not None and "product_line" in test_df.columns:
        # Backward compatibility: handle old format
        results_df = pd.DataFrame(
            {
                "product_line": test_df["product_line"].values,
                "actual": y_test.values,
                "predicted": y_pred_series.values,
            }
        )

        # Calculate metrics per product
        per_product_metrics = []
        for product in results_df["product_line"].unique():
            product_data = results_df[results_df["product_line"] == product]
            product_actual = product_data["actual"]
            product_pred = product_data["predicted"]

            mae = mean_absolute_error(product_actual, product_pred)
            rmse = np.sqrt(mean_squared_error(product_actual, product_pred))

            per_product_metrics.append(
                {
                    "product_line": product,
                    "MAE": mae,
                    "RMSE": rmse,
                    "n_samples": len(product_data),
                }
            )

        performance_df = pd.DataFrame(per_product_metrics)
        performance_df = performance_df.sort_values("RMSE")

        # Save to CSV
        save_path = Path("results/metrics/per_product_performance.csv")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        performance_df.to_csv(save_path, index=False)

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        products = performance_df["product_line"]
        rmse_values = performance_df["RMSE"]

        colors = [
            "green" if rmse < rmse_values.quantile(0.33)
            else "orange" if rmse < rmse_values.quantile(0.67)
            else "red"
            for rmse in rmse_values
        ]

        bars = ax.barh(products, rmse_values, color=colors, alpha=0.7, edgecolor="black")

        ax.set_xlabel("RMSE", fontsize=12)
        ax.set_ylabel("Product Line", fontsize=12)
        ax.set_title(
            f"{model_name}: Per-Product Performance (RMSE)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")

        for i, (idx, row) in enumerate(performance_df.iterrows()):
            ax.text(
                row["RMSE"],
                i,
                f"{row['RMSE']:.2f}",
                va="center",
                ha="left",
                fontsize=10,
            )

        plt.tight_layout()

        fig_path = Path("results/figures/per_product_performance.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

        return performance_df
    else:
        # Silently handle missing product identification columns
        return pd.DataFrame()


def evaluate_revenue_predictions(
    y_true: pd.Series, y_pred: pd.Series
) -> Dict[str, float]:
    """
    Calculate revenue prediction metrics.

    Args:
        y_true: True revenue values
        y_pred: Predicted revenue values

    Returns:
        Dictionary with MAE, RMSE, and total revenue metrics
    """
    y_pred = pd.Series(y_pred).clip(lower=0)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    total_revenue_true = y_true.sum()
    total_revenue_pred = y_pred.sum()
    revenue_error_pct = abs(total_revenue_pred - total_revenue_true) / total_revenue_true * 100 if total_revenue_true > 0 else 0.0
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "Total_Revenue_True": total_revenue_true,
        "Total_Revenue_Pred": total_revenue_pred,
        "Revenue_Error_Pct": revenue_error_pct,
    }


def create_revenue_forecast_report(
    predictions: pd.Series,
    test_df: pd.DataFrame = None,
    model_name: str = "best_model",
) -> pd.DataFrame:
    """
    Generate revenue forecast report with breakdowns by Weather/Seasonality/Region.

    Args:
        predictions: Predicted revenue values
        test_df: Test DataFrame with Weather, Seasonality, Region columns
        model_name: Name of the model used

    Returns:
        DataFrame with revenue forecasts and breakdowns
    """
    if isinstance(predictions, np.ndarray):
        predictions = pd.Series(predictions)
    
    predictions = predictions.clip(lower=0)
    
    report_data = []
    
    if test_df is not None:
        # Create DataFrame with predictions and dimensions
        forecast_df = pd.DataFrame(
            {"predicted_revenue": predictions.values}, index=test_df.index
        )
        
        # Add dimension columns if available
        for col in ["Weather Condition", "Seasonality", "Region", "Store ID", "Product ID", "Category"]:
            if col in test_df.columns:
                forecast_df[col] = test_df[col].values
        
        # Overall summary
        report_data.append({
            "Dimension": "Overall",
            "Value": "All",
            "Predicted_Revenue": forecast_df["predicted_revenue"].sum(),
            "Count": len(forecast_df),
        })
        
        # Breakdown by Weather Condition
        if "Weather Condition" in forecast_df.columns:
            weather_breakdown = forecast_df.groupby("Weather Condition")["predicted_revenue"].agg(["sum", "count"])
            for weather, row in weather_breakdown.iterrows():
                report_data.append({
                    "Dimension": "Weather Condition",
                    "Value": weather,
                    "Predicted_Revenue": row["sum"],
                    "Count": int(row["count"]),
                })
        
        # Breakdown by Seasonality
        if "Seasonality" in forecast_df.columns:
            season_breakdown = forecast_df.groupby("Seasonality")["predicted_revenue"].agg(["sum", "count"])
            for season, row in season_breakdown.iterrows():
                report_data.append({
                    "Dimension": "Seasonality",
                    "Value": season,
                    "Predicted_Revenue": row["sum"],
                    "Count": int(row["count"]),
                })
        
        # Breakdown by Region
        if "Region" in forecast_df.columns:
            region_breakdown = forecast_df.groupby("Region")["predicted_revenue"].agg(["sum", "count"])
            for region, row in region_breakdown.iterrows():
                report_data.append({
                    "Dimension": "Region",
                    "Value": region,
                    "Predicted_Revenue": row["sum"],
                    "Count": int(row["count"]),
                })
    else:
        # Simple case: no breakdowns
        report_data.append({
            "Dimension": "Overall",
            "Value": "All",
            "Predicted_Revenue": predictions.sum(),
            "Count": len(predictions),
        })
    
    report_df = pd.DataFrame(report_data)
    
    # Save report
    save_path = Path("results/metrics/revenue_forecast_report.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(save_path, index=False)
    
    return report_df


def plot_revenue_forecasts(
    y_true: pd.Series,
    y_pred: pd.Series,
    test_df: pd.DataFrame = None,
    model_name: str = "best_model",
    save_path: str = "results/figures/revenue_forecasts.png",
):
    """
    Visualize revenue predictions with breakdowns by Weather/Seasonality/Region.

    Args:
        y_true: True revenue values
        y_pred: Predicted revenue values
        test_df: Test DataFrame with Weather, Seasonality, Region columns
        model_name: Name of the model
        save_path: Path to save the figure
    """
    y_pred = pd.Series(y_pred).clip(lower=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Time series comparison
    dates = y_true.index if hasattr(y_true, "index") else range(len(y_true))
    axes[0, 0].plot(dates, y_true.values, label="Actual Revenue", linewidth=2, alpha=0.7)
    axes[0, 0].plot(dates, y_pred, label="Predicted Revenue", linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel("Time", fontsize=12)
    axes[0, 0].set_ylabel("Revenue", fontsize=12)
    axes[0, 0].set_title(f"{model_name}: Revenue Time Series", fontsize=14, fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    axes[0, 1].scatter(y_true, y_pred, alpha=0.6, s=50)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
    axes[0, 1].set_xlabel("Actual Revenue", fontsize=12)
    axes[0, 1].set_ylabel("Predicted Revenue", fontsize=12)
    axes[0, 1].set_title(f"{model_name}: Revenue Predictions vs Actual", fontsize=14, fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Breakdown by Weather Condition (if available)
    if test_df is not None and "Weather Condition" in test_df.columns:
        forecast_df = pd.DataFrame({"revenue": y_pred.values}, index=test_df.index)
        forecast_df["Weather Condition"] = test_df["Weather Condition"].values
        weather_revenue = forecast_df.groupby("Weather Condition")["revenue"].sum()
        axes[1, 0].bar(weather_revenue.index, weather_revenue.values, color="steelblue", alpha=0.7)
        axes[1, 0].set_xlabel("Weather Condition", fontsize=12)
        axes[1, 0].set_ylabel("Total Predicted Revenue", fontsize=12)
        axes[1, 0].set_title("Revenue by Weather Condition", fontsize=14, fontweight="bold")
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis="y")
    else:
        axes[1, 0].text(0.5, 0.5, "Weather Condition data not available", 
                       ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("Revenue by Weather Condition", fontsize=14, fontweight="bold")
    
    # Plot 4: Breakdown by Region (if available)
    if test_df is not None and "Region" in test_df.columns:
        forecast_df = pd.DataFrame({"revenue": y_pred.values}, index=test_df.index)
        forecast_df["Region"] = test_df["Region"].values
        region_revenue = forecast_df.groupby("Region")["revenue"].sum()
        axes[1, 1].bar(region_revenue.index, region_revenue.values, color="steelblue", alpha=0.7)
        axes[1, 1].set_xlabel("Region", fontsize=12)
        axes[1, 1].set_ylabel("Total Predicted Revenue", fontsize=12)
        axes[1, 1].set_title("Revenue by Region", fontsize=14, fontweight="bold")
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis="y")
    else:
        axes[1, 1].text(0.5, 0.5, "Region data not available", 
                       ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Revenue by Region", fontsize=14, fontweight="bold")
    
    plt.suptitle(f"{model_name}: Revenue Forecast Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_weather_seasonality_region_impact(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
    model_name: str,
    scaler=None,
) -> pd.DataFrame:
    """
    Analyze how Weather, Seasonality, and Region affect predictions.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        test_df: Test DataFrame with Weather, Seasonality, Region columns
        model_name: Name of the model
        scaler: Fitted scaler (optional)

    Returns:
        DataFrame with impact analysis
    """
    # Scale test features if scaler provided
    if scaler is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
        X_test = X_test_scaled
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_series = pd.Series(y_pred, index=y_test.index).clip(lower=0)
    
    # Create analysis DataFrame
    analysis_df = pd.DataFrame(
        {"actual": y_test.values, "predicted": y_pred_series.values},
        index=test_df.index
    )
    
    # Add dimension columns
    for col in ["Weather Condition", "Seasonality", "Region"]:
        if col in test_df.columns:
            analysis_df[col] = test_df[col].values
    
    # Calculate metrics by dimension
    impact_data = []
    
    # Overall metrics
    mae_overall = mean_absolute_error(analysis_df["actual"], analysis_df["predicted"])
    rmse_overall = np.sqrt(mean_squared_error(analysis_df["actual"], analysis_df["predicted"]))
    impact_data.append({
        "Dimension": "Overall",
        "Value": "All",
        "MAE": mae_overall,
        "RMSE": rmse_overall,
        "Mean_Actual": analysis_df["actual"].mean(),
        "Mean_Predicted": analysis_df["predicted"].mean(),
    })
    
    # By Weather Condition
    if "Weather Condition" in analysis_df.columns:
        for weather in analysis_df["Weather Condition"].unique():
            weather_data = analysis_df[analysis_df["Weather Condition"] == weather]
            mae = mean_absolute_error(weather_data["actual"], weather_data["predicted"])
            rmse = np.sqrt(mean_squared_error(weather_data["actual"], weather_data["predicted"]))
            impact_data.append({
                "Dimension": "Weather Condition",
                "Value": weather,
                "MAE": mae,
                "RMSE": rmse,
                "Mean_Actual": weather_data["actual"].mean(),
                "Mean_Predicted": weather_data["predicted"].mean(),
            })
    
    # By Seasonality
    if "Seasonality" in analysis_df.columns:
        for season in analysis_df["Seasonality"].unique():
            season_data = analysis_df[analysis_df["Seasonality"] == season]
            mae = mean_absolute_error(season_data["actual"], season_data["predicted"])
            rmse = np.sqrt(mean_squared_error(season_data["actual"], season_data["predicted"]))
            impact_data.append({
                "Dimension": "Seasonality",
                "Value": season,
                "MAE": mae,
                "RMSE": rmse,
                "Mean_Actual": season_data["actual"].mean(),
                "Mean_Predicted": season_data["predicted"].mean(),
            })
    
    # By Region
    if "Region" in analysis_df.columns:
        for region in analysis_df["Region"].unique():
            region_data = analysis_df[analysis_df["Region"] == region]
            mae = mean_absolute_error(region_data["actual"], region_data["predicted"])
            rmse = np.sqrt(mean_squared_error(region_data["actual"], region_data["predicted"]))
            impact_data.append({
                "Dimension": "Region",
                "Value": region,
                "MAE": mae,
                "RMSE": rmse,
                "Mean_Actual": region_data["actual"].mean(),
                "Mean_Predicted": region_data["predicted"].mean(),
            })
    
    impact_df = pd.DataFrame(impact_data)
    
    # Save analysis
    save_path = Path("results/metrics/weather_seasonality_region_impact.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    impact_df.to_csv(save_path, index=False)
    
    return impact_df

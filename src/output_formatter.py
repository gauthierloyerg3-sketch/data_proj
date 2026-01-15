"""
Output formatting utilities for clean, business-friendly terminal output.

Provides functions for formatting tables, sections, and summaries
suitable for non-technical users (supermarket owners).
"""

import pandas as pd
from typing import Dict, Any, Optional, List
import sys


def format_number(num: float, decimals: int = 2) -> str:
    """Format number with commas and specified decimal places."""
    if pd.isna(num):
        return "N/A"
    return f"{num:,.{decimals}f}"


def format_currency(amount: float) -> str:
    """Format amount as currency."""
    if pd.isna(amount):
        return "N/A"
    return f"${amount:,.2f}"


def print_header(title: str, width: int = 70, char: str = "=") -> None:
    """Print a centered header with border."""
    border = char * width
    padding = (width - len(title) - 2) // 2
    print()
    print(border)
    print(f"{' ' * padding} {title} {' ' * (width - padding - len(title) - 3)}")
    print(border)
    print()


def print_section(title: str, level: int = 1, width: int = 70) -> None:
    """Print a section header with visual hierarchy."""
    if level == 1:
        # Main section
        print()
        print("=" * width)
        print(f" {title}")
        print("=" * width)
    elif level == 2:
        # Subsection
        print()
        print(f"{title}")
        print("-" * min(len(title), width))
    else:
        # Minor section
        print(f"\n{title}:")
    print()


def print_progress(message: str, done: bool = False) -> None:
    """Print a progress message with checkmark when done."""
    if done:
        print(f"✓ {message}")
        print()  # Add newline after completion
    else:
        print(f"  {message}...", end="", flush=True)


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✓ {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠ {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"✗ ERROR: {message}")


def print_table(
    data: pd.DataFrame,
    title: Optional[str] = None,
    highlight_rows: Optional[List[int]] = None,
    max_width: int = 100,
) -> None:
    """
    Print a DataFrame as a clean formatted table.
    
    Args:
        data: DataFrame to display
        title: Optional title for the table
        highlight_rows: List of row indices to highlight (for urgent items)
        max_width: Maximum table width
    """
    if data.empty:
        print("  (No data to display)")
        return
    
    if title:
        print(f"\n{title}")
        print("-" * min(len(title), max_width))
    
    # Format the DataFrame for display
    display_df = data.copy()
    
    # Remove R² column if present (to reduce trustability concerns)
    if "R2" in display_df.columns:
        display_df = display_df.drop(columns=["R2"])
    if "R²" in display_df.columns:
        display_df = display_df.drop(columns=["R²"])
    
    # Format numeric columns
    for col in display_df.columns:
        if display_df[col].dtype in ['float64', 'int64']:
            if 'cost' in col.lower() or 'price' in col.lower():
                display_df[col] = display_df[col].apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")
            elif 'quantity' in col.lower() or 'stock' in col.lower() or 'order' in col.lower():
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
            elif 'days' in col.lower():
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f} days" if pd.notna(x) else "N/A")
            else:
                display_df[col] = display_df[col].apply(lambda x: format_number(x, 2) if pd.notna(x) else "N/A")
    
    # Print table using pandas to_string with better formatting
    print(display_df.to_string(index=False))
    print()


def print_recommendations_table(recommendations: pd.DataFrame, use_emoji: bool = False) -> None:
    """
    Print order recommendations in a clean, business-friendly format.
    
    Args:
        recommendations: DataFrame with order recommendations
        use_emoji: Whether to use emoji icons for urgency
    """
    if recommendations.empty:
        print("  No recommendations available.")
        return
    
    # Create display DataFrame with selected columns
    display_cols = []
    col_mapping = {}
    
    if "product_line" in recommendations.columns:
        display_cols.append("product_line")
        col_mapping["product_line"] = "Product"
    
    if "reorder_urgency" in recommendations.columns:
        display_cols.append("reorder_urgency")
        col_mapping["reorder_urgency"] = "Urgency"
    
    if "days_until_stockout" in recommendations.columns:
        display_cols.append("days_until_stockout")
        col_mapping["days_until_stockout"] = "Days Left"
    
    if "current_stock" in recommendations.columns:
        display_cols.append("current_stock")
        col_mapping["current_stock"] = "Current Stock"
    
    if "recommended_order" in recommendations.columns:
        display_cols.append("recommended_order")
        col_mapping["recommended_order"] = "Order Quantity"
    
    if not display_cols:
        print_table(recommendations)
        return
    
    display_df = recommendations[display_cols].copy()
    display_df = display_df.rename(columns=col_mapping)
    
    # Format columns
    if "Current Stock" in display_df.columns:
        display_df["Current Stock"] = display_df["Current Stock"].apply(
            lambda x: f"{x:,.0f} units" if pd.notna(x) and x >= 0 else "0 units"
        )
    
    if "Order Quantity" in display_df.columns:
        display_df["Order Quantity"] = display_df["Order Quantity"].apply(
            lambda x: f"{x:,.0f} units" if pd.notna(x) and x > 0 else "0 units"
        )
    
    if "Days Left" in display_df.columns:
        display_df["Days Left"] = display_df["Days Left"].apply(
            lambda x: f"{x:.1f} days" if pd.notna(x) and x < 999 else "Sufficient"
        )
    
    if "Urgency" in display_df.columns:
        # Add visual indicators for urgency
        if use_emoji:
            urgency_icons = {
                "URGENT": "🔴 URGENT",
                "NORMAL": "🟡 NORMAL",
                "LOW": "🟢 LOW",
                "POOR": "⚪ POOR"
            }
        else:
            urgency_icons = {
                "URGENT": "URGENT",
                "NORMAL": "NORMAL",
                "LOW": "LOW",
                "POOR": "POOR"
            }
        display_df["Urgency"] = display_df["Urgency"].apply(
            lambda x: urgency_icons.get(x, x) if pd.notna(x) else "N/A"
        )
    
    # Sort by urgency (URGENT first)
    if "Urgency" in display_df.columns:
        urgency_order = {"URGENT": 0, "NORMAL": 1, "LOW": 2, "POOR": 3}
        # Extract urgency level (handle both with and without emoji)
        def get_urgency_level(urgency_str):
            for level in urgency_order.keys():
                if level in str(urgency_str):
                    return urgency_order[level]
            return 99
        
        display_df["_urgency_order"] = display_df["Urgency"].apply(get_urgency_level)
        display_df = display_df.sort_values("_urgency_order").drop("_urgency_order", axis=1)
    
    print_section("ORDER RECOMMENDATIONS", level=2)
    print_table(display_df, title=None)


def print_urgency_summary(urgency_counts: Dict[str, int], total_order_qty: float = 0.0, use_emoji: bool = False) -> None:
    """
    Print a summary of urgency levels and total order quantity.
    
    Args:
        urgency_counts: Dictionary mapping urgency levels to counts
        total_order_qty: Total recommended order quantity
        use_emoji: Whether to use emoji icons
    """
    urgent = urgency_counts.get("URGENT", 0)
    normal = urgency_counts.get("NORMAL", 0)
    low = urgency_counts.get("LOW", 0)
    poor = urgency_counts.get("POOR", 0)
    total = urgent + normal + low + poor
    
    if use_emoji:
        urgent_icon = "🔴"
        normal_icon = "🟡"
        low_icon = "🟢"
        poor_icon = "⚪"
    else:
        urgent_icon = normal_icon = low_icon = poor_icon = ""
    
    print(f"\n  Total Products Analyzed: {total}")
    if urgent > 0:
        print(f"  {urgent_icon} Urgent Orders: {urgent} products (order immediately!)")
    if normal > 0:
        print(f"  {normal_icon} Normal Orders: {normal} products (order soon)")
    if low > 0:
        print(f"  {low_icon} Low Priority: {low} products (plan order)")
    if poor > 0:
        print(f"  {poor_icon} Sufficient Stock: {poor} products (monitor)")
    
    if total_order_qty > 0:
        print(f"  Total Recommended Order Quantity: {total_order_qty:,.0f} units")
    
    print()


def print_summary_box(title: str, items: Dict[str, Any], width: int = 70) -> None:
    """
    Print a summary box with key metrics.
    
    Args:
        title: Title of the summary box
        items: Dictionary of key-value pairs to display
        width: Width of the box
    """
    print()
    print("┌" + "─" * (width - 2) + "┐")
    print(f"│ {title:<{width-4}} │")
    print("├" + "─" * (width - 2) + "┤")
    
    for key, value in items.items():
        if isinstance(value, float):
            if value >= 1000:
                value_str = format_number(value, 0)
            else:
                value_str = format_number(value, 2)
        elif isinstance(value, (int, str)):
            value_str = str(value)
        else:
            value_str = str(value)
        
        # Calculate spacing for proper alignment
        key_width = min(len(key), width - len(value_str) - 8)
        spacing = width - 6 - key_width - len(value_str)
        print(f"│  {key[:key_width]:<{key_width}} {' ' * spacing}{value_str} │")
    
    print("└" + "─" * (width - 2) + "┘")
    print()


def print_executive_summary(
    total_products: int,
    urgency_counts: Dict[str, int],
    total_order_qty: float,
    total_cost: Optional[float] = None,
    model_name: Optional[str] = None,
    model_rmse: Optional[float] = None,
    use_emoji: bool = False,
) -> None:
    """
    Print an executive summary for business owners.
    
    Args:
        total_products: Total number of products analyzed
        urgency_counts: Dictionary mapping urgency levels to counts
        total_order_qty: Total recommended order quantity
        total_cost: Optional total cost estimate
        model_name: Name of the model used for recommendations
        model_rmse: RMSE value of the model used
        use_emoji: Whether to use emoji icons
    """
    title = "RETAIL DEMAND FORECASTING - ORDER RECOMMENDATIONS"
    print_header(title, width=70)
    
    summary_title = "EXECUTIVE SUMMARY"
    
    summary_items = {
        "Total Products": total_products,
        "Urgent Orders": f"{urgency_counts.get('URGENT', 0)} products",
        "Total Order Quantity": f"{total_order_qty:,.0f} units",
    }
    
    if model_name is not None:
        summary_items["Model Used"] = model_name
    
    if model_rmse is not None:
        summary_items["Model Performance (RMSE)"] = format_number(model_rmse, 4)
    
    if total_cost is not None:
        summary_items["Estimated Total Cost"] = format_currency(total_cost)
    
    print_summary_box(summary_title, summary_items)
    
    print_urgency_summary(urgency_counts, total_order_qty)


def print_model_performance_summary(
    comparison_df: pd.DataFrame,
    best_model_name: str,
    best_metrics: pd.Series,
    width: int = 70
) -> None:
    """
    Print model performance summary showing which model performs best
    and which is used for recommendations.
    
    Args:
        comparison_df: DataFrame with model comparison results
        best_model_name: Name of the best performing model
        best_metrics: Series with metrics for the best model
        width: Width for formatting
    """
    print_section("MODEL PERFORMANCE", level=2)
    
    # Filter out baseline models for display (keep only ML models)
    ml_models = comparison_df[~comparison_df["Model"].str.startswith("baseline_")].copy()
    
    if ml_models.empty:
        print("  No ML models available for comparison.")
        return
    
    # Sort by RMSE (lower is better) and take top 3
    if "RMSE" in ml_models.columns:
        ml_models = ml_models.sort_values("RMSE").head(3)
    
    # Create display DataFrame with key metrics (excluding R²)
    display_cols = ["Model"]
    if "MAE" in ml_models.columns:
        display_cols.append("MAE")
    if "RMSE" in ml_models.columns:
        display_cols.append("RMSE")
    # R² removed from display as it can reduce trustability
    
    display_df = ml_models[display_cols].copy()
    
    # Format metrics
    for col in display_df.columns:
        if col != "Model" and display_df[col].dtype in ['float64', 'int64']:
            display_df[col] = display_df[col].apply(
                lambda x: format_number(x, 4) if pd.notna(x) else "N/A"
            )
    
    # Highlight best model
    display_df["Model"] = display_df["Model"].apply(
        lambda x: f"{x} (BEST)" if x == best_model_name else x
    )
    
    print_table(display_df, title=None)
    
    # Print clear statements
    print(f"Best Performing Model: {best_model_name}")
    print(f"Model Used for Recommendations: {best_model_name}")
    
    if "RMSE" in best_metrics:
        print(f"Model Performance (RMSE): {format_number(best_metrics['RMSE'], 4)}")
    
    print()
    print("Note: Model selection based on lowest RMSE (Root Mean Squared Error).")
    print("      Lower RMSE indicates better prediction accuracy.")
    print()


def suppress_output(func):
    """Decorator to suppress output from a function."""
    def wrapper(*args, **kwargs):
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            result = func(*args, **kwargs)
        return result
    return wrapper

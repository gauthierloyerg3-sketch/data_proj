"""
Central configuration for the demand forecasting project.

Keeps dataset-specific paths and tunable parameters in one place so the
pipeline can switch datasets with minimal changes.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class PathsConfig:
    raw_data: str = "data/raw/Retail_store_inventory_forecasting_dataset.zip"  # Updated to new dataset
    processed_data: str = "data/processed/processed_data.csv"
    stock_data: str = "data/raw/current_stock.csv"
    metrics_dir: str = "results/metrics"
    figures_dir: str = "results/figures"
    models_dir: str = "models/trained"


@dataclass(frozen=True)
class PreprocessingConfig:
    test_size: float = 0.2
    min_history_days: int = 30
    lags: Tuple[int, ...] = (1, 7, 30)
    moving_average_windows: Tuple[int, ...] = (7, 30)
    add_stock_to_processed: bool = True
    days_of_coverage: float = 10.0
    coverage_method: str = "mean"  # mean|median|p75|p90


@dataclass(frozen=True)
class TrainingConfig:
    random_state: int = 42
    n_splits: int = 5
    use_hyperparameter_tuning: bool = False  # Enable hyperparameter tuning (slower but more accurate)
    tuning_n_iter: int = 30  # Number of iterations for RandomizedSearchCV
    tuning_cv_splits: int = 3  # Cross-validation splits for tuning
    include_xgboost: bool = False  # Include XGBoost model (requires xgboost package)


@dataclass(frozen=True)
class RecommendationConfig:
    safety_stock_multiplier: float = 1.2
    target_days_of_coverage: float = 14.0  # Target days of inventory coverage


@dataclass(frozen=True)
class OutputConfig:
    """Configuration for terminal output verbosity and formatting."""
    verbose: bool = False  # Show technical details
    show_technical_details: bool = False  # Show model training, cross-validation details
    show_progress: bool = True  # Show progress indicators
    use_emoji: bool = False  # Use emoji/icons in output


@dataclass(frozen=True)
class CostConfig:
    """Economic cost parameters for evaluating prediction errors."""
    stockout_cost_per_unit: float = 10.0  # Cost of missing one sale (lost revenue + customer dissatisfaction)
    holding_cost_per_unit_per_day: float = 0.1  # Storage cost per unit per day
    unit_price: float = 5.0  # Average product price (for revenue calculations)


@dataclass(frozen=True)
class ForecastingConfig:
    """Configuration for forecasting features and targets."""
    forecast_quantity: bool = True  # Forecast units sold
    forecast_revenue: bool = True  # Forecast revenue (Price × Units Sold)
    # Key features (must be True - always included)
    use_weather: bool = True  # Weather Condition (key feature)
    use_seasonality: bool = True  # Seasonality (key feature)
    use_region: bool = True  # Region (key feature)
    # Additional features
    use_store_id: bool = True  # Store ID
    use_product_id: bool = True  # Product ID
    use_category: bool = True  # Category


@dataclass(frozen=True)
class ProjectConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    recommendation: RecommendationConfig = field(
        default_factory=RecommendationConfig
    )
    costs: CostConfig = field(default_factory=CostConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    forecasting: ForecastingConfig = field(default_factory=ForecastingConfig)


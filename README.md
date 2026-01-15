# Retail Demand Forecasting Project

A machine learning system for predicting product demand and revenue, and generating order recommendations for retail stores using Weather, Seasonality, and Region as key predictive features.

## Project Description

This project builds a machine learning system capable of predicting product demand and revenue for retail stores and generating order recommendations based on forecasted needs. Using retail store inventory forecasting data, the system models how much of each product will be sold and the associated revenue in the near future, with Weather Condition, Seasonality, and Region as key predictive features. These predictions are converted into reorder suggestions with breakdowns by weather, seasonality, and region.

The project implements multiple supervised learning models:
- **Linear Regression**: Baseline interpretable model
- **Random Forest Regressor**: Captures non-linear relationships
- **Gradient Boosting Regressor**: Sequential learning for improved accuracy
- **XGBoost Regressor**: Advanced gradient boosting (often best performance, optional)

## Project Structure

```
data_proj/
├── README.md                    # This file
├── PROPOSAL.md                  # Project proposal
├── environment.yml              # Conda environment dependencies
├── requirements.txt             # Pip dependencies
├── main.py                      # Main entry point
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── models.py               # Model definitions and training
│   └── evaluation.py           # Evaluation, visualization, recommendations
├── data/
│   ├── raw/                    # Original dataset (Retail_store_inventory_forecasting_dataset.zip)
│   └── processed/              # Preprocessed data
├── models/
│   └── trained/                # Saved trained models
└── results/
    ├── figures/                # Visualizations
    └── metrics/                # Evaluation metrics and reports
```

## Dataset

The dataset (`data/raw/Retail_store_inventory_forecasting_dataset.zip`) contains retail store inventory forecasting data with the following key features:

### Key Predictive Features (Always Included)
- **Weather Condition**: Rainy, Sunny, Cloudy, Snowy (4 conditions) - one-hot encoded
- **Seasonality**: Autumn, Summer, Winter, Spring (4 seasons) - one-hot encoded
- **Region**: North, South, West, East (4 regions) - one-hot encoded

### Additional Categorical Features
- **Store ID**: S001-S005 (5 stores) - one-hot encoded
- **Product ID**: P0001-P0010+ (multiple products) - one-hot encoded
- **Category**: Groceries, Toys, Electronics, Furniture, Clothing (5 categories) - one-hot encoded

### Numerical Features
- **Inventory Level**: Current stock levels
- **Units Ordered**: Historical order quantities
- **Price**: Product price
- **Discount**: Discount percentage
- **Competitor Pricing**: Competitor price information
- **Holiday/Promotion**: Binary indicators

**Note**: The dataset contains a "Demand Forecast" column, but it is **excluded** from features to prevent data leakage. The models create their own forecasts from scratch.

### Target Variables
- **Units Sold**: Primary target for quantity forecasting
- **Revenue**: Calculated as Price × Units Sold (secondary target for revenue forecasting)

The dataset is included in the project at `data/raw/Retail_store_inventory_forecasting_dataset.zip`.

## Setup Instructions

### Option 1: Using Conda (Recommended)

1. Create the conda environment:
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate retail_demand_forecasting
```

### Option 2: Using pip

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete workflow:

```bash
python main.py
```

This will:
1. Load and preprocess the data (with Weather, Seasonality, Region as key features)
2. Train all models for quantity forecasting (Linear Regression, Random Forest, Gradient Boosting, and optionally XGBoost)
3. Train all models for revenue forecasting (if enabled)
4. Evaluate and compare models
5. Generate predictions with breakdowns by Weather/Seasonality/Region
6. Create order recommendations
7. Save visualizations and metrics

### Configuration

Configuration (paths, split ratio, feature settings, safety stock, hyperparameter tuning) lives in `src/config.py`. Key settings:

- **Hyperparameter Tuning**: Set `use_hyperparameter_tuning=True` in `TrainingConfig` to enable tuning (slower but more accurate)
- **XGBoost**: Set `include_xgboost=True` in `TrainingConfig` to include XGBoost model (requires xgboost package)
- **Tuning Parameters**: Adjust `tuning_n_iter` (default: 30) and `tuning_cv_splits` (default: 3) to balance speed vs accuracy

Example configuration for maximum accuracy:
```python
@dataclass(frozen=True)
class TrainingConfig:
    random_state: int = 42
    n_splits: int = 5
    use_hyperparameter_tuning: bool = True  # Enable tuning
    tuning_n_iter: int = 50  # More iterations = better results (slower)
    tuning_cv_splits: int = 3
    include_xgboost: bool = True  # Include XGBoost
```

### Outputs

After running, you'll find:

**Metrics** (`results/metrics/`):
- Model comparison (`model_comparison.csv`) - MAE, RMSE for all quantity models
- Revenue model comparison (`revenue_model_comparison.csv`) - MAE, RMSE for all revenue models
- Cross-validation results (`cross_validation_comparison.csv`) - Mean±std metrics
- Best hyperparameters (`best_hyperparameters.json`) - Optimal hyperparameters found during tuning (if enabled)
- Per-product performance (`per_product_performance.csv`) - Performance by product
- Weather/Seasonality/Region impact (`weather_seasonality_region_impact.csv`) - Impact analysis
- Revenue forecast report (`revenue_forecast_report.csv`) - Revenue breakdowns by dimensions
- Model interpretation (`model_interpretation.txt`) - Analysis of model performance
- Final recommendations (`final_order_recommendations.csv`) - Best model's recommendations with urgency

**Figures** (`results/figures/`):
- Predictions vs actual (`{model_name}_predictions.png`)
- Revenue forecasts (`revenue_forecasts.png`) - Revenue predictions with Weather/Region breakdowns
- Sales trends (`sales_trends.png`, saved once)
- Residual diagnostics (`{model_name}_residuals.png`)
- Feature importance (`{model_name}_feature_importance.png`)
- Per-product performance (`per_product_performance.png`)

**Models** (`models/trained/`):
- Saved trained models (`.joblib` files) for future use

## Reproducibility

All random operations use `random_state=42` for reproducibility:
- Model training (Random Forest, Gradient Boosting)
- Data splitting (if applicable)
- NumPy random operations

## Methodology

### Time-Series Cross-Validation

The project uses time-series cross-validation (`TimeSeriesSplit`) instead of standard k-fold because:
- **Temporal order matters**: Sales data has time dependencies
- **Prevents data leakage**: Future data cannot be used to predict the past
- **More realistic**: Simulates real-world forecasting scenarios
- **Robust estimates**: Provides mean and standard deviation across multiple folds

### Urgency Calculation

Urgency levels are calculated based on:
```
days_until_stockout = current_stock / predicted_daily_demand
```

This metric helps prioritize which products need immediate attention.

### Feature Engineering

The preprocessing pipeline creates:
- **Time-based features**: Day of week, month, quarter, day of month, is_weekend
- **Key categorical features** (one-hot encoded): Weather Condition, Seasonality, Region (must be included)
- **Additional categorical features** (one-hot encoded): Store ID, Product ID, Category
- **Numerical features**: Inventory Level, Units Ordered, Price, Discount, Competitor Pricing, Holiday/Promotion
- **Lag features**: Previous day's sales (lag_1, lag_7, lag_30) grouped by Store×Product×Category×Region
- **Moving averages**: Rolling statistics (ma_7, ma_30) grouped by Store×Product×Category×Region
- **Revenue calculation**: Revenue = Price × Units Sold

**Note**: Lag and moving average features are grouped by Store×Product×Category×Region (not including Weather/Seasonality) to avoid over-fragmentation while preserving temporal patterns.

### Key Features: Weather, Seasonality, and Region

The models emphasize three key predictive features that are always included:
- **Weather Condition**: Captures how weather affects sales patterns (e.g., Rainy days may have different demand)
- **Seasonality**: Captures seasonal trends (e.g., Winter vs Summer patterns)
- **Region**: Captures regional differences (e.g., North vs South regional variations)

These features are one-hot encoded and included in all models. The system analyzes their impact on predictions and provides breakdowns by these dimensions.

### Hyperparameter Tuning

The system supports hyperparameter tuning for tree-based models (Random Forest, Gradient Boosting, XGBoost) to improve accuracy:

- **Method**: RandomizedSearchCV with time-series cross-validation
- **Default**: Tuning is disabled by default (`use_hyperparameter_tuning=False`) for faster execution
- **Enable**: Set `use_hyperparameter_tuning=True` in `TrainingConfig` to enable tuning
- **Parameters**: Adjust `tuning_n_iter` (default: 30) and `tuning_cv_splits` (default: 3) to balance speed vs accuracy
- **Results**: Best hyperparameters are saved to `results/metrics/best_hyperparameters.json`

**Expected improvements with tuning**:
- Random Forest: 5-10% error reduction
- Gradient Boosting: 10-15% error reduction
- XGBoost: 15-25% error reduction

### XGBoost Model

XGBoost (eXtreme Gradient Boosting) is an advanced gradient boosting algorithm that often achieves the best performance:

- **Included by default**: Set `include_xgboost=True` in `TrainingConfig` (default: True)
- **Requires**: XGBoost package (`pip install xgboost`)
- **Performance**: Typically outperforms Gradient Boosting by 10-20%
- **Tuning**: Supports hyperparameter tuning when `use_hyperparameter_tuning=True`

### Revenue Forecasting

In addition to quantity forecasting, the system also forecasts revenue:
- **Revenue Calculation**: Revenue = Price × Units Sold
- **Separate Models**: Dedicated models trained specifically for revenue prediction
- **Breakdowns**: Revenue forecasts include breakdowns by Weather, Seasonality, and Region
- **Impact Analysis**: Analyzes how Weather, Seasonality, and Region affect revenue predictions

### Assumptions & Limitations

- **Stationarity**: Assumes sales patterns are relatively stable over time
- **Linearity**: Linear Regression assumes linear relationships (may not capture all patterns)
- **Data size**: Limited data may favor simpler models (Linear Regression)
- **Product independence**: Treats each product independently within dimensions
- **Weather/Seasonality consistency**: Assumes Weather and Seasonality are consistent within day/region combinations

## Code Quality

- **Formatting**: Code formatted with `black` (PEP 8 compliant)
- **Naming**: Clear, descriptive variable and function names
- **Structure**: Modular code in `src/` directory
- **Comments**: Explain business logic and rationale
- **Docstrings**: Comprehensive function documentation

## Dependencies

- Python 3.9+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scipy >= 1.10.0 (for statistical tests and Q-Q plots)
- black >= 23.0.0 (for code formatting)
- joblib >= 1.3.0 (for model persistence)
- xgboost >= 2.0.0 (optional, for XGBoost model - install with `pip install xgboost`)

## Model Evaluation

Models are evaluated using:
- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model

The best model is selected based on lowest RMSE.

### Model Diagnostics

The project includes comprehensive diagnostic tools:

- **Residual Plots**: Three-panel diagnostic plots showing:
  - Residuals vs predicted values (heteroscedasticity check)
  - Residuals vs actual values (systematic error detection)
  - Q-Q plot (normality check)
  - Saved to `results/figures/{model_name}_residuals.png`

- **Feature Importance Analysis**: 
  - For Random Forest & Gradient Boosting: extracts `feature_importances_`
  - For Linear Regression: extracts absolute coefficient values
  - Visualizes top 10 most important features
  - Saved to `results/figures/{model_name}_feature_importance.png`

### Cross-Validation

Time-series cross-validation is performed using `TimeSeriesSplit` (5 folds) to:
- Respect temporal order and avoid data leakage
- Provide robust performance estimates across multiple splits
- Identify model stability and overfitting
- Results saved to `results/metrics/cross_validation_comparison.csv`

### Model Interpretation

The project includes model interpretation analysis that:
- Analyzes feature correlations
- Compares data size vs model complexity
- Identifies overfitting indicators
- Explains why certain models perform better
- Provides recommendations for model selection
- Report saved to `results/metrics/model_interpretation.txt`

### Per-Product Performance Analysis

Analyzes model performance for each product line separately:
- Calculates MAE, RMSE, and R² for each product category
- Identifies which products are predicted best/worst
- Helps target improvements for specific categories
- Results saved to `results/metrics/per_product_performance.csv`
- Visualization saved to `results/figures/per_product_performance.png`

## Order Recommendations

The system generates order recommendations based on:
- Predicted demand for each product line
- Safety stock buffer (default: 20% above predicted demand)
- Current stock levels (automatically included in processed data)

### Urgency Levels

Each recommendation includes an urgency level:
- **URGENT**: Stockout within 0-3 days (order immediately!)
- **NORMAL**: Stockout within 4-7 days (order soon)
- **LOW**: Stockout within 8-14 days (plan order)
- **POOR**: Stockout beyond 14 days or sufficient stock (monitor)

Recommendations are sorted by urgency and include:
- Days until stockout
- Current stock level
- Predicted daily demand
- Safety stock target
- Recommended order quantity

### Current Stock Configuration

The `current_stock` column is automatically added to the processed dataset during preprocessing. The system uses a data-driven approach to determine stock levels:

1. **Stock column in raw dataset**: If the raw dataset contains an inventory/stock column, it will be preserved and aggregated by product_line.

2. **Estimation from historical demand**: If no stock column exists in the raw data, stock levels are automatically estimated from historical sales patterns using:
   - Average daily demand per product
   - Configurable days of coverage (default: 10 days)
   - Formula: `estimated_stock = average_daily_demand × days_of_coverage`

3. **Static per product**: Stock is the same for all rows of the same product_line in the processed data.

Stock levels are saved in `data/processed/processed_data.csv` as a `current_stock` column, ensuring all necessary information is contained in a single processed dataset.

**Optional**: To manually specify stock levels, you can still use `data/raw/current_stock.csv`:
```csv
product_line,current_stock
Food and beverages,150
Health and beauty,75
Home and lifestyle,200
Sports and travel,50
Electronic accessories,100
Fashion accessories,120
```

The system will prioritize this file if it exists, otherwise it will auto-estimate from historical data.

Recommendations are saved to CSV files and visualized in bar charts.

## Testing

Run the lightweight checks:

```bash
pytest
```

Tests cover schema validation, chronological splitting, and safe recommendation generation (clipping negative/NaN predictions).

## Reproducibility & Finance Notes

- Seeds are set centrally in `src/config.py` (Python and NumPy) and reused across models and cross-validation.
- Chronological splits (`TimeSeriesSplit`, no shuffling) avoid look-ahead bias.
- Research question: forecast short-horizon daily demand per product line to inform reorder quantities with a safety-stock buffer.
- Assumptions: no transaction/holding costs modelled; product lines treated independently; stable data-generating process over the observed window. Adjust horizons and costs in `config.py` as needed.

## Author

Data Science Project - Retail Demand Forecasting

## License

This project is for educational purposes.

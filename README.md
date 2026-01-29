# Retail Demand Forecasting Project

A machine learning system for predicting product demand and revenue, and generating order recommendations for retail stores. Uses Weather, Seasonality, and Region as key predictive features.

---

## Does `python main.py` work?

**The #1 requirement: the code must run on a fresh environment.** Test before submitting.

### Quick start (3 steps)

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python main.py
```

**Expected:** The script runs without errors, prints model performance and order recommendations, and writes outputs to `results/metrics/`, `results/figures/`, and `models/trained/`. Exit code 0.

**Data:** The dataset is included in the repo at `data/raw/Retail_store_inventory_forecasting_dataset.zip`. No download required.

---

## Reproducibility

- **Random seed:** `random_state=42` in `src/config.py` (data split, model training, NumPy).
- **Chronological split:** `TimeSeriesSplit`; no shuffling. Same train/test split across runs.
- **Pinned versions:** `requirements.txt` uses pinned package versions. Use a fresh venv and `pip install -r requirements.txt` to reproduce the same environment.

You should get the same metrics and outputs when you run `python main.py` on a clean setup.

---

## Prerequisites

- **Python 3.9+**
- Dependencies in `requirements.txt` (see below).

---

## Setup

### Option 1: venv + pip (recommended for grading)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Option 2: Conda

```bash
conda env create -f environment.yml
conda activate retail_demand_forecasting
pip install -r requirements.txt   # use pinned versions
python main.py
```

**Note:** `environment.yml` does not pin package versions. For full reproducibility, install from `requirements.txt` inside the conda environment.

---

## What `python main.py` does

1. Loads and preprocesses data from `data/raw/Retail_store_inventory_forecasting_dataset.zip`
2. Trains models: Linear Regression, Random Forest, Gradient Boosting, plus baselines (last value, mean, seasonal naïve)
3. Evaluates models (MAE, RMSE, cost proxy) and picks the best by RMSE
4. Trains revenue-forecast models (if enabled in config)
5. Generates order recommendations with urgency levels
6. Saves metrics, figures, and trained models

### Outputs

| Location | Contents |
|----------|----------|
| `results/metrics/` | `model_comparison.csv`, `revenue_model_comparison.csv`, `final_order_recommendations.csv`, etc. |
| `results/figures/` | Predictions, residuals, feature importance, per-product performance, sales trends |
| `models/trained/` | Trained models (`.joblib`) |
| `data/processed/` | `processed_data.csv` |

---

## Project structure

```
data_proj/
├── README.md
├── PROPOSAL.md
├── requirements.txt      # Pinned deps for reproducibility
├── environment.yml       # Conda (optional)
├── pyproject.toml        # pytest pythonpath, project info
├── main.py               # Entry point: run this
├── src/
│   ├── config.py         # Paths, training config, random_state
│   ├── data_loader.py
│   ├── models.py
│   ├── evaluation.py
│   ├── output_config.py
│   └── output_formatter.py
├── data/
│   ├── raw/              # Retail_store_inventory_forecasting_dataset.zip (included)
│   └── processed/
├── models/trained/
└── results/
    ├── figures/
    └── metrics/
```

---

## Configuration

Key settings in `src/config.py`:

- **Paths:** `PathsConfig` — raw data, processed data, metrics, figures, models.
- **Training:** `TrainingConfig` — `random_state=42`, `test_size=0.2`, `n_splits=5`, etc.
- **XGBoost:** `include_xgboost=False` by default. Set `True` to add XGBoost (requires `pip install xgboost`).
- **Hyperparameter tuning:** `use_hyperparameter_tuning=False` by default. Set `True` for tuning (slower).

---

## Dependencies (`requirements.txt`)

| Package | Version |
|---------|---------|
| pandas | 2.2.0 |
| numpy | 1.26.4 |
| scikit-learn | 1.4.2 |
| scipy | 1.11.4 |
| matplotlib | 3.8.2 |
| seaborn | 0.13.2 |
| joblib | 1.3.2 |
| pytest | 8.0.2 |
| black | 24.2.0 |

**Optional:** XGBoost (`pip install xgboost`) if you enable `include_xgboost` in `config.py`.

---

## Testing

From the project root:

```bash
pytest
```

Uses `pyproject.toml` (pythonpath) so that `src` is importable. Covers schema validation, chronological splitting, and recommendation generation (e.g. clipping negative/NaN predictions).

---

## Dataset

- **Source:** `data/raw/Retail_store_inventory_forecasting_dataset.zip` (included).
- **Features:** Weather Condition, Seasonality, Region (key), plus Store ID, Product ID, Category, Price, Inventory Level, Units Ordered, Discount, Competitor Pricing, etc.
- **Targets:** Units sold (primary), Revenue = Price × Units sold (secondary).
- **Preprocessing:** Lag and moving-average features, time-based features, one-hot encoding. `Demand Forecast` is excluded to avoid leakage.

---

## Methodology (short)

- **Models:** Linear Regression, Random Forest, Gradient Boosting; baselines: last value, mean, seasonal naïve.
- **Validation:** Time-series cross-validation (`TimeSeriesSplit`), chronological train/test split.
- **Metrics:** MAE, RMSE; economic cost proxy (holding vs. stockout) for model comparison.
- **Forecast:** Day-*t* demand predicted from features available up to day *t*−1.

See the project report (`report/project_report.tex` / PDF) for full methodology, research question, and results.

---

## Order recommendations

Recommendations use predicted demand, safety-stock buffer, and current stock. Urgency is based on days until stockout:

- **URGENT:** 0–3 days  
- **NORMAL:** 4–7 days  
- **LOW:** 8–14 days  
- **POOR:** >14 days or sufficient stock  

---

## License

Educational use.

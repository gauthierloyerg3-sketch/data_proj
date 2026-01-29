# Master's-Level Project Review: Retail Demand Forecasting

**Reviewer**: Senior Data Science & Finance Mentor  
**Date**: 2024  
**Review Type**: Production-Grade Assessment for Master's in Finance  
**Project**: Retail Store Inventory Forecasting ML System

---

## A. EXECUTIVE SUMMARY

### What Works ✅
- **Time-series methodology**: Proper use of `TimeSeriesSplit` for cross-validation, chronological train/test split
- **Data leakage prevention**: Explicit exclusion of "Demand Forecast" column, scaler fitted only on training data
- **Code structure**: Clean modular design with separation of concerns (data_loader, models, evaluation)
- **Configuration management**: Centralized config via dataclasses, easy to adapt to new datasets
- **Feature engineering**: Comprehensive pipeline with time-based, lag, and moving average features
- **Data quality**: Recent addition of data cleaning and validation functions

### What is Broken ❌
- **Seasonal naive baseline**: Uses last value instead of day-of-week matching (line 214 in `src/models.py`)
- **Missing MAPE metric**: Only MAE/RMSE reported, no scale-independent metric for cross-product comparison
- **Potential data leakage**: "Inventory Level" and "Units Ordered" included without temporal validation
- **Limited test coverage**: Only 6 test functions, missing critical path tests

### What is Risky ⚠️
- **Feature semantics undocumented**: Temporal availability of features not documented (when is Inventory Level measured?)
- **Error handling gaps**: Many functions lack comprehensive error handling
- **Dependency versioning**: `requirements.txt` uses `>=` instead of pinned versions (reproducibility risk)
- **Production readiness**: No model versioning, monitoring, or deployment infrastructure

### Top 10 Most Severe Issues (Ranked)

1. **CRITICAL**: Seasonal naive baseline broken (invalid baseline comparison)
2. **CRITICAL**: Missing MAPE metric (cannot assess performance across products)
3. **CRITICAL**: Potential data leakage from "Inventory Level" and "Units Ordered" (unvalidated temporal semantics)
4. **MAJOR**: Limited test coverage (only 6 tests, missing critical paths)
5. **MAJOR**: Feature semantics undocumented (risk of misuse in production)
6. **MAJOR**: Dependency versions not pinned (reproducibility risk)
7. **MAJOR**: Error handling gaps (production robustness concerns)
8. **MINOR**: Empty `scripts/` directory (dead code)
9. **MINOR**: Missing comprehensive model diagnostics (error analysis by product/time)
10. **MINOR**: No production deployment considerations (versioning, monitoring, API)

### Quick Wins vs Deeper Refactors

**Quick Wins** (1-2 hours each):
- Fix seasonal naive baseline
- Add MAPE metric
- Pin dependency versions
- Document feature semantics
- Remove empty `scripts/` directory

**Deeper Refactors** (4-8 hours each):
- Add comprehensive test coverage
- Improve error handling throughout
- Validate and document data leakage risks
- Add model diagnostics and error analysis
- Production deployment infrastructure

---

## B. REPO MAP

### Project Purpose
This project builds a machine learning system for retail demand forecasting. It predicts daily product demand (units sold) and revenue using weather, seasonality, region, and historical sales patterns. Predictions are converted into order recommendations with urgency levels based on current stock and predicted demand.

### File Tree Overview
```
data_proj/
├── main.py                      # Entry point: orchestrates entire pipeline
├── src/                         # Core source code
│   ├── config.py               # Centralized configuration (paths, hyperparameters)
│   ├── data_loader.py          # Data loading, cleaning, feature engineering
│   ├── models.py               # Model definitions, training, cross-validation
│   ├── evaluation.py           # Metrics, visualizations, recommendations
│   ├── output_config.py        # Output file management
│   └── output_formatter.py     # Terminal output formatting
├── data/
│   ├── raw/                    # Original datasets (ZIP, CSV)
│   └── processed/              # Preprocessed data (CSV cache)
├── models/trained/             # Saved trained models (.joblib)
├── results/
│   ├── figures/                # Visualizations (PNG)
│   └── metrics/                # Evaluation metrics (CSV, JSON, TXT)
├── tests/                       # Test suite (minimal: 6 tests)
├── requirements.txt             # Python dependencies (unpinned versions)
├── environment.yml              # Conda environment (unpinned versions)
└── README.md                    # Project documentation
```

### Entry Points
- **Primary**: `python main.py` - Runs complete pipeline (data loading → training → evaluation → recommendations)
- **Testing**: `pytest` - Runs test suite (6 tests covering basic validation)

### Data Flow
```
Raw Data (ZIP/CSV)
  ↓ [load_raw_data]
Raw DataFrame
  ↓ [clean_and_validate_data]
Cleaned DataFrame
  ↓ [parse_datetime, aggregate_sales_by_all_dimensions]
Aggregated DataFrame
  ↓ [create_time_features, create_lag_features, create_moving_averages, create_categorical_features, create_numerical_features]
Feature-Engineered DataFrame
  ↓ [time_aware_train_test_split]
Train/Test Split
  ↓ [prepare_features_and_target]
X_train, y_train, X_test, y_test
  ↓ [train_all_models]
Trained Models + Scalers
  ↓ [compare_models, evaluate_and_visualize]
Metrics + Visualizations
  ↓ [create_order_recommendations]
Order Recommendations (CSV)
```

---

## C. QUALITY AUDIT CHECKLIST

### 1. Correctness (Bugs, Edge Cases, Determinism)
**Status**: ⚠️ **PARTIAL PASS**

**Evidence**:
- ✅ Random seeds set (line 31-33 in `main.py`, line 39-43 in `models.py`)
- ✅ Time-aware splitting prevents look-ahead bias (line 754-782 in `data_loader.py`)
- ❌ Seasonal naive baseline broken (line 214 in `models.py` - uses last value, not day-of-week)
- ❌ No edge case handling for empty data, all zeros, missing columns in many functions
- ⚠️ `clean_features()` clips all columns including datetime (line 68 in `models.py` - fixed in recent version)

**Files**: `src/models.py:214`, `src/data_loader.py:754-782`, `src/models.py:46-70`

### 2. Reproducibility (Requirements, Lockfiles, Seeds, Env Setup)
**Status**: ❌ **FAIL**

**Evidence**:
- ✅ Random seeds set consistently
- ✅ `environment.yml` and `requirements.txt` present
- ❌ Dependency versions not pinned (`>=` instead of `==` in `requirements.txt`)
- ❌ No `requirements-lock.txt` or `Pipfile.lock`
- ⚠️ Conda environment uses `>=` versions (line 7-14 in `environment.yml`)

**Files**: `requirements.txt:1-9`, `environment.yml:7-14`

### 3. Data Pipeline (Schema Handling, Missing Values, Type Safety, Leakage Checks)
**Status**: ⚠️ **PARTIAL PASS**

**Evidence**:
- ✅ Schema validation (`validate_required_columns` in `data_loader.py:34-46`)
- ✅ Missing value handling (fillna in multiple places)
- ✅ "Demand Forecast" explicitly excluded (line 251-254 in `models.py`)
- ❌ "Inventory Level" and "Units Ordered" included without temporal validation (line 544-545 in `data_loader.py`)
- ⚠️ No validation that features are available at prediction time

**Files**: `src/data_loader.py:544-545`, `src/models.py:251-254`

### 4. Modeling (Baseline, CV, Metrics, Calibration, Stability)
**Status**: ⚠️ **PARTIAL PASS**

**Evidence**:
- ✅ Time-series cross-validation (`TimeSeriesSplit` in `models.py`)
- ✅ Multiple baselines implemented (last_value, mean, seasonal_naive)
- ❌ Seasonal naive baseline broken (line 214 in `models.py`)
- ❌ Missing MAPE metric (only MAE/RMSE in `evaluation.py:24-40`)
- ✅ Economic cost calculation (holding cost, stockout cost)
- ⚠️ No calibration checks (prediction intervals, uncertainty quantification)

**Files**: `src/models.py:196-223`, `src/evaluation.py:24-40`

### 5. Finance-Specific Rigor (Time Series Splits, Look-Ahead Bias, Survivorship Bias, Stationarity, Transaction Costs, Interpretability)
**Status**: ✅ **PASS**

**Evidence**:
- ✅ Time-aware train/test split (chronological, no shuffling)
- ✅ Time-series cross-validation (prevents look-ahead bias)
- ✅ Economic cost calculation (transaction costs: holding cost, stockout cost)
- ✅ Feature importance analysis (for tree models)
- ⚠️ No stationarity tests (assumes stable patterns)
- ⚠️ No survivorship bias checks (all products included)

**Files**: `src/data_loader.py:754-782`, `src/evaluation.py:43-83`

### 6. Testing (Unit/Integration, Coverage, Deterministic Tests)
**Status**: ❌ **FAIL**

**Evidence**:
- ✅ 6 test functions in `tests/test_validation_and_recommendations.py`
- ❌ No tests for feature engineering (lag features, moving averages)
- ❌ No tests for data leakage detection
- ❌ No tests for edge cases (empty data, all zeros, missing columns)
- ❌ No integration tests (end-to-end pipeline)
- ❌ No coverage report (likely <20% coverage)

**Files**: `tests/test_validation_and_recommendations.py`

### 7. Performance (Complexity, Vectorization, Caching, IO)
**Status**: ✅ **PASS**

**Evidence**:
- ✅ Vectorized operations (pandas, numpy)
- ✅ Data caching (processed_data.csv)
- ✅ Efficient feature engineering (groupby operations)
- ✅ No obvious performance bottlenecks

**Files**: `src/data_loader.py:632-751`

### 8. Security & Secrets (API Keys, Tokens, PII)
**Status**: ✅ **PASS**

**Evidence**:
- ✅ No API keys or tokens in code
- ✅ No PII handling (retail data, no customer info)
- ✅ No secrets in repository

**Files**: N/A (no security concerns)

### 9. Engineering Hygiene (Lint, Formatting, Typing, Logs, Exceptions)
**Status**: ⚠️ **PARTIAL PASS**

**Evidence**:
- ✅ Code formatted (black mentioned in README)
- ✅ Type hints in function signatures (partial)
- ❌ Minimal error handling (many functions lack try-except)
- ❌ No logging (uses print statements)
- ⚠️ Inconsistent docstring quality

**Files**: Throughout codebase

### 10. Documentation (README, Usage, Assumptions, Methodology)
**Status**: ✅ **PASS**

**Evidence**:
- ✅ Comprehensive README with setup, usage, methodology
- ✅ Function docstrings (most functions)
- ⚠️ Feature semantics not documented (temporal availability)
- ⚠️ Assumptions documented but could be more explicit

**Files**: `README.md`, function docstrings

### 11. Project Structure (Modules, Separation of Concerns, No Duplicate Logic)
**Status**: ✅ **PASS**

**Evidence**:
- ✅ Clear module separation (data_loader, models, evaluation)
- ✅ No duplicate logic
- ✅ Configuration centralized
- ⚠️ Empty `scripts/` directory (dead code)

**Files**: `src/`, `scripts/`

### 12. Dependency Hygiene (Unused Deps, Version Pins)
**Status**: ❌ **FAIL**

**Evidence**:
- ❌ Versions not pinned (`>=` instead of `==`)
- ⚠️ XGBoost optional but listed in requirements.txt (line 9)
- ✅ No obviously unused dependencies

**Files**: `requirements.txt:1-9`

---

## D. ISSUE LIST (EXHAUSTIVE)

| Severity | Category | File:line(s) | Symptom | Root Cause | Fix | Acceptance Criteria |
|----------|----------|--------------|---------|------------|-----|-------------------|
| Blocker | Baseline | `src/models.py:214` | Seasonal naive uses last value instead of day-of-week | Implementation shortcut, comment admits it | Implement proper day-of-week matching | Seasonal naive predictions match historical value from 7 days ago, same day of week |
| Critical | Metrics | `src/evaluation.py:24-40` | No MAPE metric reported | MAPE not implemented | Add MAPE calculation with zero-division handling | MAPE included in all comparison tables, handles zero actuals |
| Critical | Data Leakage | `src/data_loader.py:544-545` | Inventory Level/Units Ordered included without validation | No temporal semantics documented/validated | Document temporal semantics, add validation | Features documented, validation checks temporal availability |
| Major | Testing | `tests/test_validation_and_recommendations.py` | Only 6 tests, missing critical paths | Insufficient test coverage | Add tests for feature engineering, leakage, edge cases | >80% code coverage, all critical paths tested |
| Major | Documentation | `src/data_loader.py:525-562` | Feature semantics not documented | Missing docstrings for temporal semantics | Add comprehensive feature documentation | All features documented with temporal availability |
| Major | Reproducibility | `requirements.txt:1-9` | Versions not pinned | Uses `>=` instead of `==` | Pin all dependency versions | All versions pinned, reproducible builds |
| Major | Error Handling | Throughout codebase | Minimal error handling | Missing try-except blocks | Add comprehensive error handling | All file I/O wrapped, informative error messages |
| Minor | Dead Code | `scripts/` | Empty directory | Unused directory | Remove or add scripts | Directory removed or populated |
| Minor | Diagnostics | `src/evaluation.py` | No systematic error analysis | Missing diagnostic functions | Add error analysis by product/time | Diagnostic reports generated |
| Minor | Production | Entire codebase | No deployment infrastructure | Not designed for production | Add versioning, monitoring, API | Production deployment guide created |

---

## E. FIX PLAN (COMMIT-BY-COMMIT)

### Commit 1: Fix Seasonal Naive Baseline
**Files**: `src/models.py`
**Changes**: Implement proper day-of-week matching in `SeasonalNaiveBaseline.predict()`
**Why**: Invalid baseline comparison invalidates model evaluation
**Validation**: Run `pytest tests/test_validation_and_recommendations.py::test_naive_baselines`, verify seasonal naive uses 7-day-ago value

### Commit 2: Add MAPE Metric
**Files**: `src/evaluation.py`, `src/models.py` (if needed)
**Changes**: Add `mean_absolute_percentage_error` calculation, include in all comparison tables
**Why**: Essential for cross-product performance comparison
**Validation**: Verify MAPE appears in `model_comparison.csv`, handles zero actuals gracefully

### Commit 3: Document and Validate Feature Semantics
**Files**: `src/data_loader.py`, `src/config.py`
**Changes**: Add docstrings documenting temporal semantics of Inventory Level and Units Ordered, add validation checks
**Why**: Prevents data leakage and misuse in production
**Validation**: Documentation clear, validation checks pass/fail appropriately

### Commit 4: Pin Dependency Versions
**Files**: `requirements.txt`, `environment.yml`
**Changes**: Replace `>=` with `==` for all dependencies, add version numbers
**Why**: Ensures reproducible builds
**Validation**: `pip install -r requirements.txt` produces identical environment

### Commit 5: Improve Error Handling
**Files**: `src/data_loader.py`, `src/models.py`, `src/evaluation.py`
**Changes**: Add try-except blocks with informative error messages for file I/O, data validation, model training
**Why**: Production robustness
**Validation**: Test with corrupted files, missing columns, edge cases - all handled gracefully

### Commit 6: Add Comprehensive Tests
**Files**: `tests/test_validation_and_recommendations.py`, new test files
**Changes**: Add tests for feature engineering, data leakage, edge cases, integration tests
**Why**: Enables safe refactoring and catches bugs
**Validation**: `pytest --cov=src` shows >80% coverage

### Commit 7: Cleanup Dead Code
**Files**: `scripts/` directory
**Changes**: Remove empty directory or add placeholder README
**Why**: Clean codebase
**Validation**: No empty directories, all files have purpose

### Commit 8: Add Model Diagnostics
**Files**: `src/evaluation.py`
**Changes**: Add functions to analyze errors by product, time period, feature importance
**Why**: Enables model improvement
**Validation**: Diagnostic reports generated and saved

---

## F. APPLY FIXES

*[Fixes will be applied in subsequent steps]*

---

## G. FINAL REPORT

*[Will be generated after fixes are applied]*

---

**Review Complete - Ready for Fixes**

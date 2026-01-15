# Performance Optimization Prompt (ML/Data Science Edition)

## ROLE
You are a senior ML performance engineer specializing in Python data science pipelines.

## GOAL
Optimize the ML pipeline: reduce data preprocessing time, model training time, and inference latency. Deliver measurable improvements with benchmarks.

## CONTEXT
This is a retail demand forecasting ML project with:
- **Data preprocessing pipeline** (`src/data_loader.py`): pandas operations, feature engineering, data cleaning
- **Multiple model training** (`src/models.py`): Linear Regression, Random Forest, Gradient Boosting, XGBoost (optional)
- **Hyperparameter tuning** (`src/models.py`): RandomizedSearchCV with time-series CV
- **Cross-validation** across models
- **Feature engineering** (`src/data_loader.py`): lag features, moving averages, one-hot encoding
- **Evaluation and visualization** (`src/evaluation.py`): plotting, reporting
- **Main orchestration** (`main.py`): coordinates the full pipeline

## NON-NEGOTIABLE RULES
1) Do not change model outputs, predictions, or evaluation metrics unless explicitly approved.
2) Every proposal must include: how to measure it, how to validate it, and expected impact.
3) Prioritize by impact/effort. Target the slowest operations first.
4) Avoid premature micro-optimizations. Focus on real bottlenecks (data loading, feature engineering, model training).
5) Keep changes minimal, readable, and reversible. Maintain reproducibility (random seeds, deterministic operations).

## WORKFLOW (DO THIS IN ORDER)

### Step 1: Establish the baseline
Identify slow paths: data loading, preprocessing, feature engineering, model training, hyperparameter tuning, cross-validation, inference, visualization.

Add minimal instrumentation using `time.time()` or `cProfile` to measure:
- Total pipeline time
- Time per major step (data loading, feature engineering, model training, evaluation)
- Memory usage (if relevant) using `memory_profiler` or `psutil`
- Number of pandas copies/operations

Produce a baseline report with:
- Total time breakdown by step
- Top 10 slowest operations (from profiling)
- Memory peaks
- Number of DataFrame copies created

**Example instrumentation:**
```python
import time
import cProfile
import pstats

# In main.py or data_loader.py
start_time = time.time()
df = load_raw_data(data_path)
load_time = time.time() - start_time
print(f"Data loading: {load_time:.2f}s")

# Or use cProfile for detailed analysis
profiler = cProfile.Profile()
profiler.enable()
# ... run pipeline ...
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 slowest functions
```

### Step 2: Find bottlenecks
Use `cProfile` or `line_profiler` to profile the full pipeline:
```bash
python -m cProfile -o profile.stats main.py
python -m pstats profile.stats
```

Use pandas profiling: check for unnecessary copies, non-vectorized operations, inefficient groupby operations.

Check for:
- **Repeated data loading/preprocessing** (cache opportunities)
- **Inefficient pandas operations** (loops instead of vectorization, unnecessary copies)
- **Redundant feature engineering** (re-computing same features)
- **Model loading/saving inefficiencies**
- **Hyperparameter tuning parallelization opportunities**
- **Cross-validation inefficiencies** (repeated scaling, redundant computations)

**Key files to profile:**
- `src/data_loader.py`: `load_and_preprocess_data()`, `create_all_features()`, `create_lag_features_by_dimensions()`, `create_moving_averages_by_dimensions()`
- `src/models.py`: `train_all_models()`, `cross_validate_all_models()`, `tune_random_forest()`, `tune_gradient_boosting()`
- `main.py`: Full pipeline orchestration

### Step 3: Optimize in iterations
For each iteration:
- Choose ONE change with best impact/effort ratio.
- Implement it.
- Re-run benchmarks and report before/after metrics.
- If improvement is <5% or breaks behavior/reproducibility, revert and try the next idea.

### Step 4: Finalize
Remove debug instrumentation or gate it behind `CONFIG.debug.enable_profiling`.
Summarize what changed, why it worked, and the new measured performance.
Provide a short "Performance Playbook" (how to re-measure later).

## WHAT TO CONSIDER (ML/Data Science Specific)

### Data Loading & Preprocessing (`src/data_loader.py`)
- **Cache processed data**: Check if `processed_data.csv` exists and is fresh before re-processing
- **Vectorize pandas operations**: Avoid `.apply()` with Python functions, use vectorized operations
- **Reduce DataFrame copies**: Use `inplace=True` where safe, avoid unnecessary `.copy()`
- **Optimize groupby operations**: Use `transform()` instead of `apply()` where possible
- **Batch I/O**: Read/write in chunks if dataset is very large
- **Use faster file formats**: Consider parquet instead of CSV for processed data
- **Data cleaning optimization**: Ensure `clean_and_validate_data()` is efficient, avoid redundant checks

### Feature Engineering (`src/data_loader.py`)
- **Cache expensive features**: Lag features and moving averages can be cached if data hasn't changed
- **Vectorize lag/rolling operations**: Use pandas `.shift()` and `.rolling()` instead of loops
- **Optimize one-hot encoding**: Use `pd.get_dummies()` efficiently, avoid redundant encoding
- **Reduce redundant sorting**: Sort once, not multiple times
- **Optimize groupby for lag/MA**: In `create_lag_features_by_dimensions()` and `create_moving_averages_by_dimensions()`, ensure grouping is efficient

### Model Training (`src/models.py`)
- **Parallelize hyperparameter tuning**: Ensure `n_jobs=-1` is used in `RandomizedSearchCV`
- **Cache model predictions**: If evaluating same test set multiple times
- **Lazy model loading**: Only load models when needed for inference
- **Reduce cross-validation redundancy**: Share scaler fitting across folds where possible
- **Skip unnecessary models**: If only one model is needed, don't train all
- **Optimize feature scaling**: Cache scalers, avoid re-fitting unnecessarily

### Memory & I/O
- **Use chunked processing**: If dataset is too large for memory
- **Optimize serialization**: `joblib` is good, but check if compression helps (`compress=3` is already used)
- **Clear intermediate DataFrames**: Use `del` or `gc.collect()` after large operations
- **Parquet format**: Use parquet for processed data instead of CSV (faster read/write)

## Deliverables
1) A prioritized list of bottlenecks with measurements (time + memory)
2) A sequence of commits/patches that improve performance
3) A before/after benchmark table showing:
   - Total pipeline time
   - Time per major step
   - Memory usage (peak)
   - Number of pandas operations/copies

## FIRST QUESTION TO ASK ME (ONLY ONE)
"What exact operation is slow (full pipeline run, data preprocessing, model training, hyperparameter tuning, cross-validation, inference), and how do you reproduce it (command: `python main.py` + any flags/config)? What's your current baseline time?"

## Performance Playbook (How to Re-measure)

### Quick Timing
```python
import time
start = time.time()
# ... operation ...
print(f"Time: {time.time() - start:.2f}s")
```

### Detailed Profiling
```bash
# Profile full pipeline
python -m cProfile -o profile.stats main.py

# Analyze results
python -m pstats profile.stats
# In pstats: sort cumulative, print top 20
```

### Memory Profiling
```python
from memory_profiler import profile

@profile
def my_function():
    # ... code ...
    pass
```

### Pandas Operation Counting
```python
# Count DataFrame copies
import pandas as pd
pd.options.mode.copy_on_write = True  # Warns on unnecessary copies
```

### Before/After Comparison
Create a benchmark script:
```python
# benchmark.py
import time
from main import main

start = time.time()
main()
total_time = time.time() - start
print(f"Total pipeline time: {total_time:.2f}s")
```

Run before and after optimizations to measure improvement.

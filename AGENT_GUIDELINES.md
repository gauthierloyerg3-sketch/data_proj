# Agent Guidelines - Retail Demand Forecasting Project

## ⚠️ CRITICAL: READ BEFORE MAKING ANY CHANGES

This document outlines **mandatory guidelines** that must be followed when modifying this project. These guidelines ensure code quality, reproducibility, and adherence to academic requirements.

---

## 1. TEACHER'S MANDATORY REQUIREMENTS

### ✅ **CRITICAL REQUIREMENTS - MUST ALWAYS BE SATISFIED**

#### 1.1. Execution Requirement
- **`python main.py` MUST run without errors** - This is non-negotiable
- Test execution after ANY code modification
- Fix any errors immediately, do not leave broken code

#### 1.2. Dependencies Management
- **ALL dependencies MUST be listed** in both:
  - `requirements.txt` (for pip users)
  - `environment.yml` (for conda users)
- Include version numbers for reproducibility
- Add new dependencies to BOTH files when introducing new packages
- Example: If using a new library, add it to both files

#### 1.3. Data Location
- **Data MUST be local** to the project
- Data files in `data/raw/` directory
- **NEVER** reference external paths (iCloud, network drives, etc.)
- Dataset must be included in project or have clear download instructions

#### 1.4. Reproducibility
- **ALWAYS use `random_state=42`** for ALL random operations:
  - Model training (RandomForestRegressor, GradientBoostingRegressor)
  - NumPy random operations: `np.random.seed(42)`
  - Any sklearn functions with random_state parameter
  - Data splitting (if applicable)
- Use the constant `RANDOM_STATE = 42` defined in modules
- **NEVER** use random operations without setting random_state

---

## 2. CODE QUALITY STANDARDS

### 2.1. Code Formatting
- **Use `black` formatter** - Code must be PEP 8 compliant
- Format code before committing: `black src/ main.py`
- Maximum line length: 88 characters (black default)
- Use 4 spaces for indentation (no tabs)

### 2.2. Naming Conventions
- **Function names**: `snake_case`, descriptive (e.g., `calculate_reorder_urgency`)
- **Variable names**: `snake_case`, clear and meaningful (e.g., `current_stock_level`)
- **Class names**: `PascalCase` (if needed)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `RANDOM_STATE`)
- **NO abbreviations** unless universally understood (e.g., `df` for DataFrame is acceptable)

### 2.3. Code Structure
- **Modular code** - All code in `src/` directory
- **Single Responsibility** - Each function does one thing well
- **DRY Principle** - Don't Repeat Yourself, refactor common code
- **Clear imports** - Group imports: stdlib, third-party, local

### 2.4. Comments and Documentation
- **Comments explain WHY, not WHAT** - Code should be self-explanatory
- Use comments for business logic, complex algorithms, non-obvious decisions
- **All functions MUST have docstrings** with:
  - Brief description
  - Args section (parameters and types)
  - Returns section (return value and type)
  - Optional: Examples or notes
- Format: Google-style or NumPy-style docstrings

Example:
```python
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
```

---

## 3. PROJECT-SPECIFIC GUIDELINES

### 3.1. Output File Management
- **ALWAYS use `OutputConfig`** from `src/output_config.py` for controlling file generation
- **NEVER hard-code file saving** - Use configuration-driven approach
- Default behavior: Only save essential files, prevent redundancy
- When adding new outputs:
  1. Check if it should be configurable via OutputConfig
  2. Add cleanup logic if temporary/redundant
  3. Document new outputs in README.md

### 3.2. Data Preprocessing
- Handle missing values gracefully with clear warnings
- Validate data types and provide informative error messages
- Use consistent column naming (snake_case)

### 3.3. Model Training
- Always use `RANDOM_STATE = 42` for models with randomness
- Save models to `models/trained/` directory
- Use descriptive model names (snake_case)
- Log training progress for user feedback

### 3.4. Evaluation and Metrics
- Calculate all metrics consistently (MAE, RMSE, R²)
- Round numerical values appropriately (2-4 decimal places)
- Save metrics in consistent format
- Provide clear console output with metrics

---

## 4. FILE GENERATION RULES

### 4.1. When Adding New Output Files

**MUST DO:**
1. Check `OutputConfig` - Can it be controlled via configuration?
2. Add cleanup logic if file can become redundant
3. Use consistent naming: `{model_name}_{type}.{ext}` or `{type}.{ext}`
4. Save to appropriate directory:
   - Metrics → `results/metrics/`
   - Figures → `results/figures/`
   - Models → `models/trained/`
5. Document new outputs in README.md

**MUST NOT DO:**
- Create duplicate files with same information
- Hard-code file paths (except default config path)
- Save files without checking OutputConfig
- Leave temporary/debug files in output directories

### 4.2. Default Output Configuration

The project uses minimal file generation by default:
- ❌ Individual model `.txt` files (info in comparison CSV)
- ❌ Individual recommendation CSV files (use final report)
- ❌ Recommendations plots (info in final report)
- ✅ Essential comparison files (model_comparison.csv, etc.)
- ✅ Diagnostic plots for comparison (predictions, residuals, feature importance)
- ✅ Final recommendations report

**When in doubt**: Use the default configuration, it prevents redundancy.

---

## 5. ERROR HANDLING

### 5.1. Required Practices
- **ALWAYS handle FileNotFoundError** for data files with clear messages
- **Validate inputs** before processing (check for None, empty, wrong types)
- **Provide informative error messages** that help users fix issues
- **Log warnings** for non-critical issues (use `warnings.warn()`)

### 5.2. User-Friendly Error Messages
```python
# ✅ GOOD
if not data_path.exists():
    raise FileNotFoundError(
        f"Data file not found: {data_path}\n"
        "Please ensure the dataset is in data/raw/supermarket_sales.csv"
    )

# ❌ BAD
if not data_path.exists():
    raise FileNotFoundError(f"File not found: {data_path}")
```

### 5.3. Graceful Degradation
- Provide sensible defaults when optional files are missing
- Warn users about missing optional configuration
- Continue processing when possible (don't fail on non-critical issues)

---

## 6. TESTING REQUIREMENTS

### 6.1. Before Completing Any Change
**MUST verify:**
1. ✅ `python main.py` runs without errors
2. ✅ All expected outputs are generated
3. ✅ No redundant files created
4. ✅ Code follows formatting standards (black)
5. ✅ Documentation updated if behavior changed

### 6.2. Testing Checklist
```bash
# 1. Format code
black src/ main.py

# 2. Check syntax
python3 -m py_compile src/*.py main.py

# 3. Run full workflow
python3 main.py

# 4. Verify outputs exist
ls results/metrics/
ls results/figures/
```

---

## 7. CODE MODIFICATION RULES

### 7.1. When Modifying Existing Code
- **Maintain backward compatibility** when possible
- **Don't break existing functionality** - Test before and after
- **Update documentation** if behavior changes
- **Update type hints** if function signatures change
- **Keep function signatures stable** - Use optional parameters for new features

### 7.2. When Adding New Features
- **Follow existing patterns** - Maintain consistency
- **Add to appropriate module** - Don't create new files unnecessarily
- **Use OutputConfig** for any new file generation
- **Update README.md** with new features
- **Test thoroughly** - Ensure it works with existing code

### 7.3. When Fixing Bugs
- **Identify root cause** - Don't just patch symptoms
- **Test the fix** - Verify it resolves the issue
- **Check for similar issues** - Fix pattern, not just one instance
- **Update tests/documentation** if behavior changed

---

## 8. CONFIGURATION AND HARD-CODING

### 8.1. File Paths and Defaults
- **Avoid hard-coding file paths** - Use default parameters in functions
- **Use pathlib.Path** for file operations (safer than strings)
- Default file paths in function signatures are acceptable:
  ```python
  def load_data(data_path: str = "data/raw/supermarket_sales.csv"):
      # This is acceptable
  ```

### 8.2. Constants
- **ALLOWED constants:**
  - ✅ `RANDOM_STATE = 42` (reproducibility requirement)
  - ✅ Default file paths in function signatures (as defaults)
  - ✅ Mathematical constants (e.g., safety stock multiplier defaults)
  - ✅ Business logic defaults (urgency thresholds)

### 8.3. Configuration Pattern
- Use configuration files or dataclasses for settings when appropriate
- Make configurations easy to modify
- Document all configuration options
- Provide sensible defaults

---

## 9. IMPORT ORGANIZATION

### 9.1. Standard Import Order
```python
# 1. Standard library imports
import os
from pathlib import Path
from typing import Dict, List, Tuple

# 2. Third-party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# 3. Local application imports
from src.data_loader import load_raw_data
from src.output_config import OutputConfig
```

### 9.2. Import Rules
- **Group imports** by type (stdlib, third-party, local)
- **One import per line** for clarity
- **Sort imports alphabetically** within each group
- **Use specific imports** when possible (not `from module import *`)

---

## 10. DOCUMENTATION STANDARDS

### 10.1. README.md Updates
**MUST update README.md when:**
- Adding new features or outputs
- Changing configuration options
- Modifying workflow steps
- Adding new dependencies
- Changing default behavior

### 10.2. Code Comments
- **Explain business logic** - Why is this calculation done this way?
- **Explain non-obvious decisions** - Why this algorithm/approach?
- **Document edge cases** - What happens with edge case inputs?
- **Clarify complex operations** - Break down complicated logic

**Good comment examples:**
```python
# Calculate days until stockout to prioritize urgent reorders
# Formula accounts for zero demand (no stockout risk)
days_until_stockout = current_stock / predicted_demand if predicted_demand > 0 else float('inf')
```

### 10.3. Inline Documentation
- Use type hints for all function parameters and returns
- Document complex data structures
- Explain algorithm choices when not obvious

---

## 11. PERFORMANCE AND EFFICIENCY

### 11.1. Code Efficiency
- **Avoid unnecessary computations** - Cache results when appropriate
- **Use vectorized operations** (pandas/numpy) instead of loops when possible
- **Lazy evaluation** - Only compute what's needed
- **Efficient data structures** - Use appropriate types (DataFrame vs dict vs list)

### 11.2. Memory Management
- Close file handles properly
- Close matplotlib figures: `plt.close()`
- Clear large intermediate variables when no longer needed
- Use generators for large datasets when possible

---

## 12. SECURITY AND BEST PRACTICES

### 12.1. Data Handling
- **Never commit sensitive data** - Check .gitignore
- **Validate user inputs** - Don't trust external data
- **Sanitize file paths** - Prevent path traversal attacks
- **Handle exceptions gracefully** - Don't expose internal errors

### 12.2. Code Safety
- **Use pathlib.Path** for file operations (safer than strings)
- **Check file existence** before operations
- **Use context managers** for file operations when possible
- **Validate data types** before processing

---

## 13. GIT AND VERSION CONTROL

### 13.1. Before Committing
- ✅ Code formatted with black
- ✅ All tests pass (`python main.py` runs successfully)
- ✅ Documentation updated
- ✅ No redundant files in output directories
- ✅ Sensible commit messages

### 13.2. Commit Message Format
- Use clear, descriptive messages
- Explain what changed and why
- Reference issue numbers if applicable
- Example: "Fix: Reduce redundant figure generation using OutputConfig"

---

## 14. COMMON MISTAKES TO AVOID

### ❌ **DO NOT:**
1. Create files without checking OutputConfig
2. Use random operations without random_state
3. Break existing functionality
4. Leave debug/print statements in production code
5. Commit large output files to git
6. Ignore linter warnings
7. Create redundant files
8. Modify files without testing
9. Forget to update README.md for significant changes
10. Remove functionality without documenting it

### ✅ **ALWAYS DO:**
1. Test after every modification
2. Use OutputConfig for file generation
3. Set random_state=42 for all random operations
4. Update documentation for changes
5. Follow existing code patterns
6. Handle errors gracefully
7. Use type hints
8. Write docstrings for functions
9. Format code with black
10. Verify `python main.py` works

---

## 15. VERIFICATION CHECKLIST

**Before marking any task as complete, verify:**

- [ ] `python main.py` runs without errors
- [ ] Code formatted with black (PEP 8 compliant)
- [ ] All functions have docstrings
- [ ] Type hints used for all function parameters
- [ ] `random_state=42` used for all random operations
- [ ] OutputConfig used for file generation
- [ ] No redundant files created
- [ ] Error handling implemented
- [ ] README.md updated if needed
- [ ] Dependencies listed in requirements.txt and environment.yml
- [ ] Code follows existing patterns
- [ ] Comments explain WHY, not WHAT

---

## 16. PROJECT STRUCTURE REMINDERS

```
data_proj/
├── data/
│   ├── raw/              # Original datasets ONLY
│   └── processed/        # Processed data
├── src/                  # ALL source code (modular)
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models.py
│   ├── evaluation.py
│   └── output_config.py  # Output control
├── models/
│   └── trained/          # Saved models
├── results/
│   ├── figures/          # Visualizations
│   └── metrics/          # Reports and metrics
├── main.py               # Entry point
├── requirements.txt      # Dependencies
├── environment.yml       # Conda environment
└── README.md             # Documentation
```

**Keep structure clean:**
- No empty directories
- No unused files
- Only essential outputs

---

## 17. QUICK REFERENCE

### Essential Commands
```bash
# Format code
black src/ main.py

# Check syntax
python3 -m py_compile src/*.py main.py

# Run project
python3 main.py

# Verify imports
python3 -c "import src.output_config; print('OK')"
```

### Key Constants
- `RANDOM_STATE = 42` - ALWAYS use for reproducibility
- Output paths: `results/metrics/`, `results/figures/`, `models/trained/`

### Important Files
- `src/output_config.py` - Control file generation
- `main.py` - Entry point (must run without errors)
- `README.md` - Keep documentation current

---

## 18. FINAL REMINDERS

**🎯 Primary Goals:**
1. Code must run: `python main.py` without errors
2. Professional quality: Clear, concise, well-documented
3. Reproducible: Use random_state=42 everywhere
4. Clean outputs: No redundant files
5. Teacher compliant: All requirements met

**🚨 Red Flags - Fix Immediately:**
- Code doesn't run
- Missing random_state
- Redundant files
- Broken functionality
- Missing documentation
- Hard-coded file paths (except defaults)

**✅ Success Criteria:**
- Code runs smoothly
- All tests pass
- No redundant outputs
- Clear documentation
- Follows all guidelines

---

**When in doubt:** Ask, don't assume. Better to clarify than break something.

**Remember:** This is an academic project - quality and adherence to requirements matter more than speed.

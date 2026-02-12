# README.md

This file is a report on a replication attempt of the mentioned study from Tuan et al. (2015). This study is to be found at https://doi.org/10.1371/journal.pntd.0003638.

**Key modeling goal**: Predict dengue diagnosis from clinical and laboratory variables using logistic regression, with an emphasis on the "EDC" (Early Dengue Classifier) model using Age, WBC (white blood cell count), and PLT (platelet count).

## Repository Structure

```
dengue-generalizability/
├── scripts/ # Python scripts used throughout the replication attempt.
├── results/ # Result files dump
└── data/    # Original dataset used
```

## Running Analyses
### Python Analysis (Replication/Extension)
```bash
# Main logistic regression analysis (Dataset 1: Tuan et al. 2015)
python code/logistic_regression_set01.py

# Backward stepwise variable selection from full 19-variable model
python code/backward_selection_set01.py

# S4 Table: Leave-one-site-out cross-validation for 5 models
python code/s4_table_loso.py

# Tree-based models (CART and Random Forest)
python code/tree_models_set01.py

# Data exploration
python code/histograms_set01.py
```

**Python Dependencies**: pandas, numpy, statsmodels, scikit-learn, matplotlib, scipy, openpyxl (for Excel reading)

## Key Data Characteristics

### Tuan et al. (2015) Dataset (set-01)
- **N = 5707** patients after complete-case analysis (dropping rows with any missing Table 3 variable)
- **Outcome**: `Lab_Confirmed_Dengue` coded as 1=dengue, 2=non-dengue
  - **CRITICAL**: Must recode to `Dengue = 2 - Lab_Confirmed_Dengue` (1=dengue, 0=non-dengue)
- **Prevalence**: ~29.6% dengue cases
- **7 sites** (SiteNo: 1, 2, 3, 4, 7, 15, 16) used for leave-one-site-out cross-validation

### Table 3 Variables (19 predictors)
Core variables used in Tuan et al. (2015):
```python
table3_vars = [
    'Age', 'Sex', 'BMI', 'DayDisease', 'Temp',
    'Vomiting', 'Abdo', 'Skin', 'Muco',
    'Flush', 'Hepatomegaly', 'Rash', 'Injection',
    'WBC', 'PLT', 'HCT', 'ALB', 'AST', 'CK',
]
```

**Boolean column handling**: Some columns (Flush, Hepatomegaly, Rash, Injection) may be boolean dtype. Convert to int before using with statsmodels:
```python
for col in table3_vars:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)
```

### EDC (Early Dengue Classifier)
The simplified 3-variable model: **Age, WBC, PLT**
- Paper reports AUC=0.829 at cutoff=0.333 (prevalence-based threshold)
- Coefficients should closely match Tuan et al. (2015) Table 3

## Critical Implementation Details

### 1. Complete-Case Analysis
All analyses use complete-case filtering: drop rows with ANY missing Table 3 variable. This should yield exactly **5707 rows** for Dataset 1.

### 2. Outcome Recoding
```python
df['Dengue'] = 2 - df['Lab_Confirmed_Dengue']  # 1=dengue, 0=non-dengue
```

### 3. Classification Threshold
Use **cutoff = 0.333** (or 0.33) for sensitivity/specificity calculations. This approximates the dengue prevalence in the training data.

### 4. Logistic Regression with Statsmodels
```python
import statsmodels.api as sm
X = sm.add_constant(df[['Age', 'WBC', 'PLT']], has_constant='add')
model = sm.Logit(y, X).fit(disp=0)
```

Note: Use `has_constant='add'` to force adding a constant column, avoiding auto-detection issues during LOSO CV.

### 5. CART Tree Approximation (rpart → sklearn)
Replicating R's `rpart` with sklearn's `DecisionTreeClassifier`:
- **Structural defaults**: `min_samples_split=20`, `min_samples_leaf=7`
- **Pruning**: Use `ccp_alpha=0.010` to approximate rpart's `cp=0.01` post-pruning
- Produces 3-7 leaf trees (vs 100s without pruning)
- Perfect replication of rpart is not possible due to different pruning algorithms (post-pruning vs pre-pruning)

### 6. Leave-One-Site-Out Cross-Validation
For Dataset 1's 7 sites, train on 6 sites and test on the held-out site. Report **mean (min-max)** across folds for AUC, sensitivity, specificity, PPV, NPV.

**AIC stepwise backward selection**: When used with LOSO, perform backward elimination on EACH training fold independently (not on the full dataset).

### 7. Calibration Metrics
```python
from scipy.special import logit as sp_logit

logit_pred = sp_logit(np.clip(predicted_probs, 1e-10, 1 - 1e-10))
X_cal = sm.add_constant(logit_pred)
cal_model = sm.Logit(y_true, X_cal).fit(disp=0)
cal_intercept = cal_model.params[0]
cal_slope = cal_model.params[1]
```
Ideal calibration: intercept ≈ 0, slope ≈ 1.

### 8. Windows Unicode Handling
For Windows systems with cp1252 encoding issues:
```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

## Analysis Architecture

### Python Script Organization

**Main scripts** (run independently):
1. **logistic_regression_set01.py**: Full analysis of EDC model on Dataset 1
   - Coefficient comparison with paper
   - ROC/AUC calculation
   - Sensitivity/specificity at cutoff=0.333
   - Calibration plots
   - Temporal validation (split at 2012-06-15)
   - Seasonal validation (dry=Jan-Jun, wet=Jul-Dec)
   - Leave-one-site-out cross-validation

2. **s4_table_loso.py**: Replicates paper's S4 Table comparing 5 modeling approaches via LOSO:
   - Full model (19 vars + Age×WBC + Age×PLT interactions)
   - AIC stepwise backward (from full model)
   - STAB (EDC: Age, WBC, PLT)
   - CART (decision tree)
   - Random Forest (500 trees)

3. **backward_selection_set01.py**: Stepwise backward elimination starting from full 19-variable model
   - Shows which variables are eliminated at each step
   - Reports final model vs EDC (AIC comparison)

4. **tree_models_set01.py**: CART and Random Forest comparison with 10-fold cross-validation

**Exploratory scripts**:
- **histograms_set01.py**: Distribution plots for Table 3 variables
- **correlation_heatmap_set01.py**: Correlation matrix visualization
- **missing_data_set01.py**: Missingness pattern analysis

### Key Helper Functions (Reusable Patterns)

```python
# Add constant safely
def add_const(data):
    return sm.add_constant(data, has_constant='add')

# Backward selection by AIC
def backward_select(train_X, train_y, candidates):
    # Iteratively remove variables that minimize AIC
    # Returns list of selected variable names

# Evaluate fold performance
def evaluate_fold(y_true, y_pred, cutoff=0.333):
    # Returns dict with: auc, sens, spec, ppv, npv, cal_int, cal_slope
```

## Paper-Specific Terminology

- **EDC**: Early Dengue Classifier (3-variable model: Age, WBC, PLT)
- **STAB**: Stability selection (in S4 Table context, this is the EDC)
- **Table 3**: The 19 clinical/lab predictors from Tuan et al. (2015)
- **S4 Table**: Leave-one-site-out performance comparison of 5 models
- **S5 Table**: Temporal and seasonal validation results

## Common Issues and Solutions

1. **Boolean columns causing statsmodels errors**: Convert to int before fitting
2. **CART specificity too low**: Increase `ccp_alpha` for more aggressive pruning
3. **Temporal split date boundary**: June 15, 2012 falls in TEST set (use `>=` not `>`)
4. **Seasonal counts mismatch**: Paper's 1,916 dry / 3,791 wet cannot be reproduced from available data
5. **AUC confidence intervals**: Python scripts compute point estimates only; use R's `pROC::ci.auc()` for DeLong CIs

## Model Performance Benchmarks (Dataset 1)

Expected results for **EDC (Age, WBC, PLT)** on full dataset:
- **AUC**: 0.828-0.829
- **Sensitivity**: 0.74-0.75 (at cutoff=0.333)
- **Specificity**: 0.76-0.77 (at cutoff=0.333)
- **Calibration slope**: ≈ 1.0

Expected results for **LOSO validation (S4 Table)**:
- STAB (EDC): AUC 0.831-0.835, Sens 0.751-0.795, Spec 0.726-0.745
- Full model: AUC 0.852-0.855
- Random Forest: AUC 0.852-0.872
- CART: AUC 0.750-0.774 (with high specificity, low sensitivity)

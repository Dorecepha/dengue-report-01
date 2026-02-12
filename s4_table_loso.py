import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from scipy.special import logit as sp_logit

# ======================================================================
# Data preparation
# ======================================================================
df = pd.read_excel('d:/Projects/dengue-generalizability/data/set-01/dengue-data-01.xls')

table3_vars = [
    'Age', 'Sex', 'BMI', 'DayDisease', 'Temp',
    'Vomiting', 'Abdo', 'Skin', 'Muco',
    'Flush', 'Hepatomegaly', 'Rash', 'Injection',
    'WBC', 'PLT', 'HCT', 'ALB', 'AST', 'CK',
]

df = df[['SiteNo'] + table3_vars + ['Lab_Confirmed_Dengue']].dropna()
df['Dengue'] = 2 - df['Lab_Confirmed_Dengue']

# Convert boolean columns to int for statsmodels
for col in table3_vars:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

# Interaction terms for Full model
df['Age_WBC'] = df['Age'] * df['WBC']
df['Age_PLT'] = df['Age'] * df['PLT']

full_vars = table3_vars + ['Age_WBC', 'Age_PLT']   # 21 terms
edc_vars = ['Age', 'WBC', 'PLT']

sites = sorted(df['SiteNo'].unique())

print(f"Dataset: {len(df)} rows, {len(table3_vars)} predictors, {len(sites)} sites")
print(f"Dengue prevalence: {df['Dengue'].mean():.3f}")
print(f"Sites: {sites}")
for site in sites:
    sub = df[df['SiteNo'] == site]
    print(f"  Site {site}: n={len(sub)}, dengue={sub['Dengue'].sum():.0f} "
          f"({sub['Dengue'].mean()*100:.1f}%)")

# ======================================================================
# Helper functions
# ======================================================================


def add_const(data):
    return sm.add_constant(data, has_constant='add')


def evaluate_fold(y_true, y_pred, cutoff=0.333):
    """Compute AUC, sens, spec, PPV, NPV, cal slope/intercept for one fold."""
    y_t = np.asarray(y_true).ravel()
    p_t = np.asarray(y_pred).ravel()

    if len(np.unique(y_t)) < 2:
        return None

    # AUC
    fpr_v, tpr_v, _ = roc_curve(y_t, p_t)
    auc_v = auc(fpr_v, tpr_v)

    # Classification at cutoff
    pos = p_t > cutoff
    tp = int(np.sum(pos & (y_t == 1)))
    fp = int(np.sum(pos & (y_t == 0)))
    tn = int(np.sum(~pos & (y_t == 0)))
    fn = int(np.sum(~pos & (y_t == 1)))

    n_pos = tp + fn
    n_neg = tn + fp
    n_pp = tp + fp
    n_pn = tn + fn

    sens_v = tp / n_pos if n_pos > 0 else np.nan
    spec_v = tn / n_neg if n_neg > 0 else np.nan
    ppv_v = tp / n_pp if n_pp > 0 else np.nan
    npv_v = tn / n_pn if n_pn > 0 else np.nan

    # Calibration slope and intercept
    logit_p = sp_logit(np.clip(p_t, 1e-10, 1 - 1e-10))
    try:
        X_cal = sm.add_constant(logit_p)
        cal_m = sm.Logit(y_t, X_cal).fit(disp=0)
        cal_int = float(np.asarray(cal_m.params).ravel()[0])
        cal_slope = float(np.asarray(cal_m.params).ravel()[1])
    except Exception as e:
        cal_int = np.nan
        cal_slope = np.nan

    return {
        'auc': auc_v,
        'sens': sens_v, 'spec': spec_v,
        'ppv': ppv_v, 'npv': npv_v,
        'cal_int': cal_int, 'cal_slope': cal_slope,
        'n': len(y_t), 'n_dengue': int(y_t.sum()),
    }


def backward_select(train_df, train_y, candidates, verbose=False):
    """Backward elimination by AIC, removing one term at a time."""
    current = list(candidates)
    X_c = add_const(train_df[current])
    best_aic = sm.Logit(train_y, X_c).fit(disp=0).aic
    if verbose:
        print(f"    Start: {len(current)} vars, AIC={best_aic:.2f}")
    improved = True
    step = 0
    while improved and len(current) > 1:
        improved = False
        best_drop = None
        for var in current:
            trial = [v for v in current if v != var]
            try:
                t_aic = sm.Logit(train_y, add_const(train_df[trial])).fit(disp=0).aic
                if t_aic < best_aic:
                    best_aic = t_aic
                    best_drop = var
                    improved = True
            except Exception:
                pass
        if improved:
            step += 1
            current.remove(best_drop)
            if verbose:
                print(f"    Step {step}: remove {best_drop}, "
                      f"{len(current)} vars, AIC={best_aic:.2f}")
    if verbose:
        print(f"    Final: {', '.join(current)}")
    return current


def fmt_mean_range(vals):
    clean = [v for v in vals if v is not None and not np.isnan(v)]
    if not clean:
        return "N/A"
    return f"{np.mean(clean):.3f} ({np.min(clean):.3f}-{np.max(clean):.3f})"


# ======================================================================
# LEAVE-ONE-SITE-OUT CROSS-VALIDATION
# ======================================================================
print(f"\n{'=' * 70}")
print(f"LEAVE-ONE-SITE-OUT CROSS-VALIDATION (S4 Table replication)")
print(f"{'=' * 70}")

# Storage for per-fold results
results = {k: [] for k in ['full', 'aic', 'stab', 'cart', 'rf']}
selected_vars_per_fold = {}

# Storage for pooled (all-fold concatenated) predictions
pooled_preds = {k: np.full(len(df), np.nan) for k in ['full', 'aic', 'stab', 'cart', 'rf']}
pooled_y = df['Dengue'].values

for i, site in enumerate(sites):
    print(f"\nFold {i+1}/{len(sites)}: Hold out site {site}")
    test_mask_bool = (df['SiteNo'].values == site)
    train_df = df[~test_mask_bool].copy()
    test_df = df[test_mask_bool].copy()
    y_train = train_df['Dengue']
    y_test = test_df['Dengue']
    print(f"  Train: {len(train_df)}, Test: {len(test_df)} "
          f"({int(y_test.sum())} dengue, {y_test.mean()*100:.1f}%)")

    # --- 1. Full model: all 19 Table 3 vars + Age*WBC + Age*PLT ---
    try:
        m = sm.Logit(y_train, add_const(train_df[full_vars])).fit(disp=0)
        preds = m.predict(add_const(test_df[full_vars]))
        pooled_preds['full'][test_mask_bool] = np.asarray(preds).ravel()
        r = evaluate_fold(y_test, preds)
        r['site'] = site
        results['full'].append(r)
        print(f"  Full:  AUC={r['auc']:.3f}, Sens={r['sens']:.3f}, Spec={r['spec']:.3f}")
    except Exception as e:
        print(f"  Full:  ERROR - {e}")
        results['full'].append(None)

    # --- 2. AIC stepwise backward from full model ---
    try:
        sel = backward_select(train_df, y_train, full_vars, verbose=False)
        selected_vars_per_fold[site] = sel
        m = sm.Logit(y_train, add_const(train_df[sel])).fit(disp=0)
        preds = m.predict(add_const(test_df[sel]))
        pooled_preds['aic'][test_mask_bool] = np.asarray(preds).ravel()
        r = evaluate_fold(y_test, preds)
        r['site'] = site
        r['n_vars'] = len(sel)
        results['aic'].append(r)
        print(f"  AIC:   {len(sel)} vars, AUC={r['auc']:.3f}, Sens={r['sens']:.3f}, "
              f"Spec={r['spec']:.3f}")
    except Exception as e:
        print(f"  AIC:   ERROR - {e}")
        results['aic'].append(None)

    # --- 3. STAB (stability selection result = EDC: Age, WBC, PLT) ---
    try:
        m = sm.Logit(y_train, sm.add_constant(train_df[edc_vars])).fit(disp=0)
        preds = m.predict(sm.add_constant(test_df[edc_vars]))
        pooled_preds['stab'][test_mask_bool] = np.asarray(preds).ravel()
        r = evaluate_fold(y_test, preds)
        r['site'] = site
        results['stab'].append(r)
        print(f"  STAB:  AUC={r['auc']:.3f}, Sens={r['sens']:.3f}, Spec={r['spec']:.3f}")
    except Exception as e:
        print(f"  STAB:  ERROR - {e}")
        results['stab'].append(None)

    # --- 4. CART (approximate rpart defaults) ---
    # rpart default: minsplit=20, minbucket=7, cp=0.01 (post-pruning)
    # Best sklearn approximation: ccp_alpha=0.010 + structural rpart defaults
    # (ccp_alpha corresponds to rpart cp ~ 0.024 given root_gini ~ 0.42)
    try:
        cart = DecisionTreeClassifier(
            min_samples_split=20,
            min_samples_leaf=7,
            ccp_alpha=0.010,
            random_state=42,
        )
        cart.fit(train_df[table3_vars].values, y_train)
        preds = cart.predict_proba(test_df[table3_vars].values)[:, 1]
        pooled_preds['cart'][test_mask_bool] = preds
        r = evaluate_fold(y_test, preds)
        r['site'] = site
        r['n_leaves'] = cart.get_n_leaves()
        results['cart'].append(r)
        print(f"  CART:  AUC={r['auc']:.3f}, Sens={r['sens']:.3f}, Spec={r['spec']:.3f} "
              f"({cart.get_n_leaves()} leaves)")
    except Exception as e:
        print(f"  CART:  ERROR - {e}")
        results['cart'].append(None)

    # --- 5. Random Forest (500 trees, all Table 3 variables) ---
    try:
        rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        rf.fit(train_df[table3_vars].values, y_train)
        preds = rf.predict_proba(test_df[table3_vars].values)[:, 1]
        pooled_preds['rf'][test_mask_bool] = preds
        r = evaluate_fold(y_test, preds)
        r['site'] = site
        results['rf'].append(r)
        print(f"  RF:    AUC={r['auc']:.3f}, Sens={r['sens']:.3f}, Spec={r['spec']:.3f}")
    except Exception as e:
        print(f"  RF:    ERROR - {e}")
        results['rf'].append(None)

# ======================================================================
# SUMMARY vs PAPER S4 TABLE
# ======================================================================
print(f"\n{'=' * 70}")
print(f"S4 TABLE COMPARISON: mean (min-max) across {len(sites)} leave-one-site-out folds")
print(f"{'=' * 70}")

# Paper S4 Table values
paper_s4 = {
    'full': {
        'AUC':  '0.855 (0.815-0.861)',
        'Sens': '0.785 (0.651-0.868)',
        'Spec': '0.760 (0.699-0.868)',
        'PPV':  '—',
        'NPV':  '—',
    },
    'aic': {
        'AUC':  '0.854 (0.815-0.861)',
        'Sens': '0.790 (0.659-0.890)',
        'Spec': '0.756 (0.699-0.868)',
        'PPV':  '—',
        'NPV':  '—',
    },
    'stab': {
        'AUC':  '0.835 (0.792-0.850)',
        'Sens': '0.795 (0.659-0.912)',
        'Spec': '0.726 (0.550-0.839)',
        'PPV':  '—',
        'NPV':  '—',
    },
    'cart': {
        'AUC':  '0.774 (0.703-0.777)',
        'Sens': '0.614 (0.459-0.736)',
        'Spec': '0.896 (0.735-0.939)',
        'PPV':  '—',
        'NPV':  '—',
    },
    'rf': {
        'AUC':  '0.872 (0.830-0.873)',
        'Sens': '0.812 (0.579-0.846)',
        'Spec': '0.809 (0.709-0.873)',
        'PPV':  '—',
        'NPV':  '—',
    },
}

model_labels = {
    'full': 'FULL (all 19 + Age*WBC + Age*PLT)',
    'aic':  'AIC stepwise (from full model)',
    'stab': 'STAB (Age + WBC + PLT)',
    'cart': 'CART (rpart defaults)',
    'rf':   'Random Forest (500 trees)',
}

metrics_order = [
    ('AUC',       'auc'),
    ('Sensitivity','sens'),
    ('Specificity','spec'),
    ('PPV',        'ppv'),
    ('NPV',        'npv'),
    ('Cal. slope', 'cal_slope'),
    ('Cal. int.',  'cal_int'),
]

for model_key in ['full', 'aic', 'stab', 'cart', 'rf']:
    valid = [r for r in results[model_key] if r is not None]
    print(f"\n  {model_labels[model_key]} (n_folds={len(valid)})")
    print(f"  {'Metric':<14} {'Paper S4':>30}  {'Fitted':>30}")
    print(f"  {'-' * 78}")
    for label, key in metrics_order:
        fitted_str = fmt_mean_range([r[key] for r in valid])
        paper_str = paper_s4[model_key].get(label, '—')
        print(f"  {label:<14} {paper_str:>30}  {fitted_str:>30}")

# ======================================================================
# POOLED STATISTICS (all held-out predictions concatenated)
# ======================================================================
print(f"\n{'=' * 70}")
print(f"POOLED HELD-OUT STATISTICS (all 7 folds concatenated)")
print(f"{'=' * 70}")
print(f"  {'Model':<35} {'AUC':>8} {'Sens':>8} {'Spec':>8} {'PPV':>8} {'NPV':>8}")
print(f"  {'-' * 75}")

cutoff = 0.333
for model_key, label in [('full','Full'), ('aic','AIC'), ('stab','STAB'),
                           ('cart','CART'), ('rf','RF')]:
    p = pooled_preds[model_key]
    y = pooled_y
    valid_mask = ~np.isnan(p)
    p_v = p[valid_mask]
    y_v = y[valid_mask]
    fpr_v, tpr_v, _ = roc_curve(y_v, p_v)
    auc_v = auc(fpr_v, tpr_v)
    pos = p_v > cutoff
    tp = np.sum(pos & (y_v == 1))
    fp = np.sum(pos & (y_v == 0))
    tn = np.sum(~pos & (y_v == 0))
    fn = np.sum(~pos & (y_v == 1))
    sens_v = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec_v = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv_v = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv_v = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    print(f"  {label:<35} {auc_v:>8.3f} {sens_v:>8.3f} {spec_v:>8.3f} {ppv_v:>8.3f} {npv_v:>8.3f}")

print(f"\n  Paper STAB row: AUC=0.835, Sens=0.795, Spec=0.726, PPV=0.505, NPV=0.909")

# Per-fold AIC variable selection summary
print(f"\n{'=' * 70}")
print(f"AIC STEPWISE: Variables selected in each fold")
print(f"{'=' * 70}")
for site in sites:
    if site in selected_vars_per_fold:
        sel = selected_vars_per_fold[site]
        print(f"  Site {site}: {len(sel)} vars: {', '.join(sel)}")

# Stability of AIC selection
all_selected = [v for sel in selected_vars_per_fold.values() for v in sel]
from collections import Counter
counts = Counter(all_selected)
print(f"\n  Variable selection frequency across {len(selected_vars_per_fold)} folds:")
for var, cnt in sorted(counts.items(), key=lambda x: -x[1]):
    marker = " <-- always selected" if cnt == len(sites) else ""
    print(f"    {var:<16} {cnt}/{len(sites)}{marker}")

print(f"\nDone.")

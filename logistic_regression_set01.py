import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Read and prepare data (matching the R code's transformation)
df = pd.read_excel('d:/Projects/dengue-generalizability/data/set-01/dengue-data-01.xls')
print(f"Raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# All variables from Tuan et al. Table 3 (used for complete-case filtering)
table3_vars = [
    'Age', 'Sex', 'BMI', 'DayDisease', 'Temp',
    'Vomiting', 'Abdo', 'Skin', 'Muco',
    'Flush', 'Hepatomegaly', 'Rash', 'Injection',
    'WBC', 'PLT', 'HCT', 'ALB', 'AST', 'CK',
]

# Check missingness per Table 3 variable before dropping
print("\nMissingness in Table 3 variables:")
for var in table3_vars:
    n_miss = df[var].isna().sum()
    if n_miss > 0:
        print(f"  {var:<15} {n_miss:>5} missing ({n_miss/len(df)*100:.1f}%)")

n_before = len(df)

# Drop rows missing ANY Table 3 variable, then keep columns needed for modelling
df = df[table3_vars + ['Lab_Confirmed_Dengue']].dropna()
print(f"\nAfter dropping rows missing any Table 3 variable: {len(df)} rows "
      f"(dropped {n_before - len(df)})")
print(f"Target from paper: 5707 rows")

# Recode outcome: original 1 = dengue, 2 = non-dengue -> 1 = dengue, 0 = non-dengue
df['Dengue'] = 2 - df['Lab_Confirmed_Dengue']

print(f"\nOutcome distribution:")
print(f"  Dengue (1):     {df['Dengue'].sum()}")
print(f"  Non-Dengue (0): {(1 - df['Dengue']).sum():.0f}")
print(f"  Prevalence:     {df['Dengue'].mean():.3f}")

# Fit logistic regression: Dengue ~ Age + WBC + PLT
X = df[['Age', 'WBC', 'PLT']]
X = sm.add_constant(X)
y = df['Dengue']

model = sm.Logit(y, X)
result = model.fit()

# Full summary
print("\n" + "=" * 70)
print("LOGISTIC REGRESSION SUMMARY")
print("=" * 70)
print(result.summary())

# Detailed coefficient comparison with Tuan et al. (2015)
print("\n" + "=" * 70)
print("COMPARISON WITH TUAN ET AL. (2015) REPORTED COEFFICIENTS")
print("=" * 70)

paper_coefs = {
    'const': 1.236,
    'Age': 0.139,
    'WBC': -0.254,
    'PLT': -0.006,
}

paper_ors = {
    'const': None,
    'Age': 1.15,
    'WBC': 0.78,
    'PLT': 0.94,
}

# Note: paper reports OR for PLT per 10-unit increase
plt_or_note = "(per 10-unit increase)"

fitted_coefs = result.params
fitted_ci = result.conf_int()
fitted_pvalues = result.pvalues

print(f"\n{'Variable':<12} {'Fitted coef':>12} {'Paper coef':>12} {'Difference':>12} "
      f"{'Fitted OR':>12} {'Paper OR':>12} {'p-value':>12}")
print("-" * 86)

for var in ['const', 'Age', 'WBC', 'PLT']:
    coef = fitted_coefs[var]
    paper = paper_coefs[var]
    diff = coef - paper
    fitted_or = np.exp(coef)
    paper_or = paper_ors[var]
    pval = fitted_pvalues[var]

    if var == 'PLT':
        # Paper reports OR per 10-unit increase for PLT
        or_str = f"{np.exp(coef * 10):.4f}"
        paper_or_str = f"{paper_or:.2f}"
        label = "PLT (OR/10)"
    elif var == 'const':
        or_str = "—"
        paper_or_str = "—"
        label = "Intercept"
    else:
        or_str = f"{fitted_or:.4f}"
        paper_or_str = f"{paper_or:.2f}"
        label = var

    print(f"{label:<12} {coef:>12.4f} {paper:>12.3f} {diff:>+12.4f} "
          f"{or_str:>12} {paper_or_str:>12} {pval:>12.4f}")

# Confidence intervals
print(f"\n{'Variable':<12} {'Coef':>10} {'95% CI':>24} {'OR':>10} {'95% CI (OR)':>24}")
print("-" * 82)

for var in ['const', 'Age', 'WBC', 'PLT']:
    coef = fitted_coefs[var]
    ci_lo, ci_hi = fitted_ci.loc[var]
    or_val = np.exp(coef)
    or_lo, or_hi = np.exp(ci_lo), np.exp(ci_hi)

    label = "Intercept" if var == 'const' else var
    print(f"{label:<12} {coef:>10.4f} [{ci_lo:>10.4f}, {ci_hi:>10.4f}] "
          f"{or_val:>10.4f} [{or_lo:>10.4f}, {or_hi:>10.4f}]")

print(f"\nNote: Tuan et al. report OR for PLT per 10-unit increase:")
print(f"  Fitted OR per 10-unit PLT increase: {np.exp(fitted_coefs['PLT'] * 10):.4f}")
print(f"  Paper OR per 10-unit PLT increase:  0.94")

# ======================================================================
# Predictions, ROC curve, and sensitivity/specificity trade-off
# ======================================================================

pred_prob = result.predict(X)

# ROC curve
fpr, tpr, roc_thresholds = roc_curve(y, pred_prob)
roc_auc = auc(fpr, tpr)

print(f"\n{'=' * 70}")
print(f"ROC / AUC")
print(f"{'=' * 70}")
print(f"  Fitted AUC: {roc_auc:.4f}")
print(f"  Paper AUC:  0.829")
print(f"  Difference: {roc_auc - 0.829:+.4f}")

# Sensitivity and specificity across all thresholds
thresholds = np.linspace(0, 1, 1001)
sensitivity = np.array([np.mean(pred_prob[y == 1] > t) for t in thresholds])
specificity = np.array([np.mean(pred_prob[y == 0] <= t) for t in thresholds])

# Youden's J statistic to find optimal cutoff
youden_j = sensitivity + specificity - 1
optimal_idx = np.argmax(youden_j)
optimal_cutoff = thresholds[optimal_idx]

# Look up sens/spec at the two marked cutoffs
def sens_spec_at(cutoff):
    s = np.mean(pred_prob[y == 1] > cutoff)
    sp = np.mean(pred_prob[y == 0] <= cutoff)
    return s, sp

sens_033, spec_033 = sens_spec_at(0.333)
sens_034, spec_034 = sens_spec_at(0.342)
sens_opt, spec_opt = sens_spec_at(optimal_cutoff)

print(f"\n  Cutoff = 0.333 (paper):    Sens = {sens_033:.3f}, Spec = {spec_033:.3f}")
print(f"  Cutoff = 0.342 (paper optimal): Sens = {sens_034:.3f}, Spec = {spec_034:.3f}")
print(f"  Cutoff = {optimal_cutoff:.3f} (Youden):  Sens = {sens_opt:.3f}, Spec = {spec_opt:.3f}")

# ---- Figure: two-panel plot ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel A: ROC curve
ax1.plot(fpr, tpr, color='#D6604D', linewidth=2,
         label=f'AUC = {roc_auc:.3f}  (paper: 0.829)')
ax1.plot([0, 1], [0, 1], '--', color='grey', linewidth=0.8)
ax1.set_xlabel('1 - Specificity (FPR)', fontsize=11)
ax1.set_ylabel('Sensitivity (TPR)', fontsize=11)
ax1.set_title('A.  ROC Curve', fontsize=12, loc='left', fontweight='bold')
ax1.legend(loc='lower right', fontsize=10, frameon=False)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)
ax1.set_aspect('equal')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel B: Sensitivity / Specificity trade-off
ax2.plot(thresholds, sensitivity, color='#D6604D', linewidth=2, label='Sensitivity')
ax2.plot(thresholds, specificity, color='#4393C3', linewidth=2, label='Specificity')

# Mark cutoff = 0.333
ax2.axvline(0.333, color='grey', linestyle='--', linewidth=1)
ax2.annotate(f'Cutoff = 0.333\nSens = {sens_033:.2f}, Spec = {spec_033:.2f}',
             xy=(0.333, sens_033), xytext=(0.333 + 0.07, sens_033 + 0.08),
             fontsize=8.5, ha='left',
             arrowprops=dict(arrowstyle='->', color='grey', lw=0.8))

# Mark cutoff = 0.342
ax2.axvline(0.342, color='black', linestyle=':', linewidth=1)
ax2.annotate(f'Optimal = 0.342\nSens = {sens_034:.2f}, Spec = {spec_034:.2f}',
             xy=(0.342, spec_034), xytext=(0.342 + 0.07, spec_034 - 0.10),
             fontsize=8.5, ha='left',
             arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

ax2.set_xlabel('Classification cutoff', fontsize=11)
ax2.set_ylabel('Rate', fontsize=11)
ax2.set_title('B.  Sensitivity / Specificity Trade-off', fontsize=12,
              loc='left', fontweight='bold')
ax2.legend(loc='center right', fontsize=10, frameon=False)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('d:/Projects/dengue-generalizability/images/roc_set01.pdf',
            bbox_inches='tight', dpi=300)
fig.savefig('d:/Projects/dengue-generalizability/images/roc_set01.png',
            bbox_inches='tight', dpi=300)
print(f"\nSaved to images/roc_set01.pdf and images/roc_set01.png")

# ======================================================================
# Classification performance at cutoff = 0.333 with Wilson CIs
# ======================================================================

cutoff = 0.333
y_arr = y.values
pred_pos = pred_prob > cutoff

TP = int(np.sum(pred_pos & (y_arr == 1)))
FP = int(np.sum(pred_pos & (y_arr == 0)))
TN = int(np.sum(~pred_pos & (y_arr == 0)))
FN = int(np.sum(~pred_pos & (y_arr == 1)))

n_pos = TP + FN   # actual positives
n_neg = TN + FP   # actual negatives
n_pred_pos = TP + FP
n_pred_neg = TN + FN

sens = TP / n_pos
spec = TN / n_neg
ppv  = TP / n_pred_pos
npv  = TN / n_pred_neg

sens_lo, sens_hi = proportion_confint(TP, n_pos, alpha=0.05, method='wilson')
spec_lo, spec_hi = proportion_confint(TN, n_neg, alpha=0.05, method='wilson')
ppv_lo, ppv_hi   = proportion_confint(TP, n_pred_pos, alpha=0.05, method='wilson')
npv_lo, npv_hi   = proportion_confint(TN, n_pred_neg, alpha=0.05, method='wilson')

print(f"\n{'=' * 70}")
print(f"CLASSIFICATION PERFORMANCE AT CUTOFF = {cutoff}")
print(f"{'=' * 70}")

print(f"\nConfusion matrix:")
print(f"                  Predicted +    Predicted -")
print(f"  Actual +  (TP)  {TP:>10}  (FN) {FN:>10}   | {n_pos} actual positives")
print(f"  Actual -  (FP)  {FP:>10}  (TN) {TN:>10}   | {n_neg} actual negatives")
print(f"                  {n_pred_pos:>10}       {n_pred_neg:>10}")

paper = {
    'Sensitivity': (74.8, 73.0, 76.8),
    'Specificity': (76.3, 75.2, 77.6),
    'PPV':         (57.1, 56.2, 59.0),
    'NPV':         (87.8, 86.8, 88.5),
}

fitted = {
    'Sensitivity': (sens * 100, sens_lo * 100, sens_hi * 100),
    'Specificity': (spec * 100, spec_lo * 100, spec_hi * 100),
    'PPV':         (ppv * 100, ppv_lo * 100, ppv_hi * 100),
    'NPV':         (npv * 100, npv_lo * 100, npv_hi * 100),
}

print(f"\n{'Metric':<14} {'Fitted':>22}   {'Paper':>22}   {'Diff':>6}")
print("-" * 70)
for metric in ['Sensitivity', 'Specificity', 'PPV', 'NPV']:
    f_val, f_lo, f_hi = fitted[metric]
    p_val, p_lo, p_hi = paper[metric]
    diff = f_val - p_val
    print(f"{metric:<14} {f_val:>5.1f}% ({f_lo:>5.1f}-{f_hi:>5.1f}%)   "
          f"{p_val:>5.1f}% ({p_lo:>5.1f}-{p_hi:>5.1f}%)   {diff:>+5.1f}%")

# ======================================================================
# Calibration plot and calibration statistics
# ======================================================================

from scipy.special import logit as sp_logit

# Decile calibration groups
n_groups = 10
cal_df = pd.DataFrame({'pred': pred_prob, 'obs': y_arr})
cal_df['decile'] = pd.qcut(cal_df['pred'], n_groups, labels=False, duplicates='drop')

cal_table = cal_df.groupby('decile').agg(
    n=('obs', 'size'),
    mean_pred=('pred', 'mean'),
    observed=('obs', 'mean'),
    n_events=('obs', 'sum'),
).reset_index()

print(f"\n{'=' * 70}")
print(f"CALIBRATION (DECILE GROUPS)")
print(f"{'=' * 70}")
print(f"\n{'Group':>5} {'N':>6} {'Events':>7} {'Mean pred':>10} {'Observed':>10} {'Diff':>8}")
print("-" * 50)
for _, row in cal_table.iterrows():
    diff = row['observed'] - row['mean_pred']
    print(f"{int(row['decile'])+1:>5} {int(row['n']):>6} {int(row['n_events']):>7} "
          f"{row['mean_pred']:>10.4f} {row['observed']:>10.4f} {diff:>+8.4f}")

# Calibration slope and intercept
# Regress outcome on logit(predicted probability) using logistic regression
logit_pred = sp_logit(np.clip(pred_prob, 1e-10, 1 - 1e-10))
X_cal = sm.add_constant(logit_pred)
cal_model = sm.Logit(y_arr, X_cal).fit(disp=0)
cal_intercept = cal_model.params.iloc[0]
cal_slope = cal_model.params.iloc[1]

# Brier score
brier = np.mean((pred_prob - y_arr) ** 2)
brier_max = np.mean(y_arr) * (1 - np.mean(y_arr))
brier_scaled = 1 - brier / brier_max

print(f"\nCalibration statistics:")
print(f"  {'Metric':<20} {'Fitted':>10} {'Paper':>10}")
print(f"  {'-' * 42}")
print(f"  {'Cal. intercept':<20} {cal_intercept:>10.4f} {'—':>10}")
print(f"  {'Cal. slope':<20} {cal_slope:>10.4f} {1.00:>10.2f}")
print(f"  {'C-statistic (AUC)':<20} {roc_auc:>10.4f} {0.83:>10.2f}")
print(f"  {'Brier score':<20} {brier:>10.4f} {'—':>10}")
print(f"  {'Brier scaled':<20} {brier_scaled:>10.4f} {0.31:>10.2f}")

# ---- Calibration figure ----
fig2, ax = plt.subplots(figsize=(6, 6))

# Ideal line
ax.plot([0, 1], [0, 1], '--', color='grey', linewidth=1, label='Ideal')

# Decile points with error bars (Wilson CIs on observed proportion)
for _, row in cal_table.iterrows():
    lo, hi = proportion_confint(int(row['n_events']), int(row['n']),
                                alpha=0.05, method='wilson')
    ax.errorbar(row['mean_pred'], row['observed'],
                yerr=[[row['observed'] - lo], [hi - row['observed']]],
                fmt='o', color='#D6604D', markersize=7, capsize=3,
                ecolor='#D6604D', elinewidth=1)

# LOESS-style smooth via lowess
from statsmodels.nonparametric.smoothers_lowess import lowess
smooth = lowess(cal_df['obs'], cal_df['pred'], frac=0.4)
ax.plot(smooth[:, 0], smooth[:, 1], color='#D6604D', linewidth=2,
        label='Observed (smoothed)')

# Rug plot of predicted probabilities
ax.plot(pred_prob[y_arr == 1], np.full(int(n_pos), -0.02), '|',
        color='#D6604D', alpha=0.15, markersize=4, label='Dengue')
ax.plot(pred_prob[y_arr == 0], np.full(int(n_neg), -0.04), '|',
        color='#4393C3', alpha=0.15, markersize=4, label='Non-dengue')

# Annotation box
stats_text = (f"Cal. slope = {cal_slope:.3f}\n"
              f"Cal. intercept = {cal_intercept:.4f}\n"
              f"C-statistic = {roc_auc:.3f}\n"
              f"Brier scaled = {brier_scaled:.3f}")
ax.text(0.62, 0.18, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='lightgrey', alpha=0.9))

ax.set_xlabel('Mean predicted probability', fontsize=11)
ax.set_ylabel('Observed proportion', fontsize=11)
ax.set_title('Calibration Plot (Decile Groups)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, frameon=False)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.06, 1.02)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig2.tight_layout()
fig2.savefig('d:/Projects/dengue-generalizability/images/calibration_set01.pdf',
             bbox_inches='tight', dpi=300)
fig2.savefig('d:/Projects/dengue-generalizability/images/calibration_set01.png',
             bbox_inches='tight', dpi=300)
print(f"\nSaved to images/calibration_set01.pdf and images/calibration_set01.png")

# ======================================================================
# S5 Table replication: temporal, seasonal, and leave-one-site-out
# ======================================================================

from scipy.special import logit as sp_logit_fn


def add_const(df_or_arr):
    """Always add a constant column, bypassing add_constant's auto-detection."""
    return sm.add_constant(df_or_arr, has_constant='add')


# Reload full data to get DateEnrol and SiteNo alongside Table 3 vars
df_full = pd.read_excel('d:/Projects/dengue-generalizability/data/set-01/dengue-data-01.xls')
df_full['DateEnrol'] = pd.to_datetime(df_full['DateEnrol'])
df_full = df_full[['DateEnrol', 'SiteNo'] + table3_vars + ['Lab_Confirmed_Dengue']].dropna()
df_full['Dengue'] = 2 - df_full['Lab_Confirmed_Dengue']
df_full['Month'] = df_full['DateEnrol'].dt.month

# Convert boolean columns to int for statsmodels compatibility
for col in table3_vars:
    if df_full[col].dtype == bool:
        df_full[col] = df_full[col].astype(int)

# Numeric Table 3 predictors for variable selection (exclude Sex which is coded 1/2)
numeric_predictors = [v for v in table3_vars if v != 'Sex'
                      and df_full[v].dtype in ('float64', 'int64')]


def evaluate_on_test(model, test_X, test_y, cutoff=0.333):
    """Calculate all S5 Table metrics for a train/test split."""
    preds = model.predict(test_X)
    y_t = np.asarray(test_y).ravel()
    p_t = np.asarray(preds).ravel()

    # AUC
    fpr_v, tpr_v, _ = roc_curve(y_t, p_t)
    auc_v = auc(fpr_v, tpr_v)

    # Classification at cutoff
    pos = p_t > cutoff
    tp = int(np.sum(pos & (y_t == 1)))
    fp = int(np.sum(pos & (y_t == 0)))
    tn = int(np.sum(~pos & (y_t == 0)))
    fn = int(np.sum(~pos & (y_t == 1)))

    sens_v = tp / (tp + fn)
    spec_v = tn / (tn + fp)
    ppv_v  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv_v  = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    # CIs (Wilson)
    sens_ci = proportion_confint(tp, tp + fn, method='wilson')
    spec_ci = proportion_confint(tn, tn + fp, method='wilson')
    ppv_ci  = proportion_confint(tp, tp + fp, method='wilson')
    npv_ci  = proportion_confint(tn, tn + fn, method='wilson')

    # Calibration slope/intercept
    logit_p = sp_logit_fn(np.clip(p_t, 1e-10, 1 - 1e-10))
    X_c = add_const(logit_p)
    cal_m = sm.Logit(y_t, X_c).fit(disp=0)
    cal_params = np.asarray(cal_m.params).ravel()

    return {
        'auc': auc_v,
        'sens': sens_v, 'sens_ci': sens_ci,
        'spec': spec_v, 'spec_ci': spec_ci,
        'ppv': ppv_v,   'ppv_ci': ppv_ci,
        'npv': npv_v,   'npv_ci': npv_ci,
        'cal_int': cal_params[0],
        'cal_slope': cal_params[1],
        'n_test': len(y_t),
        'n_dengue': int(y_t.sum()),
    }


def backward_select(train_X, train_y, candidates, verbose=False):
    """Backward elimination by AIC. Set verbose=True for step-by-step output."""
    current = list(candidates)
    X_c = add_const(train_X[current])
    best_aic = sm.Logit(train_y, X_c).fit(disp=0).aic
    if verbose:
        print(f"    Start: {len(current)} variables, AIC = {best_aic:.2f}")

    step = 0
    improved = True
    while improved and len(current) > 1:
        improved = False
        best_drop = None
        for var in current:
            trial = [v for v in current if v != var]
            X_t = add_const(train_X[trial])
            trial_aic = sm.Logit(train_y, X_t).fit(disp=0).aic
            if trial_aic < best_aic:
                best_aic = trial_aic
                best_drop = var
                improved = True
        if improved:
            step += 1
            current.remove(best_drop)
            if verbose:
                print(f"    Step {step}: Remove {best_drop:<16} -> "
                      f"{len(current)} vars, AIC = {best_aic:.2f}")
    if verbose:
        print(f"    Final: {', '.join(current)}")
    return current


def fmt_pct(val, ci):
    return f"{val:.3f} ({ci[0]:.3f}-{ci[1]:.3f})"


# --- Temporal validation ---
split_date = pd.Timestamp('2012-06-15')
train_df = df_full[df_full['DateEnrol'] < split_date]
test_df  = df_full[df_full['DateEnrol'] >= split_date]

y_train = train_df['Dengue']
y_test  = test_df['Dengue']

print(f"\n{'=' * 70}")
print(f"S5 TABLE REPLICATION")
print(f"{'=' * 70}")

print(f"\n--- TEMPORAL VALIDATION (train < 2012-06-15, test >= 2012-06-15) ---")
print(f"  Training: {len(train_df)} patients, "
      f"Dengue: {int(y_train.sum())} ({y_train.mean()*100:.1f}%)")
print(f"  Test:     {len(test_df)} patients, "
      f"Dengue: {int(y_test.sum())} ({y_test.mean()*100:.1f}%)")

# Approach A: Fixed Age + WBC + PLT (what we did before)
X_train_edc = sm.add_constant(train_df[['Age', 'WBC', 'PLT']])
X_test_edc  = sm.add_constant(test_df[['Age', 'WBC', 'PLT']])
temp_model_edc = sm.Logit(y_train, X_train_edc).fit(disp=0)
temp_res_edc = evaluate_on_test(temp_model_edc, X_test_edc, y_test)

print(f"\n  Approach A: Fixed model (Age + WBC + PLT)")
print(f"    AUC = {temp_res_edc['auc']:.3f}")

# Approach B: Backward selection on training set from all Table 3 vars
print(f"\n  Approach B: Backward selection on training set")
selected_vars = backward_select(train_df, y_train, numeric_predictors, verbose=True)
X_train_sel = add_const(train_df[selected_vars])
X_test_sel  = add_const(test_df[selected_vars])
temp_model_sel = sm.Logit(y_train, X_train_sel).fit(disp=0)
temp_res_sel = evaluate_on_test(temp_model_sel, X_test_sel, y_test)

print(f"    Selected {len(selected_vars)} variables: {', '.join(selected_vars)}")
print(f"    AUC = {temp_res_sel['auc']:.3f}")

# Print comparison table: Paper vs Fixed EDC vs Backward-selected
print(f"\n  {'Metric':<18} {'Paper (S5)':>24}   {'Fixed EDC':>24}   {'Backward sel.':>24}")
print(f"  {'-' * 96}")

s5_rows = [
    ('Cal. intercept', '-0.247', temp_res_edc['cal_int'], temp_res_sel['cal_int']),
    ('Cal. slope',     '0.940',  temp_res_edc['cal_slope'], temp_res_sel['cal_slope']),
    ('AUC',            '0.776',  temp_res_edc['auc'], temp_res_sel['auc']),
    ('Sensitivity',    '0.786 (0.770-0.812)',
     fmt_pct(temp_res_edc['sens'], temp_res_edc['sens_ci']),
     fmt_pct(temp_res_sel['sens'], temp_res_sel['sens_ci'])),
    ('Specificity',    '0.595 (0.583-0.612)',
     fmt_pct(temp_res_edc['spec'], temp_res_edc['spec_ci']),
     fmt_pct(temp_res_sel['spec'], temp_res_sel['spec_ci'])),
    ('PPV',            '0.460 (0.451-0.482)',
     fmt_pct(temp_res_edc['ppv'], temp_res_edc['ppv_ci']),
     fmt_pct(temp_res_sel['ppv'], temp_res_sel['ppv_ci'])),
    ('NPV',            '0.863 (0.850-0.877)',
     fmt_pct(temp_res_edc['npv'], temp_res_edc['npv_ci']),
     fmt_pct(temp_res_sel['npv'], temp_res_sel['npv_ci'])),
]

for row in s5_rows:
    metric, paper_val = row[0], row[1]
    edc_val = f"{row[2]:.3f}" if isinstance(row[2], float) else row[2]
    sel_val = f"{row[3]:.3f}" if isinstance(row[3], float) else row[3]
    print(f"  {metric:<18} {paper_val:>24}   {edc_val:>24}   {sel_val:>24}")


# --- Seasonal validation (apparent, fit on full data) ---
# Dry = Jan-Jun, Wet = Jul-Dec
dry_mask = df_full['Month'].between(1, 6)
wet_mask = df_full['Month'].between(7, 12)

# Fit model on full data (apparent performance by season)
X_full = sm.add_constant(df_full[['Age', 'WBC', 'PLT']])
y_full = df_full['Dengue']
full_model = sm.Logit(y_full, X_full).fit(disp=0)

dry_res = evaluate_on_test(full_model, X_full[dry_mask], y_full[dry_mask])
wet_res = evaluate_on_test(full_model, X_full[wet_mask], y_full[wet_mask])

print(f"\n--- SEASONAL (APPARENT) PERFORMANCE ---")
print(f"  Dry season (Jan-Jun): {dry_res['n_test']} patients, "
      f"Dengue: {dry_res['n_dengue']} ({dry_res['n_dengue']/dry_res['n_test']*100:.1f}%)")
print(f"  Wet season (Jul-Dec): {wet_res['n_test']} patients, "
      f"Dengue: {wet_res['n_dengue']} ({wet_res['n_dengue']/wet_res['n_test']*100:.1f}%)")

s5_seasonal = [
    ('AUC',         '0.831', f"{dry_res['auc']:.3f}",
                    '0.825', f"{wet_res['auc']:.3f}"),
    ('Sensitivity', '0.731 (0.726-0.750)', fmt_pct(dry_res['sens'], dry_res['sens_ci']),
                    '0.783 (0.771-0.798)', fmt_pct(wet_res['sens'], wet_res['sens_ci'])),
    ('Specificity', '0.792 (0.781-0.803)', fmt_pct(dry_res['spec'], dry_res['spec_ci']),
                    '0.758 (0.742-0.771)', fmt_pct(wet_res['spec'], wet_res['spec_ci'])),
    ('PPV',         '0.587 (0.579-0.602)', fmt_pct(dry_res['ppv'], dry_res['ppv_ci']),
                    '0.570 (0.561-0.586)', fmt_pct(wet_res['ppv'], wet_res['ppv_ci'])),
    ('NPV',         '0.911 (0.889-0.925)', fmt_pct(dry_res['npv'], dry_res['npv_ci']),
                    '0.864 (0.851-0.972)', fmt_pct(wet_res['npv'], wet_res['npv_ci'])),
]

print(f"\n  {'Metric':<14} {'Dry (S5)':>22}  {'Dry (fit)':>22}"
      f"  {'Wet (S5)':>22}  {'Wet (fit)':>22}")
print(f"  {'-' * 106}")
for row in s5_seasonal:
    print(f"  {row[0]:<14} {row[1]:>22}  {row[2]:>22}  {row[3]:>22}  {row[4]:>22}")


# --- Leave-one-site-out cross-validation ---
sites = sorted(df_full['SiteNo'].unique())


# ---- Approach 1: Fixed Age+WBC+PLT (same as before) ----
loso_fixed = []
pooled_pred_fixed = np.full(len(df_full), np.nan)

for site in sites:
    train_mask = df_full['SiteNo'] != site
    test_mask  = df_full['SiteNo'] == site

    train_s = df_full[train_mask]
    test_s  = df_full[test_mask]
    if test_s['Dengue'].nunique() < 2:
        continue

    X_tr = sm.add_constant(train_s[['Age', 'WBC', 'PLT']])
    y_tr = train_s['Dengue']
    X_te = sm.add_constant(test_s[['Age', 'WBC', 'PLT']])
    y_te = test_s['Dengue']

    m = sm.Logit(y_tr, X_tr).fit(disp=0)
    res = evaluate_on_test(m, X_te, y_te)
    res['site'] = site
    loso_fixed.append(res)

    pooled_pred_fixed[test_mask.values] = m.predict(X_te)

# Pooled metrics from concatenated held-out predictions
pooled_y = df_full['Dengue'].values
fpr_p, tpr_p, _ = roc_curve(pooled_y, pooled_pred_fixed)
pooled_auc_fixed = auc(fpr_p, tpr_p)
pos_p = pooled_pred_fixed > 0.333
tp_p = int(np.sum(pos_p & (pooled_y == 1)))
fp_p = int(np.sum(pos_p & (pooled_y == 0)))
tn_p = int(np.sum(~pos_p & (pooled_y == 0)))
fn_p = int(np.sum(~pos_p & (pooled_y == 1)))


# ---- Approach 2: Backward elimination per fold ----
loso_varsel = []
pooled_pred_varsel = np.full(len(df_full), np.nan)
selected_vars_per_site = {}

for site in sites:
    train_mask = df_full['SiteNo'] != site
    test_mask  = df_full['SiteNo'] == site

    train_s = df_full[train_mask]
    test_s  = df_full[test_mask]
    if test_s['Dengue'].nunique() < 2:
        continue

    # Backward elimination from all numeric Table 3 vars
    selected = backward_select(train_s, train_s['Dengue'], numeric_predictors)
    selected_vars_per_site[site] = selected

    X_tr = add_const(train_s[selected])
    y_tr = train_s['Dengue']
    X_te = add_const(test_s[selected])
    y_te = test_s['Dengue']

    m = sm.Logit(y_tr, X_tr).fit(disp=0)
    res = evaluate_on_test(m, X_te, y_te)
    res['site'] = site
    res['n_vars'] = len(selected)
    loso_varsel.append(res)

    pooled_pred_varsel[test_mask.values] = m.predict(X_te)

# Pooled metrics for variable-selection approach
fpr_v, tpr_v, _ = roc_curve(pooled_y, pooled_pred_varsel)
pooled_auc_varsel = auc(fpr_v, tpr_v)
pos_v = pooled_pred_varsel > 0.333
tp_v = int(np.sum(pos_v & (pooled_y == 1)))
fp_v = int(np.sum(pos_v & (pooled_y == 0)))
tn_v = int(np.sum(~pos_v & (pooled_y == 0)))
fn_v = int(np.sum(~pos_v & (pooled_y == 1)))


# ---- Print results ----

print(f"\n--- LEAVE-ONE-SITE-OUT CROSS-VALIDATION ---")
print(f"  {len(sites)} sites")

s5_loso = {
    'AUC':         '0.835',
    'Sensitivity': '0.795 (0.659-0.912)',
    'Specificity': '0.726 (0.550-0.839)',
    'PPV':         '0.505 (0.426-0.738)',
    'NPV':         '0.909 (0.748-0.945)',
    'Cal. slope':  '1.075',
    'Cal. int.':   '-0.200',
}


def summarize_loso(results, label):
    metrics_map = {
        'AUC':         [r['auc'] for r in results],
        'Sensitivity': [r['sens'] for r in results],
        'Specificity': [r['spec'] for r in results],
        'PPV':         [r['ppv'] for r in results],
        'NPV':         [r['npv'] for r in results],
        'Cal. slope':  [r['cal_slope'] for r in results],
        'Cal. int.':   [r['cal_int'] for r in results],
    }
    print(f"\n  {label}:")
    print(f"  {'Metric':<14} {'Paper S5':>26}   {'Fitted':>26}")
    print(f"  {'-' * 70}")
    for metric in ['AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV',
                   'Cal. slope', 'Cal. int.']:
        vals = metrics_map[metric]
        fitted_str = f"{np.mean(vals):.3f} ({np.min(vals):.3f}-{np.max(vals):.3f})"
        print(f"  {metric:<14} {s5_loso[metric]:>26}   {fitted_str:>26}")


summarize_loso(loso_fixed, "Approach 1: Fixed (Age + WBC + PLT)")
summarize_loso(loso_varsel, "Approach 2: Backward elimination per fold")

# Pooled predictions comparison
print(f"\n  Pooled held-out predictions (all sites concatenated):")
print(f"  {'Metric':<14} {'Fixed (Age+WBC+PLT)':>22} {'Backward elim.':>22} {'Paper S5':>22}")
print(f"  {'-' * 84}")

for label, tp_x, fp_x, tn_x, fn_x, auc_x in [
    ('Fixed',    tp_p, fp_p, tn_p, fn_p, pooled_auc_fixed),
    ('BackElim', tp_v, fp_v, tn_v, fn_v, pooled_auc_varsel),
]:
    pass  # just building data for the table below

metrics_pooled = []
for label, tp_x, fp_x, tn_x, fn_x, auc_x in [
    ('Fixed',    tp_p, fp_p, tn_p, fn_p, pooled_auc_fixed),
    ('BackElim', tp_v, fp_v, tn_v, fn_v, pooled_auc_varsel),
]:
    metrics_pooled.append({
        'label': label,
        'AUC': auc_x,
        'Sens': tp_x / (tp_x + fn_x),
        'Spec': tn_x / (tn_x + fp_x),
        'PPV':  tp_x / (tp_x + fp_x),
        'NPV':  tn_x / (tn_x + fn_x),
    })

s5_pooled = {'AUC': '0.835', 'Sens': '0.795', 'Spec': '0.726',
             'PPV': '0.505', 'NPV': '0.909'}

for metric in ['AUC', 'Sens', 'Spec', 'PPV', 'NPV']:
    v1 = f"{metrics_pooled[0][metric]:.3f}"
    v2 = f"{metrics_pooled[1][metric]:.3f}"
    print(f"  {metric:<14} {v1:>22} {v2:>22} {s5_pooled[metric]:>22}")


# Per-site detail for fixed model
print(f"\n  Per-site detail (fixed Age+WBC+PLT):")
print(f"  {'Site':>6} {'N':>6} {'Dengue':>7} {'Prev':>6} {'AUC':>6} "
      f"{'Sens':>6} {'Spec':>6} {'PPV':>6} {'NPV':>6} {'CalSlp':>7}")
print(f"  {'-' * 72}")
for r in loso_fixed:
    prev = r['n_dengue'] / r['n_test'] * 100
    print(f"  {r['site']:>6} {r['n_test']:>6} {r['n_dengue']:>7} {prev:>5.1f}% "
          f"{r['auc']:>6.3f} {r['sens']:>6.3f} {r['spec']:>6.3f} "
          f"{r['ppv']:>6.3f} {r['npv']:>6.3f} {r['cal_slope']:>7.3f}")

# Per-site detail for backward elimination
print(f"\n  Per-site detail (backward elimination):")
print(f"  {'Site':>6} {'N':>6} {'#Vars':>6} {'AUC':>6} "
      f"{'Sens':>6} {'Spec':>6} {'PPV':>6} {'NPV':>6}  Selected variables")
print(f"  {'-' * 82}")
for r in loso_varsel:
    site = r['site']
    sel = selected_vars_per_site[site]
    vars_str = ', '.join(sel)
    print(f"  {site:>6} {r['n_test']:>6} {r['n_vars']:>6} {r['auc']:>6.3f} "
          f"{r['sens']:>6.3f} {r['spec']:>6.3f} "
          f"{r['ppv']:>6.3f} {r['npv']:>6.3f}  {vars_str}")

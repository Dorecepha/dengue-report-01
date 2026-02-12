import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ======================================================================
# Data preparation (same as logistic regression script)
# ======================================================================
df = pd.read_excel('d:/Projects/dengue-generalizability/data/set-01/dengue-data-01.xls')

table3_vars = [
    'Age', 'Sex', 'BMI', 'DayDisease', 'Temp',
    'Vomiting', 'Abdo', 'Skin', 'Muco',
    'Flush', 'Hepatomegaly', 'Rash', 'Injection',
    'WBC', 'PLT', 'HCT', 'ALB', 'AST', 'CK',
]

df = df[table3_vars + ['Lab_Confirmed_Dengue']].dropna()
df['Dengue'] = 2 - df['Lab_Confirmed_Dengue']

# Convert boolean columns to int
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

y = df['Dengue'].values
X_all = df[table3_vars].values
X_edc = df[['Age', 'WBC', 'PLT']].values

print(f"Dataset: {len(df)} rows, {len(table3_vars)} predictors")
print(f"Dengue prevalence: {y.mean():.3f}")

# ======================================================================
# 1. CART (Classification Tree) — apparent performance
# ======================================================================
print(f"\n{'=' * 70}")
print(f"CLASSIFICATION TREE (CART)")
print(f"{'=' * 70}")

# Fit with cost-complexity pruning via cross-validation
# First fit a large tree, then find optimal alpha
cart_full = DecisionTreeClassifier(random_state=42)
cart_full.fit(X_all, y)
print(f"\nUnpruned tree: {cart_full.get_n_leaves()} leaves, "
      f"depth {cart_full.get_depth()}")
auc_full = roc_auc_score(y, cart_full.predict_proba(X_all)[:, 1])
print(f"Apparent AUC (unpruned, all vars): {auc_full:.4f}  [overfitted]")

# Cost-complexity pruning path
path = cart_full.cost_complexity_pruning_path(X_all, y)
ccp_alphas = path.ccp_alphas

# 10-fold CV to find best alpha
from sklearn.model_selection import cross_val_score
best_alpha = 0
best_cv_auc = 0
alpha_results = []
# Test a range of alphas (subsample for speed)
test_alphas = ccp_alphas[::max(1, len(ccp_alphas) // 50)]
for alpha in test_alphas:
    tree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(tree, X_all, y, cv=10, scoring='roc_auc')
    alpha_results.append((alpha, scores.mean(), scores.std()))
    if scores.mean() > best_cv_auc:
        best_cv_auc = scores.mean()
        best_alpha = alpha

print(f"\nCost-complexity pruning (10-fold CV):")
print(f"  Best ccp_alpha: {best_alpha:.6f}")
print(f"  Best CV AUC:    {best_cv_auc:.4f}")

# Fit pruned tree
cart_pruned = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
cart_pruned.fit(X_all, y)
auc_pruned_apparent = roc_auc_score(y, cart_pruned.predict_proba(X_all)[:, 1])
print(f"\nPruned tree: {cart_pruned.get_n_leaves()} leaves, "
      f"depth {cart_pruned.get_depth()}")
print(f"Apparent AUC (pruned): {auc_pruned_apparent:.4f}")

# Feature importance
importances = cart_pruned.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print(f"\nFeature importances (pruned tree):")
for i in sorted_idx:
    if importances[i] > 0:
        print(f"  {table3_vars[i]:<16} {importances[i]:.4f}")

# Print tree structure (if not too large)
if cart_pruned.get_n_leaves() <= 20:
    print(f"\nTree structure:")
    tree_text = export_text(cart_pruned, feature_names=table3_vars, max_depth=10)
    print(tree_text)

# 10-fold CV AUC for pruned CART
cart_cv_probs = cross_val_predict(
    DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42),
    X_all, y, cv=10, method='predict_proba')[:, 1]
cart_cv_auc = roc_auc_score(y, cart_cv_probs)
print(f"10-fold CV AUC (pruned CART, all vars): {cart_cv_auc:.4f}")

# CART with only EDC variables
cart_edc = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
cart_edc_cv_probs = cross_val_predict(cart_edc, X_edc, y, cv=10,
                                       method='predict_proba')[:, 1]
cart_edc_cv_auc = roc_auc_score(y, cart_edc_cv_probs)
print(f"10-fold CV AUC (pruned CART, Age+WBC+PLT): {cart_edc_cv_auc:.4f}")


# ======================================================================
# 2. RANDOM FOREST
# ======================================================================
print(f"\n{'=' * 70}")
print(f"RANDOM FOREST")
print(f"{'=' * 70}")

# All 19 variables
rf_all = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
rf_all.fit(X_all, y)
auc_rf_apparent = roc_auc_score(y, rf_all.predict_proba(X_all)[:, 1])
print(f"\nRandom Forest (500 trees, all {len(table3_vars)} vars):")
print(f"  Apparent AUC: {auc_rf_apparent:.4f}  [optimistic]")

# 10-fold CV
rf_cv_probs = cross_val_predict(
    RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
    X_all, y, cv=10, method='predict_proba')[:, 1]
rf_cv_auc = roc_auc_score(y, rf_cv_probs)
print(f"  10-fold CV AUC: {rf_cv_auc:.4f}")

# Feature importance
rf_imp = rf_all.feature_importances_
rf_sorted = np.argsort(rf_imp)[::-1]
print(f"\n  Feature importances (Random Forest):")
for i in rf_sorted:
    print(f"    {table3_vars[i]:<16} {rf_imp[i]:.4f}")

# RF with only EDC variables
rf_edc = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
rf_edc_cv_probs = cross_val_predict(rf_edc, X_edc, y, cv=10,
                                     method='predict_proba')[:, 1]
rf_edc_cv_auc = roc_auc_score(y, rf_edc_cv_probs)
print(f"\nRandom Forest (500 trees, Age+WBC+PLT only):")
print(f"  10-fold CV AUC: {rf_edc_cv_auc:.4f}")


# ======================================================================
# 3. COMPARISON TABLE
# ======================================================================
print(f"\n{'=' * 70}")
print(f"MODEL COMPARISON (10-fold CV AUC)")
print(f"{'=' * 70}")

# Logistic regression CV AUC for reference
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
lr_cv_probs_all = np.zeros(len(y))
lr_cv_probs_edc = np.zeros(len(y))

X_all_df = df[table3_vars].copy()
X_edc_df = df[['Age', 'WBC', 'PLT']].copy()

for train_idx, test_idx in kf.split(X_all, y):
    # All variables
    X_tr = sm.add_constant(X_all_df.iloc[train_idx], has_constant='add')
    X_te = sm.add_constant(X_all_df.iloc[test_idx], has_constant='add')
    m = sm.Logit(y[train_idx], X_tr).fit(disp=0)
    lr_cv_probs_all[test_idx] = m.predict(X_te)

    # EDC variables
    X_tr_e = sm.add_constant(X_edc_df.iloc[train_idx])
    X_te_e = sm.add_constant(X_edc_df.iloc[test_idx])
    m_e = sm.Logit(y[train_idx], X_tr_e).fit(disp=0)
    lr_cv_probs_edc[test_idx] = m_e.predict(X_te_e)

lr_cv_auc_all = roc_auc_score(y, lr_cv_probs_all)
lr_cv_auc_edc = roc_auc_score(y, lr_cv_probs_edc)

print(f"\n{'Model':<40} {'Variables':>12} {'CV AUC':>10}")
print(f"{'-' * 65}")
print(f"{'Logistic Regression (EDC)':<40} {'3':>12} {lr_cv_auc_edc:>10.4f}")
print(f"{'Logistic Regression (all Table 3)':<40} {'19':>12} {lr_cv_auc_all:>10.4f}")
print(f"{'CART pruned (EDC vars)':<40} {'3':>12} {cart_edc_cv_auc:>10.4f}")
print(f"{'CART pruned (all Table 3)':<40} {'19':>12} {cart_cv_auc:>10.4f}")
print(f"{'Random Forest (EDC vars)':<40} {'3':>12} {rf_edc_cv_auc:>10.4f}")
print(f"{'Random Forest (all Table 3)':<40} {'19':>12} {rf_cv_auc:>10.4f}")

print(f"\nPaper reported (apparent, Table 4):")
print(f"  Logistic Regression (EDC):  AUC = 0.829")
print(f"  CART:                       AUC = 0.829")
print(f"  Random Forest:              AUC = 0.841")

# Gains from adding variables
print(f"\n{'Model':<30} {'3-var AUC':>10} {'19-var AUC':>11} {'Gain':>8}")
print(f"{'-' * 62}")
print(f"{'Logistic Regression':<30} {lr_cv_auc_edc:>10.4f} {lr_cv_auc_all:>11.4f} "
      f"{lr_cv_auc_all - lr_cv_auc_edc:>+8.4f}")
print(f"{'CART (pruned)':<30} {cart_edc_cv_auc:>10.4f} {cart_cv_auc:>11.4f} "
      f"{cart_cv_auc - cart_edc_cv_auc:>+8.4f}")
print(f"{'Random Forest':<30} {rf_edc_cv_auc:>10.4f} {rf_cv_auc:>11.4f} "
      f"{rf_cv_auc - rf_edc_cv_auc:>+8.4f}")


# ======================================================================
# 4. ROC CURVES COMPARISON PLOT
# ======================================================================
fig, ax = plt.subplots(figsize=(7, 7))

models = [
    ('LR (Age+WBC+PLT)', lr_cv_probs_edc, '#2166AC', '-'),
    ('LR (all 19 vars)', lr_cv_probs_all, '#4393C3', '--'),
    ('CART pruned (all 19)', cart_cv_probs, '#E08214', '-'),
    ('Random Forest (all 19)', rf_cv_probs, '#D6604D', '-'),
    ('RF (Age+WBC+PLT)', rf_edc_cv_probs, '#D6604D', '--'),
]

for label, probs, color, ls in models:
    fpr, tpr, _ = roc_curve(y, probs)
    auc_val = roc_auc_score(y, probs)
    ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.8,
            label=f'{label} (AUC={auc_val:.3f})')

ax.plot([0, 1], [0, 1], '--', color='grey', linewidth=0.8)
ax.set_xlabel('1 - Specificity (FPR)', fontsize=11)
ax.set_ylabel('Sensitivity (TPR)', fontsize=11)
ax.set_title('ROC Comparison: Logistic Regression vs Tree Models\n(10-fold CV)',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9, frameon=False)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('d:/Projects/dengue-generalizability/images/roc_comparison_set01.pdf',
            bbox_inches='tight', dpi=300)
fig.savefig('d:/Projects/dengue-generalizability/images/roc_comparison_set01.png',
            bbox_inches='tight', dpi=300)
print(f"\nSaved ROC comparison to images/roc_comparison_set01.pdf and .png")

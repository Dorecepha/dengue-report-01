import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Read and prepare data (matching the R code's transformation)
df = pd.read_excel('d:/Projects/dengue-generalizability/data/set-01/dengue-data-01.xls')
df = df[['Age', 'WBC', 'PLT', 'Lab_Confirmed_Dengue']].dropna()
df['Dengue'] = 2 - df['Lab_Confirmed_Dengue']  # 1 = Dengue, 0 = Non-Dengue
df['Group'] = df['Dengue'].map({1: 'Dengue', 0: 'Non-Dengue'})

dengue = df[df['Dengue'] == 1]
non_dengue = df[df['Dengue'] == 0]

# Plot configuration
variables = [
    ('Age', 'Age (years)', 1),
    ('WBC', 'WBC (×10³/µL)', 2),
    ('PLT', 'Platelet Count (×10³/µL)', 20),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, (var, label, bw) in zip(axes, variables):
    lo = df[var].min()
    hi = df[var].max()
    bins = range(int(lo), int(hi) + bw + 1, bw)

    ax.hist(non_dengue[var], bins=bins, alpha=0.6, label='Non-Dengue',
            color='#4393C3', edgecolor='white', linewidth=0.5)
    ax.hist(dengue[var], bins=bins, alpha=0.6, label='Dengue',
            color='#D6604D', edgecolor='white', linewidth=0.5)
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Distribution of Age, WBC, and Platelet Count by Dengue Status (Dataset 1)',
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig('d:/Projects/dengue-generalizability/images/histograms_set01.pdf',
            bbox_inches='tight', dpi=300)
fig.savefig('d:/Projects/dengue-generalizability/images/histograms_set01.png',
            bbox_inches='tight', dpi=300)
print('Saved to images/histograms_set01.pdf and images/histograms_set01.png')

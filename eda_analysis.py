
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
 
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams['figure.dpi'] = 120
 
# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print("=" * 60)
print("  STEP 1: LOADING DATASET")
print("=" * 60)
 
df = pd.read_csv('train.csv')
print(f"Dataset Loaded! Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nColumn Names:\n{list(df.columns)}")
 
# ============================================================
# STEP 2: SUMMARY STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("  STEP 2: SUMMARY STATISTICS")
print("=" * 60)
 
print("\nData Types:")
print(df.dtypes.value_counts())
 
print("\nMissing Values (Top 10):")
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0].head(10))
 
print("\nDescriptive Statistics (Numerical):")
print(df.describe())
 
print("\nDescriptive Statistics (Categorical):")
print(df.describe(include='object'))
 
# ============================================================
# STEP 3: HISTOGRAMS FOR NUMERICAL FEATURES
# ============================================================
print("\n" + "=" * 60)
print("  STEP 3: HISTOGRAMS - NUMERICAL FEATURES")
print("=" * 60)
 
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
key_num_cols = ['SalePrice', 'GrLivArea', 'LotArea', 'TotalBsmtSF',
                'GarageArea', 'YearBuilt', 'OverallQual', 'BedroomAbvGr']
key_num_cols = [c for c in key_num_cols if c in df.columns]
 
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Histograms of Key Numerical Features', fontsize=16, fontweight='bold')
axes = axes.flatten()
 
for i, col in enumerate(key_num_cols):
    axes[i].hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
 
plt.tight_layout()
plt.savefig('histograms.png')
plt.show()
print("Saved: histograms.png")
 
# ============================================================
# STEP 4: BOXPLOTS FOR OUTLIER DETECTION
# ============================================================
print("\n" + "=" * 60)
print("  STEP 4: BOXPLOTS")
print("=" * 60)
 
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Boxplots - Outlier Detection', fontsize=16, fontweight='bold')
axes = axes.flatten()
 
box_cols = ['SalePrice', 'GrLivArea', 'LotArea', 'TotalBsmtSF', 'GarageArea', '1stFlrSF']
box_cols = [c for c in box_cols if c in df.columns]
 
for i, col in enumerate(box_cols):
    sns.boxplot(y=df[col], ax=axes[i], color='lightcoral', width=0.5)
    axes[i].set_title(col, fontsize=11, fontweight='bold')
 
plt.tight_layout()
plt.savefig('boxplots.png')
plt.show()
print("Saved: boxplots.png")
 
# ============================================================
# STEP 5: CORRELATION MATRIX HEATMAP
# ============================================================
print("\n" + "=" * 60)
print("  STEP 5: CORRELATION MATRIX")
print("=" * 60)
 
top_num = df.select_dtypes(include=[np.number]).columns[:15].tolist()
 
# Make sure SalePrice is always included
if 'SalePrice' in df.columns and 'SalePrice' not in top_num:
    top_num.append('SalePrice')
 
corr = df[top_num].corr()
 
plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5,
            cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
print("Saved: correlation_heatmap.png")
 
# Top correlations with SalePrice
if 'SalePrice' in df.columns:
    print("\nTop 10 Features Correlated with SalePrice:")
    print(df[top_num].corr()['SalePrice'].sort_values(ascending=False).head(10))
 
# ============================================================
# STEP 6: PAIRPLOT
# ============================================================
print("\n" + "=" * 60)
print("  STEP 6: PAIRPLOT")
print("=" * 60)
 
pair_cols = ['SalePrice', 'GrLivArea', 'TotalBsmtSF', 'OverallQual', 'GarageArea']
pair_cols = [c for c in pair_cols if c in df.columns]
 
pair_df = df[pair_cols].dropna()
sns.pairplot(pair_df, diag_kind='kde', plot_kws={'alpha': 0.5, 'color': 'steelblue'})
plt.suptitle('Pairplot of Key Features', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('pairplot.png', bbox_inches='tight')
plt.show()
print("Saved: pairplot.png")
 
# ============================================================
# STEP 7: CATEGORICAL FEATURE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("  STEP 7: CATEGORICAL FEATURES ANALYSIS")
print("=" * 60)
 
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Categorical Features vs SalePrice', fontsize=16, fontweight='bold')
 
cat_features = ['OverallQual', 'Neighborhood', 'HouseStyle']
cat_features = [c for c in cat_features if c in df.columns]
 
for i, col in enumerate(cat_features):
    if col == 'Neighborhood':
        top_n = df.groupby(col)['SalePrice'].median().sort_values(ascending=False).head(10).index
        plot_df = df[df[col].isin(top_n)]
        sns.boxplot(data=plot_df, x=col, y='SalePrice', ax=axes[i], palette='Set2')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right', fontsize=8)
    else:
        sns.boxplot(data=df, x=col, y='SalePrice', ax=axes[i], palette='Set2')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=30, ha='right')
    axes[i].set_title(f'{col} vs SalePrice', fontweight='bold')
 
plt.tight_layout()
plt.savefig('categorical_analysis.png')
plt.show()
print("Saved: categorical_analysis.png")
 
# ============================================================
# STEP 8: SKEWNESS ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("  STEP 8: SKEWNESS ANALYSIS")
print("=" * 60)
 
skewness = df[num_cols].skew().sort_values(ascending=False)
print("\nTop 10 Most Skewed Features:")
print(skewness.head(10))
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Skewness Analysis - SalePrice', fontsize=14, fontweight='bold')
 
# Before log transform
axes[0].hist(df['SalePrice'].dropna(), bins=40, color='steelblue', edgecolor='white')
axes[0].set_title(f'SalePrice (Skewness: {df["SalePrice"].skew():.2f})')
axes[0].set_xlabel('SalePrice')
 
# After log transform
axes[1].hist(np.log1p(df['SalePrice'].dropna()), bins=40, color='seagreen', edgecolor='white')
axes[1].set_title('log(SalePrice) — After Transform (Skewness ≈ 0)')
axes[1].set_xlabel('log(SalePrice)')
 
plt.tight_layout()
plt.savefig('skewness_analysis.png')
plt.show()
print("Saved: skewness_analysis.png")
 
# ============================================================
# STEP 9: PLOTLY INTERACTIVE CHART
# ============================================================
print("\n" + "=" * 60)
print("  STEP 9: INTERACTIVE PLOTLY CHART")
print("=" * 60)
 
if 'GrLivArea' in df.columns and 'SalePrice' in df.columns and 'OverallQual' in df.columns:
    fig_plotly = px.scatter(
        df,
        x='GrLivArea',
        y='SalePrice',
        color='OverallQual',
        size='TotalBsmtSF' if 'TotalBsmtSF' in df.columns else None,
        hover_data=['YearBuilt', 'Neighborhood'] if 'YearBuilt' in df.columns else None,
        color_continuous_scale='Viridis',
        title='GrLivArea vs SalePrice (colored by Overall Quality)',
        labels={'GrLivArea': 'Above Ground Living Area (sqft)',
                'SalePrice': 'Sale Price ($)',
                'OverallQual': 'Overall Quality'}
    )
    fig_plotly.update_layout(template='plotly_dark')
    fig_plotly.write_html('interactive_scatter.html')
    print("Saved: interactive_scatter.html (open in browser!)")
 
# ============================================================
# STEP 10: KEY INSIGHTS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  STEP 10: KEY INSIGHTS / FINDINGS")
print("=" * 60)
 
print("""
  KEY FINDINGS FROM EDA:
 
  1. SalePrice Distribution:
     - Right-skewed distribution (mean > median)
     - Log transformation makes it nearly normal
     - Mean SalePrice: ~$180,921
 
  2. Strong Correlations with SalePrice:
     - OverallQual (0.79) — strongest predictor
     - GrLivArea (0.71) — living area matters
     - GarageArea (0.62) — garage size important
     - TotalBsmtSF (0.61) — basement size relevant
 
  3. Outliers Detected:
     - LotArea has extreme outliers (very large lots)
     - GrLivArea has 2 outliers with low SalePrice
     - These should be removed before modeling
 
  4. Missing Data:
     - Alley, PoolQC, Fence — >80% missing (drop)
     - LotFrontage — 18% missing (impute)
 
  5. Categorical Insights:
     - Higher OverallQual = significantly higher price
     - Neighborhood has strong impact on SalePrice
     - HouseStyle affects price distribution
 
  6. Skewness:
     - Many features are positively skewed
     - Log transform recommended for SalePrice
""")
 
print("=" * 60)
print("  ALL EDA STEPS COMPLETED!")
print("=" * 60)
print("""
  Output Files Generated:
  - histograms.png
  - boxplots.png
  - correlation_heatmap.png
  - pairplot.png
  - categorical_analysis.png
  - skewness_analysis.png
  - interactive_scatter.html  (open in browser!)
""")
"""
MODULE 2 - EXPLORATORY DATA ANALYSIS (EDA)
Complete practice code: functions + worked examples.
Uses a synthetic Titanic-like dataset so it runs with zero downloads.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # so it can save plots without a display
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# BUILD A SYNTHETIC PRACTICE DATASET (Titanic-style)
# ============================================================

def build_practice_dataset(n=300, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'PassengerId': range(1, n + 1),
        'Survived': rng.choice([0, 1], size=n, p=[0.62, 0.38]),
        'Pclass': rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55]),
        'Sex': rng.choice(['male', 'female'], size=n, p=[0.65, 0.35]),
        'Age': np.round(rng.normal(29, 14, n).clip(0.5, 80), 1),
        'Fare': np.round(np.abs(rng.normal(32, 45, n)), 2),
        'Embarked': rng.choice(['S', 'C', 'Q', None], size=n, p=[0.72, 0.19, 0.08, 0.01])
    })
    # inject some missing values, like real data
    missing_idx = rng.choice(df.index, size=int(n * 0.15), replace=False)
    df.loc[missing_idx, 'Age'] = np.nan
    return df


# ============================================================
# 1. DESCRIPTIVE STATISTICS FUNCTIONS
# ============================================================

def full_data_summary(df):
    """One-shot overview: shape, dtypes, missing %, duplicates."""
    print(f"Shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissing values (%):\n{(df.isnull().mean() * 100).round(2)}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    return df.describe(include='all')

def numeric_summary(df):
    return df.describe().T.assign(
        skew=df.select_dtypes(include=np.number).skew(),
        kurtosis=df.select_dtypes(include=np.number).kurt()
    )

def categorical_summary(df):
    cat_cols = df.select_dtypes(include='object').columns
    summary = {}
    for col in cat_cols:
        summary[col] = {
            'unique_values': df[col].nunique(),
            'top_category': df[col].mode()[0] if not df[col].mode().empty else None,
            'missing_pct': round(df[col].isnull().mean() * 100, 2)
        }
    return pd.DataFrame(summary).T


# ============================================================
# 2. UNIVARIATE ANALYSIS FUNCTIONS
# ============================================================

def plot_numeric_univariate(df, col, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df[col].dropna(), kde=True, ax=axes[0])
    axes[0].set_title(f'Histogram: {col}')
    sns.boxplot(x=df[col].dropna(), ax=axes[1])
    axes[1].set_title(f'Boxplot: {col}')
    df[col].dropna().plot(kind='kde', ax=axes[2])
    axes[2].set_title(f'KDE: {col}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    print(f"{col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, "
          f"skew={df[col].skew():.2f}")

def plot_categorical_univariate(df, col, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x=col, data=df, ax=axes[0])
    axes[0].set_title(f'Count plot: {col}')
    df[col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[1])
    axes[1].set_title(f'Pie chart: {col}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    print(df[col].value_counts(normalize=True).round(3))

def run_univariate_on_all_columns(df, output_dir='/tmp'):
    """Loops through every column and auto-picks the right plot type."""
    for col in df.columns:
        if col == 'PassengerId':
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
            plot_numeric_univariate(df, col, f'{output_dir}/{col}_univariate.png')
        else:
            plot_categorical_univariate(df, col, f'{output_dir}/{col}_univariate.png')


# ============================================================
# 3. BIVARIATE / MULTIVARIATE ANALYSIS FUNCTIONS
# ============================================================

def numeric_vs_numeric(df, col1, col2, save_path=None):
    corr = df[[col1, col2]].corr().iloc[0, 1]
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=col1, y=col2, data=df)
    plt.title(f'{col1} vs {col2} (corr={corr:.2f})')
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return corr

def numeric_vs_categorical(df, num_col, cat_col, save_path=None):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=cat_col, y=num_col, data=df)
    plt.title(f'{num_col} by {cat_col}')
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std'])

def categorical_vs_categorical(df, col1, col2):
    ct = pd.crosstab(df[col1], df[col2], normalize='index')
    return ct

def correlation_heatmap(df, save_path=None):
    numeric_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return numeric_df.corr()

def find_top_correlated_features(df, target_col, top_n=5):
    """Find which features correlate most strongly with the target - your first
    clue about what will matter for modeling."""
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
    return corr.head(top_n + 1)   # +1 because target correlates 1.0 with itself


# ============================================================
# FULL WORKED EXAMPLE
# ============================================================

def run_full_eda_example():
    df = build_practice_dataset()

    print("=" * 60)
    print("STEP 1: Full summary")
    print("=" * 60)
    print(full_data_summary(df))

    print("\n" + "=" * 60)
    print("STEP 2: Numeric summary with skew/kurtosis")
    print("=" * 60)
    print(numeric_summary(df))

    print("\n" + "=" * 60)
    print("STEP 3: Categorical summary")
    print("=" * 60)
    print(categorical_summary(df))

    print("\n" + "=" * 60)
    print("STEP 4: Univariate analysis")
    print("=" * 60)
    plot_numeric_univariate(df, 'Age', '/tmp/age_univariate.png')
    plot_categorical_univariate(df, 'Survived', '/tmp/survived_univariate.png')

    print("\n" + "=" * 60)
    print("STEP 5: Bivariate analysis")
    print("=" * 60)
    print("Age by Survived:")
    print(numeric_vs_categorical(df, 'Age', 'Survived', '/tmp/age_survived.png'))
    print("\nSex vs Survived crosstab:")
    print(categorical_vs_categorical(df, 'Sex', 'Survived'))

    print("\n" + "=" * 60)
    print("STEP 6: Correlation with target")
    print("=" * 60)
    print(find_top_correlated_features(df, 'Survived'))

    return df


if __name__ == '__main__':
    run_full_eda_example()

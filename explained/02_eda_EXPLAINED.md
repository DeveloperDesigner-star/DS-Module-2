# LINE-BY-LINE EXPLANATION: 02_eda.py
### Zero-to-expert walkthrough of every statistics & plotting function.

---

## Imports

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
```
- `matplotlib` is Python's core plotting library. `matplotlib.pyplot` (nicknamed `plt`) is the part you actually draw charts with.
- `matplotlib.use('Agg')` — tells matplotlib to render charts to image FILES instead of trying to pop up an interactive window. This matters because this code runs on a server/container with no screen — without this line, plotting would crash with a "no display" error. On your own laptop with Jupyter, you normally don't need this line.
- `seaborn` (nicknamed `sns`) is built on top of matplotlib — it makes statistical charts (boxplots, heatmaps) with far less code and better default styling.

---

## Building a practice dataset

```python
def build_practice_dataset(n=300, seed=42):
    rng = np.random.default_rng(seed)
```
- `n=300` — how many rows to generate, default 300.
- `np.random.default_rng(seed)` — creates a random number generator. Giving it a fixed `seed` (42) means every time you run this, you get the *exact same* "random" numbers — this is called **reproducibility**, critical in data science so your results are consistent and debuggable.

```python
    df = pd.DataFrame({
        'PassengerId': range(1, n + 1),
        'Survived': rng.choice([0, 1], size=n, p=[0.62, 0.38]),
        'Pclass': rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55]),
        'Sex': rng.choice(['male', 'female'], size=n, p=[0.65, 0.35]),
        'Age': np.round(rng.normal(29, 14, n).clip(0.5, 80), 1),
        'Fare': np.round(np.abs(rng.normal(32, 45, n)), 2),
        'Embarked': rng.choice(['S', 'C', 'Q', None], size=n, p=[0.72, 0.19, 0.08, 0.01])
    })
```
- `rng.choice([0, 1], size=n, p=[0.62, 0.38])` — randomly picks `n` values from `[0, 1]`, where `p` gives the **probability** of each: 62% chance of 0, 38% chance of 1. This simulates realistic class imbalance (most passengers didn't survive).
- `rng.normal(29, 14, n)` — draws `n` numbers from a **normal (bell-curve) distribution** with mean 29 and standard deviation 14 — simulating realistic age spread.
- `.clip(0.5, 80)` — forces any value below 0.5 up to 0.5, and any value above 80 down to 80 — prevents nonsense like negative ages.
- `np.round(..., 1)` — rounds to 1 decimal place.
- `np.abs(rng.normal(32, 45, n))` — `np.abs` takes the absolute value, turning any negative numbers positive — used here because fares can't be negative, but a normal distribution can produce negative values.
- `Embarked` includes `None` as one possible outcome with 1% probability — simulating real missing categorical data.

```python
    missing_idx = rng.choice(df.index, size=int(n * 0.15), replace=False)
    df.loc[missing_idx, 'Age'] = np.nan
    return df
```
- `df.index` — the row numbers (0, 1, 2, ... 299).
- `rng.choice(df.index, size=int(n*0.15), replace=False)` — randomly picks 15% of row numbers, `replace=False` meaning no row is picked twice (sampling without replacement).
- `df.loc[missing_idx, 'Age'] = np.nan` — `.loc[]` selects rows/columns by label. This line goes to exactly those randomly chosen rows, in the `Age` column, and sets them to `np.nan` (numpy's "Not a Number" — pandas' standard missing-value marker). This deliberately injects realistic missing data so the rest of the code has something real to detect and handle.

---

## Section 1: Descriptive statistics functions

```python
def full_data_summary(df):
    print(f"Shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissing values (%):\n{(df.isnull().mean() * 100).round(2)}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    return df.describe(include='all')
```
- `df.shape` — returns `(rows, columns)` as a tuple, e.g. `(300, 7)`.
- `df.dtypes` — shows the data type of every column (int64, float64, object/string, etc.).
- `df.isnull()` — returns a same-shaped DataFrame of `True`/`False`, where `True` means that cell is missing.
- `.mean()` on a column of `True`/`False` treats `True` as 1 and `False` as 0 — so the mean IS the fraction of missing values. `* 100` converts that fraction to a percentage.
- `.round(2)` — rounds to 2 decimal places for readability.
- `df.duplicated()` — returns `True` for any row that's an exact duplicate of an earlier row; `.sum()` counts how many.
- `df.describe(include='all')` — normally `.describe()` only summarizes numeric columns; `include='all'` forces it to also summarize categorical columns (count, unique, top, freq).

```python
def numeric_summary(df):
    return df.describe().T.assign(
        skew=df.select_dtypes(include=np.number).skew(),
        kurtosis=df.select_dtypes(include=np.number).kurt()
    )
```
- `df.describe()` — by default only summarizes numeric columns: count, mean, std, min, 25%, 50%, 75%, max.
- `.T` — **transposes** the table (flips rows and columns), so each numeric column becomes a row instead — easier to read when you have many columns.
- `.assign(...)` — adds new columns to a DataFrame without modifying the original; here it adds `skew` and `kurtosis` columns.
- `df.select_dtypes(include=np.number)` — filters the DataFrame down to only numeric columns (drops text/categorical ones).
- `.skew()` — measures **asymmetry** of the distribution. 0 = symmetric, positive = long tail to the right, negative = long tail to the left.
- `.kurt()` — measures **"tailedness"** — how much of the data is in the extreme tails vs. the center, compared to a normal distribution.

```python
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
```
- `df.select_dtypes(include='object').columns` — gets just the names of text-type columns.
- `summary = {}` — an empty dictionary that will hold one entry per column.
- `for col in cat_cols:` — loop through each categorical column name.
- `df[col].nunique()` — counts how many **distinct** values that column has.
- `df[col].mode()` — returns the most frequently occurring value(s) as a small Series (there can be ties, hence it's a list-like object, not a single value).
- `df[col].mode()[0] if not df[col].mode().empty else None` — a ternary expression: "if mode() isn't empty, take the first element `[0]`; otherwise there's no data at all in this column, so use `None`." This guards against crashing on an all-missing column.
- `summary[col] = {...}` — stores a dictionary of stats under this column's name as the key — so `summary` ends up being a "dictionary of dictionaries."
- `pd.DataFrame(summary).T` — converts that nested dictionary into a table, then transposes so each original column becomes a row of the summary.

---

## Section 2: Univariate plotting functions

```python
def plot_numeric_univariate(df, col, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
```
- `plt.subplots(1, 3, figsize=(15, 4))` — creates a figure with **1 row, 3 columns** of side-by-side subplot "slots." `fig` is the whole canvas; `axes` is an array of the 3 individual plot areas. `figsize=(15, 4)` sets width=15 inches, height=4 inches.

```python
    sns.histplot(df[col].dropna(), kde=True, ax=axes[0])
    axes[0].set_title(f'Histogram: {col}')
```
- `df[col].dropna()` — drops missing values from this column before plotting (you can't plot `NaN`).
- `sns.histplot(..., kde=True, ax=axes[0])` — draws a histogram (bars showing how many values fall in each range) with a smoothed **KDE** (Kernel Density Estimate — a smooth curve overlay showing the distribution shape) on top. `ax=axes[0]` tells seaborn to draw into the FIRST of our 3 subplot slots specifically.
- `axes[0].set_title(...)` — labels that specific subplot.

```python
    sns.boxplot(x=df[col].dropna(), ax=axes[1])
    axes[1].set_title(f'Boxplot: {col}')
```
A boxplot shows the median (middle line), the interquartile range (the box, 25th–75th percentile), and outliers (individual dots beyond the "whiskers"). Drawn into the SECOND subplot slot.

```python
    df[col].dropna().plot(kind='kde', ax=axes[2])
    axes[2].set_title(f'KDE: {col}')
```
A standalone smooth density curve, in the THIRD slot — easier to compare shape/skew without histogram bar-width artifacts.

```python
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
```
- `plt.tight_layout()` — auto-adjusts spacing so subplot titles/labels don't overlap each other.
- `if save_path:` — only save to a file if a path was actually given (this function is optional/flexible).
- `plt.savefig(save_path)` — writes the current figure to disk as an image file.
- `plt.close()` — closes the figure to free up memory. Important in loops — if you don't close figures, running this 50 times in a row will silently eat more and more RAM.

```python
    print(f"{col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, "
          f"skew={df[col].skew():.2f}")
```
- `{df[col].mean():.2f}` — the `:.2f` inside an f-string is a **format specifier**: format this number as a float with exactly 2 decimal places.
- Two adjacent string literals (`"...(" "..."`) on separate lines get automatically concatenated by Python into one string — just a way to keep long print statements readable in the code.

```python
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
```
- `sns.countplot(x=col, data=df, ax=axes[0])` — a bar chart where the height of each bar = how many rows have that category value.
- `df[col].value_counts()` — counts occurrences of each unique value, sorted from most to least common — returns a Series.
- `.plot(kind='pie', autopct='%1.1f%%', ax=axes[1])` — draws that count Series as a pie chart. `autopct='%1.1f%%'` is a format string telling matplotlib to label each pie slice with its percentage to 1 decimal place (the `%%` produces a literal `%` symbol).
- `df[col].value_counts(normalize=True)` — same as before, but `normalize=True` converts raw counts into **proportions** (fractions that sum to 1) instead of counts — easier to read class balance at a glance.

```python
def run_univariate_on_all_columns(df, output_dir='/tmp'):
    for col in df.columns:
        if col == 'PassengerId':
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
            plot_numeric_univariate(df, col, f'{output_dir}/{col}_univariate.png')
        else:
            plot_categorical_univariate(df, col, f'{output_dir}/{col}_univariate.png')
```
- `for col in df.columns:` — loop over every column name in the DataFrame.
- `if col == 'PassengerId': continue` — `continue` skips the rest of THIS loop iteration and jumps to the next column. We skip PassengerId because it's just a row identifier, not real data worth plotting.
- `pd.api.types.is_numeric_dtype(df[col])` — checks if this column's data type is numeric (int/float).
- `df[col].nunique() > 10` — checks it has more than 10 distinct values. Combined with the numeric check: this distinguishes a *truly continuous* numeric column (like Age, with many unique values) from a numeric column that's *actually categorical* in disguise (like Pclass, which is only ever 1, 2, or 3 — even though it's stored as a number, it behaves like a category).
- This is a real pattern professionals use: don't just check dtype, check *how the data actually behaves*.

---

## Section 3: Bivariate/multivariate functions

```python
def numeric_vs_numeric(df, col1, col2, save_path=None):
    corr = df[[col1, col2]].corr().iloc[0, 1]
```
- `df[[col1, col2]]` — note the double brackets: `df[col1]` (single brackets) gives you a Series (1 column), but `df[[col1, col2]]` (a list inside brackets) gives you a DataFrame (subset with 2 columns).
- `.corr()` — computes the **correlation matrix** between all numeric columns in this subset — a 2x2 table here, since we only kept 2 columns. Correlation ranges from -1 (perfect negative relationship) to +1 (perfect positive relationship), 0 = no linear relationship.
- `.iloc[0, 1]` — `.iloc` selects by integer POSITION (row 0, column 1) rather than by label. In a 2x2 correlation matrix, position [0,1] is exactly the correlation between col1 and col2 (the diagonal [0,0] and [1,1] would just be 1.0, since anything perfectly correlates with itself).

```python
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=col1, y=col2, data=df)
    plt.title(f'{col1} vs {col2} (corr={corr:.2f})')
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return corr
```
A scatter plot: each dot is one row, positioned by its col1 and col2 values — the classic way to visually spot a relationship (or lack of one) between two numeric variables.

```python
def numeric_vs_categorical(df, num_col, cat_col, save_path=None):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=cat_col, y=num_col, data=df)
    plt.title(f'{num_col} by {cat_col}')
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std'])
```
- `sns.boxplot(x=cat_col, y=num_col, data=df)` — draws one separate boxplot for each category, side by side — e.g. one Age boxplot for "Survived=0" and another for "Survived=1", letting you visually compare.
- `df.groupby(cat_col)` — splits the DataFrame into separate groups, one per unique value of `cat_col` (this is the "split" step of the classic "split-apply-combine" pattern in pandas).
- `[num_col]` — within each group, select just the numeric column we care about.
- `.agg(['mean', 'median', 'std'])` — "apply" multiple aggregation functions at once to each group, then "combine" results into one table — so you get exact numbers backing up what the boxplot shows visually.

```python
def categorical_vs_categorical(df, col1, col2):
    ct = pd.crosstab(df[col1], df[col2], normalize='index')
    return ct
```
- `pd.crosstab(df[col1], df[col2])` — builds a cross-tabulation (contingency table): rows = unique values of col1, columns = unique values of col2, cells = how many rows have that combination.
- `normalize='index'` — converts each ROW into percentages that sum to 100% across that row — e.g. "of all males, what % survived vs didn't" rather than raw counts, which is usually the more useful comparison.

```python
def correlation_heatmap(df, save_path=None):
    numeric_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return numeric_df.corr()
```
- `numeric_df.corr()` — full correlation matrix across ALL numeric column pairs at once.
- `sns.heatmap(..., annot=True, cmap='coolwarm', fmt='.2f')` — draws that matrix as a color-coded grid. `annot=True` prints the actual number inside each cell. `cmap='coolwarm'` is the color scheme (blue = negative correlation, red = positive). `fmt='.2f'` formats the printed numbers to 2 decimals.

```python
def find_top_correlated_features(df, target_col, top_n=5):
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
    return corr.head(top_n + 1)
```
- `numeric_df.corr()[target_col]` — from the full correlation matrix, pull out just the ONE column showing every feature's correlation with your target variable.
- `.abs()` — takes absolute value, because a strong NEGATIVE correlation (-0.8) is just as useful/predictive as a strong positive one (+0.8) — we care about magnitude, not direction, when ranking importance.
- `.sort_values(ascending=False)` — sorts from highest to lowest.
- `.head(top_n + 1)` — takes the top N+1 rows. The `+1` accounts for the fact that the target will always appear correlated with itself at exactly 1.0 (position 1) — so asking for `top_n=5` really means "5 real features plus the self-correlation row."

---

## The full worked example function

```python
def run_full_eda_example():
    df = build_practice_dataset()
    print("=" * 60)
```
- `"=" * 60` — string multiplication: repeats `"="` sixty times to make a visual divider line in the printed output. This is a common, simple trick to make console output more readable.

The rest of `run_full_eda_example()` just calls every function defined above, in the logical EDA order (summary → univariate → bivariate → correlation), printing labeled section headers between each — this is your **template for doing EDA on any new dataset**: swap in your real data and re-run the same sequence of function calls.

---

## Concepts you now know (recap, building on file 1)
- Random number generation with a fixed seed for reproducibility
- `.loc[]` vs `.iloc[]` (label-based vs position-based selection)
- Single `[]` vs double `[[]]` bracket indexing (Series vs DataFrame)
- `groupby()` — the "split-apply-combine" pattern
- Correlation, skewness, kurtosis (statistical concepts)
- `matplotlib`/`seaborn` figure/subplot mechanics, saving vs. closing figures
- String formatting specifiers (`:.2f`), string multiplication tricks
- `continue` in loops
- Checking *behavior* of data (nunique) rather than just its stored dtype

**Next: I'll continue with Module 3 — Feature Engineering — delivering the full practice code file first, then its own line-by-line EXPLAINED.md, same as these two. Reply "next" whenever you're ready, and I'll also refresh your repo zip to include everything so far.**

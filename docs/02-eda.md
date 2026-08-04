# Module 2 — Exploratory Data Analysis (EDA)

## 2.1 Understanding your data: descriptive statistics
**Theory:** Before touching a model, know your data's shape, types, and distribution. Descriptive stats summarize the data with a handful of numbers: central tendency (mean, median, mode), spread (std, variance, range, IQR), and shape (skewness, kurtosis).

```python
df.shape
df.info()
df.describe()               # numeric summary: count, mean, std, min, quartiles, max
df.describe(include='object')  # categorical summary
df.isnull().sum()
df.duplicated().sum()
df['col'].value_counts()
df['col'].skew()
df['col'].kurt()
df.nunique()
```

| Stat | Meaning | Robust to outliers? |
|---|---|---|
| Mean | Average | No |
| Median | Middle value | Yes |
| Mode | Most frequent value | Yes |
| Std / Variance | Spread around mean | No |
| IQR | Spread of middle 50% | Yes |

## 2.2 Univariate analysis
**Theory:** Analyzing **one column at a time**. Numerical → histograms, boxplots, KDE plots (check distribution shape, skew, outliers). Categorical → count plots, pie charts (check class balance).

```python
import matplotlib.pyplot as plt
import seaborn as sns

# numerical
sns.histplot(df['age'], kde=True)
sns.boxplot(x=df['age'])
df['age'].plot(kind='kde')

# categorical
sns.countplot(x='category', data=df)
df['category'].value_counts().plot(kind='pie', autopct='%1.1f%%')
```

## 2.3 Bivariate & multivariate analysis
**Theory:** Analyzing **relationships between two or more columns** — this is where you find predictive signal.

| Combo | Best plot |
|---|---|
| Numerical vs Numerical | Scatter plot, `sns.scatterplot`, correlation heatmap |
| Numerical vs Categorical | Bar plot, box plot, violin plot |
| Categorical vs Categorical | Cross-tab (`pd.crosstab`), stacked bar, heatmap |

```python
sns.scatterplot(x='age', y='salary', hue='gender', data=df)
sns.barplot(x='category', y='sales', data=df)
sns.boxplot(x='category', y='age', data=df)
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
pd.crosstab(df['gender'], df['purchased'])
sns.pairplot(df, hue='target')      # all pairwise relationships at once
```

## 2.4 Automated EDA (pandas-profiling / ydata-profiling)
**Theory:** Automates steps 2.1–2.3 into one interactive HTML report — great for a first-pass overview, but you should still do manual EDA for real insight (auto-reports don't understand business context).

```python
from ydata_profiling import ProfileReport   # renamed from pandas-profiling

profile = ProfileReport(df, title="EDA Report", explorative=True)
profile.to_file("report.html")
```

**Practice task:** Take the Titanic dataset. Do full univariate analysis on every column, then find the top 3 bivariate relationships that look predictive of `Survived`.

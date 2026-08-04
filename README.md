# My Data Science Learning Journey 🚀

Personal notes, practice code, and study guides as I learn data science from scratch —
one topic at a time, with runnable code and my own line-by-line explanations written while learning.

![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Why this repo exists

I'm building a complete, practical data science foundation — not just watching tutorials, but
actually writing, running, breaking, and re-explaining every concept in my own words so it
actually sticks. This repo is my public record of that process, updated module by module.

## 📁 Repository Structure

```
my-data-science-journey/
├── docs/          # Theory guides — one markdown file per topic/module
├── code/          # Runnable practice .py scripts with functions + worked examples
├── explained/     # My own line-by-line breakdowns of the code, written for revision
├── notebooks/     # Jupyter notebook experiments
├── datasets/      # Small practice datasets (large files are git-ignored)
├── requirements.txt
└── README.md
```

## 📚 Progress Tracker

| # | Module | Theory | Code | My Notes | Status |
|---|--------|:---:|:---:|:---:|:---:|
| 0 | Roadmap Overview | [📄](docs/00-roadmap.md) | — | — | ✅ |
| 1 | Data Acquisition | [📄](docs/01-data-acquisition.md) | [🐍](code/01_data_acquisition.py) | [📝](explained/01_data_acquisition_EXPLAINED.md) | ✅ |
| 2 | Exploratory Data Analysis | [📄](docs/02-eda.md) | [🐍](code/02_eda.py) | [📝](explained/02_eda_EXPLAINED.md) | ✅ |
| 3 | Feature Engineering | — | — | — | 🚧 |
| 4 | Missing Data | — | — | — | 🚧 |
| 5 | Outliers | — | — | — | 🚧 |
| 6 | Feature Construction & PCA | — | — | — | 🚧 |
| 7 | Regression | — | — | — | 🚧 |
| 8 | Classification | — | — | — | 🚧 |
| 9 | Ensemble Learning | — | — | — | 🚧 |
| 10 | Clustering | — | — | — | 🚧 |

✅ Complete &nbsp;&nbsp; 🚧 Coming soon — updated as I complete each module

## 📘 What's in each completed module

**Module 1 — Data Acquisition:** CSV (custom separators, chunking, dtype control, date parsing),
JSON flattening, SQL querying, API fetching with error handling, web scraping.

**Module 2 — Exploratory Data Analysis:** descriptive statistics, univariate analysis
(histograms, boxplots, KDE), bivariate/multivariate analysis (scatter plots, correlation
heatmaps, crosstabs), and finding which features correlate most with a target variable.

## 🛠️ How to run this locally

```bash
git clone https://github.com/YOUR-USERNAME/my-data-science-journey.git
cd my-data-science-journey
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

python code/01_data_acquisition.py
python code/02_eda.py
```

## 🧠 How each module is organized

Every topic follows the same three-layer format:
1. **Theory** (`docs/`) — concepts, formulas, decision rules for when to use what
2. **Practice code** (`code/`) — tested, working functions with real worked examples
3. **My notes** (`explained/`) — a full line-by-line breakdown I wrote while learning, for revision

## 📄 License
MIT — see [LICENSE](LICENSE)

---
*Last updated: Module 2 complete. Next up: Feature Engineering.*

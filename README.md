# Amazon India Product Analysis

An exploratory data analysis of **1,400+ Amazon India products**, examining pricing strategies, discount patterns, customer ratings, and category trends. Built with Python, pandas, and SQL.

---

## Project Overview

Online marketplaces use complex pricing and discounting strategies. This project analyzes Amazon India's product listings to answer key business questions:

- **Do bigger discounts lead to better ratings?**
- **Which categories offer the deepest discounts — and why?**
- **What separates the most-reviewed products from the rest?**
- **Where are the "best value" products hiding?**

## Key Findings

| Insight | Detail |
|---------|--------|
| **Average discount** | ~50% across all listed products |
| **Discount ↔ Rating correlation** | Near zero — heavy discounts don't buy better reviews |
| **Most-reviewed category** | Electronics dominates in total review volume |
| **Price distribution** | Products skew toward the budget segment (<₹500) |
| **Best value sweet spot** | Products with 50%+ discount AND 4★+ rating exist across multiple categories |

## Dashboard Preview

> *Run the script to generate `amazon_dashboard.png` — a 2×2 grid covering:*

| Top-left | Top-right |
|----------|-----------|
| Category discount rankings (bar chart) | Discount % vs. Rating (scatter, colored by review count) |

| Bottom-left | Bottom-right |
|-------------|--------------|
| Rating distribution with mean line | Product count & avg rating by price tier |

## Tools & Technologies

- **Python 3** — core language
- **pandas** — data cleaning, manipulation, and aggregation
- **NumPy** — numerical operations
- **Matplotlib & Seaborn** — data visualization
- **SQLite** — SQL querying on the dataset (including window functions)

## Project Structure

```
amazon-product-analysis/
│
├── amazon_analysis.py      # Main analysis script (full pipeline)
├── amazon.csv              # Raw dataset
├── amazon_dashboard.png    # Generated 2×2 chart dashboard
├── correlation_heatmap.png # Feature correlation matrix
└── README.md               # This file
```

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/amazon-product-analysis.git
   cd amazon-product-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```

3. **Run the analysis**
   ```bash
   python amazon_analysis.py
   ```

   This will print the full EDA and SQL results to the console and save two chart images to the project folder.

## Analysis Pipeline

The script follows a structured data science workflow:

```
Load CSV → Clean Data → Engineer Features → EDA → Visualize → SQL Analysis → Insights
```

**Step-by-step:**

1. **Data Loading** — Read the raw Amazon product CSV
2. **Data Cleaning** — Convert currency strings to numbers, handle missing values, remove duplicates
3. **Feature Engineering** — Create `calculated_discount_pct`, `savings_amount`, `price_tier`, and `main_category`
4. **Exploratory Analysis** — Descriptive statistics, correlation analysis, category-level aggregations
5. **Visualization** — 2×2 dashboard and correlation heatmap saved as PNG
6. **SQL Queries** — Category popularity, best-value filtering, and a window function ranking products within categories
7. **Insights Summary** — Plain-language takeaways from the data

## SQL Highlights

The project demonstrates SQL proficiency through three analytical queries:

- **Aggregation** — `GROUP BY` with `AVG`, `SUM`, `COUNT` to rank categories
- **Multi-condition filtering** — `WHERE` clause combining discount %, rating, and review thresholds
- **Window functions** — `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)` to find the top-rated product in each category

## Dataset

The dataset contains Amazon India product listings with fields including product name, category, pricing (actual and discounted), discount percentage, customer rating, and rating count. Source: [Kaggle Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset).

## Contact

Emir Celil Yurdaer — [celilyurdaer@protonmail.com](mailto:celilyurdaer@protonmail.com)

Feel free to open an issue or reach out with questions!

---

*Built as a portfolio project demonstrating data analysis, visualization, and SQL skills.*

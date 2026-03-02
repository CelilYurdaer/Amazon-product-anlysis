"""
Amazon Product Analysis — Entry-Level Data Science Portfolio Project
====================================================================
This project demonstrates a complete data analysis workflow:

    1. DATA LOADING        → Read raw CSV data
    2. DATA CLEANING       → Fix types, handle missing values, remove duplicates
    3. FEATURE ENGINEERING  → Create new columns from existing data
    4. EDA (Exploratory)   → Descriptive stats, distributions, correlations
    5. VISUALIZATION        → Publication-quality charts
    6. SQL QUERYING         → Demonstrate SQL skills via sqlite3
    7. INSIGHTS & SUMMARY   → Key takeaways printed at the end

"""

# ─────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────
# Each library has a specific role:
#   • pandas   — tabular data manipulation (the backbone of any analysis)
#   • numpy    — numerical operations (used under the hood by pandas too)
#   • matplotlib/seaborn — visualization (seaborn wraps matplotlib with
#                          better defaults and statistical plot types)
#   • sqlite3  — lightweight SQL database (ships with Python, no install needed)
#   • warnings — suppress noisy deprecation messages during development

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import sqlite3

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


# ─────────────────────────────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────────────────────────────
# CONNECTION: Everything starts here. We load the raw CSV, then every
# subsequent section transforms or queries this single DataFrame.
# Using a function makes the step reusable and testable.

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and print a quick sanity check."""
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {filepath}")
    print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}\n")
    return df


# ─────────────────────────────────────────────────────────────────────
# 3. DATA CLEANING
# ─────────────────────────────────────────────────────────────────────
# CONNECTION: Cleaning feeds directly into Feature Engineering and EDA.
# If we skip this, aggregations and plots will silently break or lie
# (e.g., "₹1,299" is a string — you can't compute its mean).


def clean_currency(series: pd.Series) -> pd.Series:
    """Strip ₹ and commas, then convert to float."""
    return (
        series
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean types, drop duplicates, and report missing values."""

    df = df.copy()  

    # --- Currency columns ---------------------------------------------------
    df["discounted_price"] = clean_currency(df["discounted_price"])
    df["actual_price"]     = clean_currency(df["actual_price"])

    # --- Numeric columns ----------------------------------------------------
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_count"] = (
        df["rating_count"]
        .str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")   # .pipe() keeps the chain clean
    )

    # --- Discount percentage (originally missing from cleaned columns) ------
    if df["discount_percentage"].dtype == object:
        df["discount_percentage"] = (
            df["discount_percentage"]
            .str.replace("%", "", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )

    # --- Duplicates ----------------------------------------------------------
    n_dupes = df.duplicated(subset=["product_id"]).sum()
    if n_dupes:
        df = df.drop_duplicates(subset=["product_id"], keep="first")
        print(f"  Removed {n_dupes} duplicate product(s)")

    # --- Missing values report -----------------------------------------------
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("  Missing values per column:")
        for col, count in missing.items():
            pct = count / len(df) * 100
            print(f"    {col:30s} → {count:>5} ({pct:.1f}%)")
    else:
        print("  No missing values ✓")

    print(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    return df


# ─────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────
# CONNECTION: New columns give us richer questions to explore in EDA.
# These features also translate directly into model inputs if you later
# add a predictive component (e.g., "predict rating from price features").

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived columns that add analytical value."""

    df = df.copy()

    df["calculated_discount_pct"] = (
        (df["actual_price"] - df["discounted_price"])
        / df["actual_price"] * 100
    )

    # Absolute savings in ₹
    df["savings_amount"] = df["actual_price"] - df["discounted_price"]

    # Price tier — useful for segmented analysis
    df["price_tier"] = pd.cut(
        df["discounted_price"],
        bins=[0, 500, 1500, 5000, np.inf],
        labels=["Budget (<₹500)", "Mid (₹500-1.5k)",
                "Premium (₹1.5k-5k)", "Luxury (>₹5k)"]
    )

    # Top-level category (first segment before '|')
    df["main_category"] = df["category"].str.split("|").str[0].str.strip()

    print(f"✓ Engineered 4 new features: calculated_discount_pct, "
          f"savings_amount, price_tier, main_category\n")
    return df


# ─────────────────────────────────────────────────────────────────────
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────
# CONNECTION: EDA is where you turn cleaned data into understanding.

def run_eda(df: pd.DataFrame) -> dict:
    """Print key statistics and return a dict of summary results."""

    insights = {}

    # --- Descriptive stats ---------------------------------------------------
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    summary_cols = ["discounted_price", "actual_price", "rating",
                    "rating_count", "calculated_discount_pct"]
    print(df[summary_cols].describe().round(2).to_string())

    # --- Category-level aggregation -----------------------------------------
    cat_stats = (
        df.groupby("main_category")
        .agg(
            product_count=("product_id", "count"),
            avg_price=("discounted_price", "mean"),
            avg_rating=("rating", "mean"),
            avg_discount_pct=("calculated_discount_pct", "mean"),
            total_reviews=("rating_count", "sum"),
        )
        .sort_values("total_reviews", ascending=False)
    )
    insights["category_stats"] = cat_stats
    print("\n\nCATEGORY SUMMARY (by total reviews):")
    print(cat_stats.round(2).to_string())

    # --- Correlation matrix --------------------------------------------------
    corr_cols = ["discounted_price", "actual_price", "rating",
                 "rating_count", "calculated_discount_pct"]
    corr_matrix = df[corr_cols].corr().round(3)
    insights["correlation"] = corr_matrix
    print("\n\nCORRELATION MATRIX:")
    print(corr_matrix.to_string())

    # --- Top products -------------------------------------------------------
    top_products = (
        df.nlargest(10, "rating_count")[
            ["product_name", "main_category", "rating",
             "rating_count", "discounted_price", "calculated_discount_pct"]
        ]
    )
    insights["top_products"] = top_products
    print("\n\nTOP 10 MOST-REVIEWED PRODUCTS:")
    print(top_products.to_string(index=False))

    print()
    return insights


# ─────────────────────────────────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────
# CONNECTION: Charts are the "show, don't tell" layer.  Each one maps
# directly to an insight from the EDA section above.


PALETTE = sns.color_palette("viridis", 10)


def plot_dashboard(df: pd.DataFrame, insights: dict):
    """Generate a 2×2 dashboard of key visualizations."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle("Amazon Product Analysis — Key Insights",
                 fontsize=18, fontweight="bold", y=1.01)

    # ---- Plot 1: Top categories by average discount -------------------------
    ax = axes[0, 0]
    cat_disc = (
        df.groupby("main_category")["calculated_discount_pct"]
        .mean()
        .sort_values(ascending=True)
        .tail(10)
    )
    cat_disc.plot(kind="barh", ax=ax, color=PALETTE, edgecolor="white")
    ax.set_title("Top 10 Categories by Avg Discount %", fontweight="bold")
    ax.set_xlabel("Average Discount (%)")
    ax.set_ylabel("")
    # Annotate bars with values
    for i, (val, name) in enumerate(zip(cat_disc.values, cat_disc.index)):
        ax.text(val + 0.5, i, f"{val:.1f}%", va="center", fontsize=9)

    # ---- Plot 2: Discount % vs Rating (scatter) ----------------------------
    ax = axes[0, 1]
    scatter = ax.scatter(
        df["calculated_discount_pct"], df["rating"],
        alpha=0.25, s=15, c=df["rating_count"],
        cmap="viridis", edgecolors="none"
    )
    plt.colorbar(scatter, ax=ax, label="Rating Count", shrink=0.8)
    ax.set_title("Discount % vs Rating", fontweight="bold")
    ax.set_xlabel("Discount (%)")
    ax.set_ylabel("Rating")
    # Add correlation annotation
    corr_val = df[["calculated_discount_pct", "rating"]].corr().iloc[0, 1]
    ax.annotate(f"r = {corr_val:.3f}", xy=(0.05, 0.95),
                xycoords="axes fraction", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ---- Plot 3: Rating distribution ----------------------------------------
    ax = axes[1, 0]
    sns.histplot(df["rating"].dropna(), bins=20, kde=True, ax=ax,
                 color=PALETTE[3], edgecolor="white")
    ax.axvline(df["rating"].mean(), color="red", ls="--", lw=1.5,
               label=f'Mean = {df["rating"].mean():.2f}')
    ax.set_title("Rating Distribution", fontweight="bold")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.legend()

    # ---- Plot 4: Price tier breakdown ---------------------------------------
    ax = axes[1, 1]
    tier_counts = df["price_tier"].value_counts().sort_index()
    tier_ratings = df.groupby("price_tier")["rating"].mean()
    colors = [PALETTE[i] for i in [0, 3, 6, 9]]

    bars = ax.bar(range(len(tier_counts)), tier_counts.values,
                  color=colors, edgecolor="white", label="Product Count")
    ax.set_xticks(range(len(tier_counts)))
    ax.set_xticklabels(tier_counts.index, rotation=15, ha="right")
    ax.set_title("Products & Avg Rating by Price Tier", fontweight="bold")
    ax.set_ylabel("Number of Products")

    # Overlay avg rating as text on bars
    for i, (bar, rating) in enumerate(zip(bars, tier_ratings.values)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"★ {rating:.2f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig("amazon_dashboard.png", dpi=150)
    print("✓ Dashboard saved → amazon_dashboard.png")
    plt.show()


def plot_correlation_heatmap(insights: dict):
    """Dedicated heatmap for the correlation matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        insights["correlation"], annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, square=True, ax=ax,
        linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig("correlation_heatmap.png", dpi=150)
    print("✓ Heatmap saved → correlation_heatmap.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# 7. SQL QUERIES
# ─────────────────────────────────────────────────────────────────────
# CONNECTION: Many data science roles require SQL, as it is essential for querying, 
# managing, and analyzing data stored in relational databases.

def run_sql_analysis(df: pd.DataFrame):
    """Load data into SQLite, run analytical queries, and return results."""

    conn = sqlite3.connect(":memory:")  # in-memory DB is faster & cleaner
    df.to_sql("products", conn, if_exists="replace", index=False)
    print("=" * 60)
    print("SQL ANALYSIS")
    print("=" * 60)

    queries = {
        # Query 1 — Category popularity 
        "Most Popular Categories (by total reviews)": """
            SELECT
                main_category                    AS category,
                COUNT(*)                         AS product_count,
                ROUND(AVG(rating), 2)            AS avg_rating,
                SUM(rating_count)                AS total_reviews
            FROM products
            WHERE rating IS NOT NULL
            GROUP BY main_category
            ORDER BY total_reviews DESC
            LIMIT 10;
        """,

        # Query 2 — Best value products (high discount + high rating)
        "Best Value Products (discount ≥ 50% AND rating ≥ 4.0)": """
            SELECT
                product_name,
                main_category,
                discounted_price,
                calculated_discount_pct  AS discount_pct,
                rating,
                rating_count
            FROM products
            WHERE calculated_discount_pct >= 50
              AND rating >= 4.0
              AND rating_count >= 100
            ORDER BY rating_count DESC
            LIMIT 10;
        """,

        # Query 3 — Window function: rank products within each category
        "Top-Rated Product per Category (window function)": """
            SELECT category, product_name, rating, rating_count
            FROM (
                SELECT
                    main_category AS category,
                    product_name,
                    rating,
                    rating_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY main_category
                        ORDER BY rating DESC, rating_count DESC
                    ) AS rn
                FROM products
                WHERE rating IS NOT NULL
            )
            WHERE rn = 1
            ORDER BY rating DESC;
        """,
    }

    results = {}
    for title, sql in queries.items():
        result = pd.read_sql(sql, conn)
        results[title] = result
        print(f"\n📊 {title}")
        print("-" * len(title))
        print(result.to_string(index=False))
        print()

    conn.close()
    return results


# ─────────────────────────────────────────────────────────────────────
# 8. INSIGHTS SUMMARY
# ─────────────────────────────────────────────────────────────────────

def print_insights(df: pd.DataFrame, insights: dict):
    """Summarise the key takeaways in plain language."""

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)

    avg_discount = df["calculated_discount_pct"].mean()
    avg_price    = df["discounted_price"].mean()
    avg_rating   = df["rating"].mean()
    corr_disc_rt = df[["calculated_discount_pct", "rating"]].corr().iloc[0, 1]

    insights_text = [
        f"1. The average discount across all products is {avg_discount:.1f}%, "
        f"with an average selling price of ₹{avg_price:,.0f}.",

        f"2. The average product rating is {avg_rating:.2f}/5, suggesting "
        f"generally positive customer satisfaction.",

        f"3. Correlation between discount % and rating is {corr_disc_rt:.3f} — "
        f"{'a weak relationship, meaning bigger discounts do NOT guarantee better ratings.' if abs(corr_disc_rt) < 0.3 else 'a moderate relationship worth investigating further.'}",

        f"4. The most-reviewed products are concentrated in "
        f"{insights['top_products']['main_category'].mode().iloc[0]}, "
        f"indicating high consumer engagement in that category.",

        f"5. {df['price_tier'].mode().iloc[0]} is the most common price tier, "
        f"suggesting Amazon India's listed products skew toward that segment.",
    ]

    for line in insights_text:
        print(f"\n  {line}")
    print()


# ─────────────────────────────────────────────────────────────────────
# 9. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────
# Everything above is a self-contained function. This main block wires
# them together in the correct order: Load → Clean → Engineer → EDA →
# Visualize → SQL → Summarize. Running the script end-to-end
# reproduces the full analysis.

if __name__ == "__main__":
    # --- Pipeline execution ---
    raw_df  = load_data("amazon.csv")
    clean_df = clean_data(raw_df)
    final_df = engineer_features(clean_df)
    eda_insights = run_eda(final_df)

    # --- Visualizations ---
    plot_dashboard(final_df, eda_insights)
    plot_correlation_heatmap(eda_insights)

    # --- SQL ---
    sql_results = run_sql_analysis(final_df)

    # --- Final summary ---
    print_insights(final_df, eda_insights)

    print("✅ Analysis complete. Outputs: amazon_dashboard.png, "
          "correlation_heatmap.png")

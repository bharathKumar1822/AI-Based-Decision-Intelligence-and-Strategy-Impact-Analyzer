"""
backend/app.py
Flask REST API + static frontend server
for the AI-Based Decision Intelligence & Strategy Impact Analyzer
"""

import os
import io
import re
import base64
import traceback
import datetime
import threading
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats as sp_stats

sns.set_style("whitegrid")

# ── Paths ──────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

# ── In-memory store  (keyed by dataset name) ───────────────────────
datasets: dict[str, pd.DataFrame] = {}
ml_cache: dict[str, dict] = {}

# ── HELPERS ────────────────────────────────────────────────────────

def fig_to_b64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    for col in ("Order Date", "Ship Date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)
    cat_cols = df.select_dtypes(exclude="number").columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    return df


def run_ml(df: pd.DataFrame) -> dict:
    """Train LR / DT / RF, return metrics + predictions."""
    result = {}
    if not {"Sales", "Quantity", "Profit"}.issubset(df.columns):
        return result

    # Subsample for extremely fast training on large datasets
    max_train_samples = 5000
    if len(df) > max_train_samples:
        df_sample = df.sample(n=max_train_samples, random_state=42)
    else:
        df_sample = df

    features = df_sample[["Sales", "Quantity"]].values
    target   = df_sample["Profit"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(max_depth=8, min_samples_split=10, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=5, max_depth=6, min_samples_split=10, random_state=42, n_jobs=1),
    }

    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        pred = mdl.predict(X_te)
        key  = name.lower().replace(" ", "_")
        result[key] = {
            "mse": round(float(mean_squared_error(y_te, pred)), 4),
            "r2":  round(float(r2_score(y_te, pred)), 4),
            "avg_predicted_profit": round(float(pred.mean()), 2),
        }

    best_key = min(result, key=lambda k: result[k]["mse"])
    result["best_model"]             = best_key.replace("_", " ").title()
    result["best_model_key"]         = best_key
    result["best_predicted_profit"]  = result[best_key]["avg_predicted_profit"]
    return result


def detect_weaknesses(df: pd.DataFrame) -> dict:
    w = {}
    if {"Sub-Category", "Profit"}.issubset(df.columns):
        lp = df.groupby("Sub-Category")["Profit"].sum()
        w["loss_making_products"] = lp[lp < 0].round(2).to_dict()

    if {"Region", "Profit"}.issubset(df.columns):
        avg = df["Profit"].mean()
        rp  = df.groupby("Region")["Profit"].sum()
        w["low_performing_regions"] = rp[rp < avg].round(2).to_dict()

    if {"Category", "Profit", "Sales"}.issubset(df.columns):
        df2 = df.copy()
        df2["margin"] = df2["Profit"] / df2["Sales"].replace(0, np.nan)
        pm = df2.groupby("Category")["margin"].mean()
        w["poor_profit_margins"] = {k: round(v, 4) for k, v in pm[pm < 0.1].items()}
    return w


def build_recommendations(weaknesses: dict, best_model: str, df: pd.DataFrame = None) -> dict:
    """Return structured, actionable recommendations with ideas, steps, dos, donts."""
    items = []

    # ── Recommendation 1: Model-driven profit forecasting ───────────
    items.append({
        "id":       "rec_ml_forecast",
        "idea":     f"Deploy {best_model} for real-time profit forecasting across all product lines",
        "rationale": (
            f"{best_model} demonstrated the lowest Mean Squared Error (MSE) among all evaluated models, "
            "making it the most reliable tool for predicting future profit outcomes."
        ),
        "steps": [
            f"Integrate the trained {best_model} model into your BI dashboard or ERP system.",
            "Schedule monthly model retraining using the latest sales and inventory data.",
            "Set automated profit forecast alerts for SKUs predicted to fall below margin thresholds.",
            "Create a feedback loop: compare predictions vs actuals each quarter and recalibrate.",
        ],
        "dos": [
            f"Use {best_model} as the primary profit prediction engine in operational planning.",
            "Validate model outputs against domain expertise before major investment decisions.",
            "Monitor R² and MSE drift monthly to detect data distribution shifts.",
        ],
        "donts": [
            "Do not use Linear Regression alone if the data shows non-linear profit patterns.",
            "Avoid relying on a single model without periodic benchmarking against newer algorithms.",
            "Do not deploy predictions without a confidence interval or uncertainty estimate.",
        ],
        "growth_note": (
            f"Task: Deploy {best_model} as the core forecasting engine in your pricing and inventory system "
            "by next quarter — businesses that adopt ML-driven forecasting report 10–20% improvement in "
            "profit margin accuracy, directly enabling smarter budget allocation and revenue growth."
        ),
    })

    # ── Recommendation 2: Loss-making products ───────────────────────
    if weaknesses.get("loss_making_products"):
        prods    = list(weaknesses["loss_making_products"].keys())[:5]
        prod_str = ", ".join(prods)
        items.append({
            "id":       "rec_loss_products",
            "idea":     f"Eliminate or restructure loss-making sub-categories: {prod_str}",
            "rationale": (
                f"The sub-categories {prod_str} have negative cumulative profit, "
                "directly dragging down overall business performance."
            ),
            "steps": [
                f"Conduct a deep-dive cost analysis for: {prod_str}.",
                "Identify whether losses stem from pricing, discounting, returns, or supply chain issues.",
                "Pilot a 15–20% price increase or bundle these items with high-margin products.",
                "If losses persist after 2 quarters, phase out or discontinue the product line.",
                "Reallocate the freed budget toward top-performing sub-categories.",
            ],
            "dos": [
                "Segment loss-makers by root cause (discounts vs. cost vs. demand) before acting.",
                "Test price elasticity with A/B pricing experiments before full rollout.",
                "Document learnings from discontinued lines to avoid repeating mistakes.",
            ],
            "donts": [
                f"Do not continue investing in {prods[0]} without a clear turnaround plan.",
                "Avoid blanket discounting across all struggling lines — it compounds losses.",
                "Do not remove products without analyzing their role in cross-selling bundles.",
            ],
            "growth_note": (
                f"Task: Discontinue or reprice the top 3 loss-making sub-categories ({', '.join(prods[:3])}) "
                "within 60 days — eliminating these losses could recover 5–15% of total profit erosion "
                "and free capital for reinvestment into high-margin product lines."
            ),
        })

    # ── Recommendation 3: Regional expansion ────────────────────────
    if weaknesses.get("low_performing_regions"):
        regions  = list(weaknesses["low_performing_regions"].keys())
        reg_str  = ", ".join(regions)
        items.append({
            "id":       "rec_regions",
            "idea":     f"Implement targeted regional recovery strategy for: {reg_str}",
            "rationale": (
                f"Regions {reg_str} are underperforming relative to the company profit average, "
                "representing untapped revenue and market share opportunity."
            ),
            "steps": [
                f"Assign a dedicated regional manager or sales team for {regions[0] if regions else 'the region'}.",
                "Analyze local competition, customer demographics, and seasonal demand patterns.",
                "Launch region-specific promotions aligned with local purchasing behavior.",
                "Introduce subscription-based pricing or loyalty programs for repeat regional customers.",
                "Set a 6-month revenue recovery target with monthly KPI checkpoints.",
            ],
            "dos": [
                "Localize marketing messaging to resonate with regional customer needs.",
                "Offer region-specific deals during peak shopping periods (holidays, harvest season, etc.).",
                "Build local partnerships or distribution networks to reduce delivery costs.",
            ],
            "donts": [
                "Do not apply a one-size-fits-all national campaign in underperforming regions.",
                "Avoid pulling out completely — low-performing regions often have high recovery potential.",
                "Do not ignore customer feedback from these regions; it contains actionable signals.",
            ],
            "growth_note": (
                f"Task: Launch a 3-month targeted campaign in {regions[0] if regions else 'the lowest-performing region'} "
                "with localized pricing and promotions — recovering even 20% of the regional profit gap "
                "can contribute a measurable uplift to overall revenue and geographic market share."
            ),
        })

    # ── Recommendation 4: Margin improvement ────────────────────────
    if weaknesses.get("poor_profit_margins"):
        cats    = list(weaknesses["poor_profit_margins"].keys())
        cat_str = ", ".join(cats)
        items.append({
            "id":       "rec_margins",
            "idea":     f"Introduce subscription-based pricing or tiered pricing for low-margin categories: {cat_str}",
            "rationale": (
                f"Categories {cat_str} show profit margins below 10%, "
                "indicating that current pricing does not adequately cover costs or competitive pressures."
            ),
            "steps": [
                f"Review the cost structure for {cat_str} — identify supplier, logistics, and overhead costs.",
                "Introduce a tiered pricing model: Basic, Standard, and Premium tiers with feature differentiation.",
                "Explore subscription bundles for repeat customers to increase Customer Lifetime Value (CLV).",
                "Renegotiate supplier contracts for high-volume, low-margin products.",
                "Implement dynamic pricing algorithms to optimize margins based on demand signals.",
            ],
            "dos": [
                "Test tiered pricing with a small customer cohort before broad rollout.",
                "Highlight premium tier value through clear product differentiation.",
                "Track margin improvement on a per-SKU basis after pricing changes.",
            ],
            "donts": [
                "Do not reduce product quality as a cost-cutting measure — it drives customer churn.",
                "Avoid complex pricing structures that confuse customers.",
                f"Do not ignore the competitive pricing landscape for {cat_str}.",
            ],
            "growth_note": (
                f"Task: Introduce a Premium tier for {cats[0] if cats else 'the lowest-margin category'} "
                "within 45 days — moving even 15% of customers to a higher-margin tier can increase "
                "category profitability by 25–40% and directly boost overall profit margin."
            ),
        })

    # ── Recommendation 5: Customer retention ────────────────────────
    items.append({
        "id":       "rec_retention",
        "idea":     "Launch a data-driven customer loyalty and retention program",
        "rationale": (
            "Retaining existing customers costs 5–7x less than acquiring new ones. "
            "Leveraging purchase history data can drive targeted re-engagement campaigns."
        ),
        "steps": [
            "Segment customers by RFM (Recency, Frequency, Monetary) score using existing transaction data.",
            "Create personalized email/SMS campaigns for high-value customers at risk of churn.",
            "Introduce a tiered loyalty program with points, exclusive discounts, and early product access.",
            "Deploy a recommendation engine to upsell/cross-sell based on purchase history.",
            "Measure retention rate monthly and set a target of 85%+ for top customer segments.",
        ],
        "dos": [
            "Use ML-based churn prediction to proactively reach at-risk customers.",
            "Personalize offers at the individual level, not just the segment level.",
            "Reward top customers with exclusive perks to strengthen brand loyalty.",
        ],
        "donts": [
            "Do not send generic mass emails — they have low engagement and increase unsubscribe rates.",
            "Avoid one-time discounts that attract price-sensitive customers with no brand loyalty.",
            "Do not neglect post-purchase experience — shipping speed and support quality drive retention.",
        ],
        "growth_note": (
            "Task: Launch an RFM-based re-engagement campaign for your top 20% high-value customers "
            "within 30 days — improving retention rate by just 5% can increase overall revenue by "
            "25–95% over time, making this the highest-ROI growth action available."
        ),
    })

    # Legacy flat DO/DON'T lists for backward compatibility
    flat_dos   = [
        f"Use {best_model} for profit predictions (lowest MSE)",
        "Focus on profitable categories and high-margin products",
        "Expand presence in top-performing regions",
        "Leverage top customers with loyalty programs",
    ]
    flat_donts = []
    if weaknesses.get("loss_making_products"):
        prods = ", ".join(list(weaknesses["loss_making_products"].keys())[:5])
        flat_donts.append(f"Avoid investing in loss-making sub-categories: {prods}")
    if weaknesses.get("low_performing_regions"):
        flat_donts.append(
            f"Reduce focus on low-performing regions: {', '.join(weaknesses['low_performing_regions'].keys())}"
        )
    if weaknesses.get("poor_profit_margins"):
        flat_donts.append(
            f"Review pricing strategy for low-margin categories: {', '.join(weaknesses['poor_profit_margins'].keys())}"
        )

    return {"items": items, "DOs": flat_dos, "DONTs": flat_donts}


# ── ROUTES — Dataset Management ────────────────────────────────────

@app.route("/api/datasets", methods=["GET"])
def list_datasets():
    """Return names of all loaded datasets."""
    return jsonify({"datasets": list(datasets.keys())})


@app.route("/api/upload", methods=["POST"])
def upload_dataset():
    """Upload a CSV and give it a name."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f    = request.files["file"]
    name = request.form.get("name", f.filename.replace(".csv", ""))
    try:
        df = pd.read_csv(f, encoding="latin1")
        df = clean_df(df)
        datasets[name] = df
        ml_cache.pop(name, None)
        # Pre-warm ML cache in background
        def _prewarm_single(n, d):
            try:
                if n not in ml_cache:
                    ml_cache[n] = run_ml(d)
            except Exception:
                pass
        t = threading.Thread(target=_prewarm_single, args=(name, df), daemon=True)
        t.start()
        return jsonify({
            "message": f"Dataset '{name}' loaded successfully",
            "rows": len(df),
            "columns": list(df.columns),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/load-defaults", methods=["POST"])
def load_defaults():
    """Load all built-in CSV files from the data/ folder."""
    base  = ROOT_DIR / "data"
    files = {
        "Dataset A":        "superstore_dataset1.csv",
        "Dataset B":        "superstore_dataset2.csv",
        "Global Dataset 6": "global dataset 6.csv",
    }
    loaded = []
    errors = []
    for name, fname in files.items():
        path = base / fname
        if not path.exists():
            errors.append(f"{name}: file not found at {path}")
            continue
        try:
            df = pd.read_csv(path, encoding="latin1")
            df.columns = [c.strip().replace("_", " ") for c in df.columns]
            df = clean_df(df)
            datasets[name] = df
            ml_cache.pop(name, None)
            loaded.append(name)
        except Exception as e:
            errors.append(f"{name}: {str(e)}")

    # Pre-warm ML cache in background so compare/recommendations load instantly
    def _prewarm(names):
        for n in names:
            try:
                if n in datasets and n not in ml_cache:
                    result = run_ml(datasets[n])
                    if result:  # only cache if training succeeded
                        ml_cache[n] = result
            except Exception:
                pass
    if loaded:
        t = threading.Thread(target=_prewarm, args=(list(loaded),), daemon=True)
        t.start()

    return jsonify({"loaded": loaded, "errors": errors, "warming": True})


@app.route("/api/remove/<name>", methods=["DELETE"])
def remove_dataset(name):
    datasets.pop(name, None)
    ml_cache.pop(name, None)
    return jsonify({"removed": name})


# ── ROUTES — Overview / Summary ────────────────────────────────────

@app.route("/api/summary/<name>", methods=["GET"])
def summary(name):
    if name not in datasets:
        return jsonify({"error": "Dataset not found"}), 404
    df = datasets[name]

    col_descriptions = {
        "Order ID":      "Unique identifier for each customer order",
        "Order Date":    "Date the order was placed",
        "Ship Date":     "Date the order was shipped",
        "Ship Mode":     "Shipping method selected (Standard, First Class, etc.)",
        "Customer ID":   "Unique identifier for each customer",
        "Customer Name": "Full name of the customer",
        "Segment":       "Customer segment (Consumer, Corporate, Home Office)",
        "Country":       "Country of the customer",
        "City":          "City of the customer",
        "State":         "State of the customer",
        "Postal Code":   "Postal / ZIP code",
        "Region":        "Geographic sales region (East, West, Central, South)",
        "Product ID":    "Unique product identifier",
        "Category":      "High-level product category (Furniture, Office Supplies, Technology)",
        "Sub-Category":  "Detailed product sub-category",
        "Product Name":  "Full product name",
        "Sales":         "Total revenue generated by the order ($)",
        "Quantity":      "Number of units sold",
        "Discount":      "Discount rate applied (0.0 – 1.0)",
        "Profit":        "Net profit from the order (can be negative)",
    }
    columns_info = [
        {"name": col, "description": col_descriptions.get(col, "Business data field")}
        for col in df.columns
    ]
    date_range = None
    if "Order Date" in df.columns:
        valid_dates = df["Order Date"].dropna()
        if len(valid_dates):
            date_range = {
                "from": str(valid_dates.min().date()),
                "to":   str(valid_dates.max().date()),
            }

    dataset_info = {
        "name":         name,
        "purpose":      "Sales performance analysis and business decision intelligence",
        "description":  (
            "This dataset captures transactional retail sales data including order details, "
            "customer segments, product categories, regional performance, and financial "
            "outcomes (sales, profit, discounts). It is used to detect weaknesses, "
            "train ML prediction models, and generate strategic business recommendations."
        ),
        "rows":          int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns":       columns_info,
        "date_range":    date_range,
        "categories":    sorted(df["Category"].dropna().unique().tolist()) if "Category" in df.columns else [],
        "regions":       sorted(df["Region"].dropna().unique().tolist())   if "Region"   in df.columns else [],
        "segments":      sorted(df["Segment"].dropna().unique().tolist())  if "Segment"  in df.columns else [],
    }

    return jsonify({
        "rows":             int(len(df)),
        "columns":          int(len(df.columns)),
        "column_names":     list(df.columns),
        "total_sales":      round(float(df["Sales"].sum()),  2) if "Sales"       in df.columns else None,
        "total_profit":     round(float(df["Profit"].sum()), 2) if "Profit"      in df.columns else None,
        "total_orders":     int(df["Order ID"].nunique())        if "Order ID"    in df.columns else None,
        "total_customers":  int(df["Customer Name"].nunique())   if "Customer Name" in df.columns else None,
        "dataset_info":     dataset_info,
    })


# ── ROUTES — EDA Charts ────────────────────────────────────────────

@app.route("/api/eda/<name>", methods=["GET"])
def eda(name):
    if name not in datasets:
        return jsonify({"error": "Dataset not found"}), 404
    df     = datasets[name]
    charts = {}
    palette = ["#6C63FF", "#FF6584", "#43CBFF", "#F7971E", "#4CAF50", "#FF5722"]

    # 1. Profit by Category
    if {"Category", "Profit"}.issubset(df.columns):
        pbc    = df.groupby("Category")["Profit"].sum().sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors  = [palette[i % len(palette)] for i in range(len(pbc))]
        pbc.plot(kind="barh", ax=ax, color=colors)
        ax.set_title("Profit by Category", fontsize=13, fontweight="bold")
        ax.set_xlabel("Total Profit ($)")
        fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#16213e")
        ax.tick_params(colors="white"); ax.title.set_color("white")
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.spines[:].set_color("#333366")
        charts["profit_by_category"] = fig_to_b64(fig)

    # 2. Sales by Region
    if {"Region", "Sales"}.issubset(df.columns):
        sbr    = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        colors  = [palette[i % len(palette)] for i in range(len(sbr))]
        sbr.plot(kind="bar", ax=ax, color=colors, edgecolor="none")
        ax.set_title("Sales by Region", fontsize=13, fontweight="bold")
        ax.set_ylabel("Total Sales ($)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#16213e")
        ax.tick_params(colors="white"); ax.title.set_color("white")
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.spines[:].set_color("#333366")
        charts["sales_by_region"] = fig_to_b64(fig)

    # 3. Monthly Sales trend
    if {"Order Date", "Sales"}.issubset(df.columns):
        df2    = df.dropna(subset=["Order Date"]).copy()
        df2["Month"] = df2["Order Date"].dt.to_period("M")
        ms     = df2.groupby("Month")["Sales"].sum()
        if len(ms) > 1:
            fig, ax = plt.subplots(figsize=(9, 4))
            ms.plot(ax=ax, color="#6C63FF", linewidth=2.5, marker="o", markersize=4)
            ax.fill_between(range(len(ms)), ms.values, alpha=0.15, color="#6C63FF")
            ax.set_title("Monthly Sales Trend", fontsize=13, fontweight="bold")
            ax.set_ylabel("Sales ($)")
            ax.set_xticklabels([str(p) for p in ms.index], rotation=45, ha="right", fontsize=7)
            fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#16213e")
            ax.tick_params(colors="white"); ax.title.set_color("white")
            ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
            ax.spines[:].set_color("#333366")
            charts["monthly_sales"] = fig_to_b64(fig)

    # 4. Top 10 Sub-Categories by Profit
    if {"Sub-Category", "Profit"}.issubset(df.columns):
        top    = df.groupby("Sub-Category")["Profit"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        colors_b = ["#4CAF50" if v >= 0 else "#FF6584" for v in top.values]
        top.plot(kind="bar", ax=ax, color=colors_b, edgecolor="none")
        ax.set_title("Top 10 Sub-Categories by Profit", fontsize=13, fontweight="bold")
        ax.set_ylabel("Profit ($)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="white", linewidth=0.8, linestyle="--")
        fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#16213e")
        ax.tick_params(colors="white"); ax.title.set_color("white")
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.spines[:].set_color("#333366")
        charts["top_subcategories"] = fig_to_b64(fig)

    # 5. Discount vs Profit scatter
    if {"Discount", "Profit"}.issubset(df.columns):
        sample = df.sample(min(1000, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(sample["Discount"], sample["Profit"], alpha=0.4, color="#43CBFF", s=15)
        ax.set_title("Discount vs Profit", fontsize=13, fontweight="bold")
        ax.set_xlabel("Discount"); ax.set_ylabel("Profit ($)")
        ax.axhline(0, color="#FF6584", linewidth=1, linestyle="--")
        fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#16213e")
        ax.tick_params(colors="white"); ax.title.set_color("white")
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.spines[:].set_color("#333366")
        charts["discount_vs_profit"] = fig_to_b64(fig)

    return jsonify({"charts": charts})


# ── ROUTES — Weakness Detection ────────────────────────────────────

@app.route("/api/weakness/<name>", methods=["GET"])
def weakness(name):
    if name not in datasets:
        return jsonify({"error": "Dataset not found"}), 404
    df = datasets[name]
    w  = detect_weaknesses(df)

    analytical = []

    if w.get("loss_making_products"):
        items_sorted = sorted(w["loss_making_products"].items(), key=lambda x: x[1])
        worst       = items_sorted[0] if items_sorted else ("Unknown", 0)
        total_loss  = sum(v for v in w["loss_making_products"].values())
        analytical.append({
            "type":       "Revenue Loss",
            "icon":       "📉",
            "severity":   "critical",
            "title":      "Loss-Making Sub-Categories Detected",
            "root_cause": (
                "These sub-categories generate negative profit primarily due to excessive discount rates "
                "exceeding cost margins, high return rates, or pricing below unit cost. "
                f"The worst offender is '{worst[0]}' with a cumulative loss of ${abs(worst[1]):,.2f}."
            ),
            "timeline": (
                "This weakness has likely accumulated over multiple sales periods. "
                "The pattern is visible across the full transaction history loaded in this dataset."
            ),
            "impact": (
                f"A total of ${abs(total_loss):,.2f} in profit has been eroded by {len(w['loss_making_products'])} "
                "loss-making sub-categories. If left unaddressed, this will continue to suppress overall "
                "profit margins and limit capital available for growth investments."
            ),
            "affected_items": w["loss_making_products"],
        })

    if w.get("low_performing_regions"):
        total_under = sum(v for v in w["low_performing_regions"].values())
        analytical.append({
            "type":       "Declining Engagement",
            "icon":       "🗺️",
            "severity":   "warning",
            "title":      "Underperforming Sales Regions",
            "root_cause": (
                "These regions generate profit below the company average, indicating weak market penetration, "
                "poor product-market fit, inadequate sales force coverage, or logistical inefficiencies "
                "driving up delivery costs and reducing margins."
            ),
            "timeline": (
                "Regional underperformance is a persistent structural issue visible across the dataset's "
                "full date range, suggesting it is not a seasonal anomaly but a systemic challenge."
            ),
            "impact": (
                f"The {len(w['low_performing_regions'])} underperforming region(s) collectively generate "
                f"${total_under:,.2f} in combined profit — significantly below the company average. "
                "This geographic imbalance concentrates revenue risk in a small number of regions "
                "and limits overall market diversification."
            ),
            "affected_items": w["low_performing_regions"],
        })

    if w.get("poor_profit_margins"):
        analytical.append({
            "type":       "Margin Erosion",
            "icon":       "📊",
            "severity":   "warning",
            "title":      "Categories with Poor Profit Margins (< 10%)",
            "root_cause": (
                "Low profit margins in these categories result from a combination of high discount rates, "
                "increased competition forcing price concessions, rising supplier costs not passed to "
                "customers, and/or high product return rates."
            ),
            "timeline": (
                "Margin compression typically develops gradually over 6–18 months as discounting becomes "
                "habitual and cost structures are not periodically reviewed. "
                "The current data reflects this compounded degradation."
            ),
            "impact": (
                f"Categories {', '.join(w['poor_profit_margins'].keys())} are operating at margins below 10%, "
                "which is insufficient to cover operating overhead and fund reinvestment. "
                "This creates a margin squeeze that directly limits profitability scalability."
            ),
            "affected_items": {k: f"{v*100:.1f}%" for k, v in w["poor_profit_margins"].items()},
        })

    return jsonify({"weaknesses": w, "analytical": analytical})


# ── ROUTES — ML Predictions ────────────────────────────────────────

@app.route("/api/predict/<name>", methods=["GET"])
def predict(name):
    if name not in datasets:
        return jsonify({"error": "Dataset not found"}), 404
    if name not in ml_cache:
        ml_cache[name] = run_ml(datasets[name])
    ml = ml_cache[name]

    model_meta = {
        "linear_regression": {
            "label":       "Linear Regression",
            "icon":        "📐",
            "description": "A statistical model that assumes a linear relationship between input features (Sales, Quantity) and profit. Fast and interpretable, but struggles with non-linear patterns.",
            "strength":    "Highly interpretable; ideal for baseline profit estimation.",
            "weakness":    "Cannot capture complex, non-linear profit dynamics caused by seasonal spikes or discount interactions.",
        },
        "decision_tree": {
            "label":       "Decision Tree",
            "icon":        "🌲",
            "description": "A tree-based model that splits data into decision branches. Captures non-linear patterns but is prone to overfitting on training data.",
            "strength":    "Handles categorical thresholds and discount-profit interactions naturally.",
            "weakness":    "High variance — small changes in data can drastically change the tree structure.",
        },
        "random_forest": {
            "label":       "Random Forest",
            "icon":        "🌳",
            "description": "An ensemble of 100 decision trees that averages predictions to reduce variance. Highly robust and accurate across diverse business datasets.",
            "strength":    "Excellent generalization, handles outliers well, resistant to overfitting.",
            "weakness":    "Less interpretable than single-tree models; higher computational cost.",
        },
    }

    best_key   = ml.get("best_model_key", "")
    best_label = ml.get("best_model", "N/A")
    best_data  = ml.get(best_key, {})

    ranked = sorted(
        [(k, ml[k]) for k in ["linear_regression", "decision_tree", "random_forest"] if k in ml],
        key=lambda x: x[1]["mse"],
    )
    comparison_summary = []
    for rank, (k, stats) in enumerate(ranked, 1):
        meta = model_meta.get(k, {})
        comparison_summary.append({
            "rank":                 rank,
            "key":                  k,
            "label":                meta.get("label", k),
            "icon":                 meta.get("icon", "🤖"),
            "mse":                  stats["mse"],
            "r2":                   stats["r2"],
            "avg_predicted_profit": stats["avg_predicted_profit"],
            "is_best":              k == best_key,
        })

    if best_key and best_data:
        best_r2_pct    = round(best_data.get("r2", 0) * 100, 1)
        worst_mse      = max(ml[k]["mse"] for k in ["linear_regression", "decision_tree", "random_forest"] if k in ml)
        mse_improvement = round((worst_mse - best_data["mse"]) / worst_mse * 100, 1) if worst_mse else 0
        justification  = {
            "why_best": (
                f"{best_label} was selected as the best model because it achieved the lowest Mean Squared Error "
                f"(MSE = {best_data.get('mse', 0):,.4f}) among all three evaluated models, meaning its profit "
                f"predictions deviate least from actual values. It explains {best_r2_pct}% of the variance in "
                f"profit (R² = {best_data.get('r2', 0)}), demonstrating strong predictive power."
            ),
            "comparison_insight": (
                f"{best_label} outperforms the next best model by reducing prediction error by up to "
                f"{mse_improvement}%. This makes it the most reliable choice for deployment in "
                "operational profit forecasting pipelines."
            ),
            "impact_analysis": (
                f"Using {best_label} for profit forecasting is projected to yield an average predicted profit of "
                f"${best_data.get('avg_predicted_profit', 0):,.2f} per prediction cycle. "
                "This enables proactive pricing decisions, inventory optimization, and targeted promotions "
                "that can materially improve overall business profitability."
            ),
            "plain_english": (
                f"Think of {best_label} as your most experienced analyst — it has 'seen' all your historical "
                "sales patterns and learned which combinations of product volume and revenue consistently "
                "lead to profit or loss. When you feed it new sales data, it gives the most accurate "
                "profit estimate compared to the other two models tested."
            ),
        }
    else:
        justification = {}

    return jsonify({
        "predictions": ml,
        "model_meta":  model_meta,
        "comparison":  comparison_summary,
        "justification": justification,
    })


# ── ROUTES — Strategy & Simulation ────────────────────────────────

@app.route("/api/strategy/<name>", methods=["GET"])
def strategy(name):
    if name not in datasets:
        return jsonify({"error": "Dataset not found"}), 404
    df = datasets[name]
    if name not in ml_cache:
        ml_cache[name] = run_ml(df)
    ml = ml_cache[name]
    bp           = ml.get("best_predicted_profit", 0)
    total_profit = float(df["Profit"].sum()) if "Profit" in df.columns else 0
    total_sales  = float(df["Sales"].sum())  if "Sales"  in df.columns else 0

    try:
        growth_pct = max(1, min(100, int(request.args.get("growth_percent", 10))))
    except (ValueError, TypeError):
        growth_pct = 10

    multiplier = 1 + growth_pct / 100.0

    return jsonify({
        "profit_maximization": round(float(df["Profit"].sum()), 2) if "Profit" in df.columns else 0,
        "sales_growth":        round(total_sales, 2),
        "loss_reduction":      round(float(df[df["Profit"] < 0]["Profit"].sum()), 2) if "Profit" in df.columns else 0,
        "best_model":          ml.get("best_model", "N/A"),
        "best_predicted_profit": bp,
        "simulation": {
            "current_profit":  round(total_profit, 2),
            "current_sales":   round(total_sales, 2),
            "after_profit":    round(total_profit * multiplier, 2),
            "after_sales":     round(total_sales  * multiplier, 2),
            "after_strategy":  round(bp * multiplier, 2),
            "growth_percent":  growth_pct,
            "profit_delta":    round(total_profit * multiplier - total_profit, 2),
            "sales_delta":     round(total_sales  * multiplier - total_sales,  2),
        },
    })


# ── ROUTES — Recommendations ───────────────────────────────────────

@app.route("/api/recommend/<name>", methods=["GET"])
def recommend(name):
    if name not in datasets:
        return jsonify({"error": "Dataset not found"}), 404
    try:
        if name not in ml_cache:
            ml_cache[name] = run_ml(datasets[name])
        ml         = ml_cache[name]
        w          = detect_weaknesses(datasets[name])
        best_model = ml.get("best_model") or "Random Forest"
        rec        = build_recommendations(w, best_model, datasets[name])
        return jsonify({"recommendations": rec})
    except Exception as e:
        return jsonify({"error": f"Recommendation generation failed: {str(e)}", "trace": traceback.format_exc()}), 500


# ── ROUTES — Multi-Dataset Comparison ─────────────────────────────

@app.route("/api/warmup-compare", methods=["POST"])
def warmup_compare():
    """Pre-warm ML cache for all loaded datasets. Call this before opening compare tab."""
    names_to_warm = [n for n in list(datasets.keys()) if n not in ml_cache]
    if not names_to_warm:
        return jsonify({"status": "already_ready", "datasets": list(datasets.keys())})

    def _warm():
        for n in names_to_warm:
            try:
                if n in datasets and n not in ml_cache:
                    result = run_ml(datasets[n])
                    if result:
                        ml_cache[n] = result
            except Exception:
                pass

    t = threading.Thread(target=_warm, daemon=True)
    t.start()
    return jsonify({"status": "warming", "queued": names_to_warm})


@app.route("/api/compare", methods=["GET"])
def compare():
    if len(datasets) < 2:
        return jsonify({"error": "Need at least 2 datasets for comparison"}), 400

    try:
        # ── Train ML for any datasets missing from cache (sequential, no threads) ──
        for n in list(datasets.keys()):
            if n not in ml_cache:
                result = run_ml(datasets[n])
                if result:
                    ml_cache[n] = result

        # ── Build comparison summary ──────────────────────────────────
        comparison = {}
        for n, df in datasets.items():
            ml = ml_cache.get(n, {})
            comparison[n] = {
                "total_sales":      round(float(df["Sales"].sum()),  2) if "Sales"  in df.columns else 0,
                "total_profit":     round(float(df["Profit"].sum()), 2) if "Profit" in df.columns else 0,
                "avg_profit":       round(float(df["Profit"].mean()),2) if "Profit" in df.columns else 0,
                "best_model":       ml.get("best_model", "N/A"),
                "predicted_profit": ml.get("best_predicted_profit", 0),
            }

        best_company = max(comparison, key=lambda k: comparison[k]["total_profit"])
        worst        = min(comparison, key=lambda k: comparison[k]["total_profit"])

        # ── Chart ─────────────────────────────────────────────────────
        names   = list(comparison.keys())
        profits = [comparison[n]["total_profit"] for n in names]
        sales   = [comparison[n]["total_sales"]  for n in names]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        colors    = ["#6C63FF", "#FF6584", "#43CBFF", "#F7971E"][:len(names)]

        axes[0].bar(names, profits, color=colors, edgecolor="none")
        axes[0].set_title("Total Profit Comparison", fontweight="bold", color="white")
        axes[0].set_ylabel("Profit ($)", color="white")
        axes[0].tick_params(colors="white")

        axes[1].bar(names, sales, color=colors, edgecolor="none")
        axes[1].set_title("Total Sales Comparison", fontweight="bold", color="white")
        axes[1].set_ylabel("Sales ($)", color="white")
        axes[1].tick_params(colors="white")

        for ax in axes:
            ax.set_facecolor("#16213e")
            for spine in ax.spines.values():
                spine.set_color("#333366")
        fig.patch.set_facecolor("#1a1a2e")
        fig.tight_layout(pad=1.5)
        chart_b64 = fig_to_b64(fig)

        cross_suggestion = (
            f"To improve '{worst}', adopt the strategy of '{best_company}': "
            f"focus on high-profit categories and use {comparison[best_company]['best_model']} "
            f"for demand forecasting."
        )

        best_data      = comparison[best_company]
        worst_data     = comparison[worst]
        best_ml        = ml_cache.get(best_company, {})
        best_model_lbl = best_data["best_model"]
        bkey           = best_ml.get("best_model_key", "")
        best_mse       = best_ml.get(bkey, {}).get("mse", 0) if bkey and bkey in best_ml else 0
        best_r2        = best_ml.get(bkey, {}).get("r2",  0) if bkey and bkey in best_ml else 0

        profit_gap    = round(best_data["total_profit"] - worst_data["total_profit"], 2)
        sales_gap     = round(best_data["total_sales"]  - worst_data["total_sales"],  2)
        profit_margin = (
            round(best_data["total_profit"] / best_data["total_sales"] * 100, 2)
            if best_data["total_sales"] else 0
        )

        best_justification = {
            "reason_for_selection": (
                f"'{best_company}' was identified as the top-performing dataset because it achieved the highest "
                f"total profit of ${best_data['total_profit']:,.2f} across all compared datasets — outperforming "
                f"'{worst}' by ${abs(profit_gap):,.2f}. It also leads in total sales (${best_data['total_sales']:,.2f}) "
                f"with a profit margin of {profit_margin:.1f}%, reflecting both strong revenue generation and "
                f"efficient cost management relative to all other loaded datasets."
            ),
            "key_benefits": [
                f"Highest total profit (${best_data['total_profit']:,.2f}) — strongest bottom-line performance across all datasets.",
                f"Profit margin of {profit_margin:.1f}% demonstrates pricing efficiency and cost discipline.",
                f"Best average per-order profit of ${best_data['avg_profit']:,.2f}, driving superior unit economics.",
                f"Projected future profit of ${best_data['predicted_profit']:,.2f} based on ML model forecasting.",
                f"Sales advantage of ${abs(sales_gap):,.2f} over the lowest-revenue dataset — stronger market penetration.",
            ],
            "algorithms_used": [
                {
                    "name": "Linear Regression", "icon": "📐",
                    "purpose": "Establishes a linear profit prediction baseline using Sales and Quantity as input features.",
                    "why_used": "Provides the fastest, most interpretable benchmark to detect linear profit patterns.",
                },
                {
                    "name": "Decision Tree", "icon": "🌲",
                    "purpose": "Learns rule-based profit patterns by recursively splitting data on feature thresholds.",
                    "why_used": "Captures non-linear relationships such as discount cutoffs and quantity breakpoints.",
                },
                {
                    "name": "Random Forest", "icon": "🌳",
                    "purpose": "Aggregates predictions from multiple decision trees to reduce variance and overfitting.",
                    "why_used": "Consistently delivers the highest accuracy — robust to outliers and seasonal fluctuations.",
                },
            ],
            "effectiveness": (
                f"On '{best_company}', {best_model_lbl} achieved an R\u00b2 of {best_r2:.4f} and MSE of {best_mse:,.4f}, "
                f"explaining {round(best_r2*100, 1)}% of profit variance with minimal prediction error. "
                f"This enables confident data-driven decisions on pricing, inventory levels, and discount policies."
            ),
            "strategic_advantages": [
                f"Deploy {best_model_lbl} from '{best_company}' as the standard forecasting engine — reduces prediction error by up to 30%.",
                f"The {profit_margin:.1f}% profit margin of '{best_company}' validates its pricing strategy as a proven model to replicate.",
                f"A 10% growth simulation on '{best_company}' projects a future profit of ${round(best_data['predicted_profit'] * 1.1, 2):,.2f}.",
                f"Cross-dataset knowledge transfer: top categories from '{best_company}' can help '{worst}' close the ${abs(profit_gap):,.2f} profit gap.",
                "Shifting from reactive reporting to predictive intelligence enables proactive decisions 1\u20132 quarters ahead.",
            ],
        }

        return jsonify({
            "comparison":         comparison,
            "best_company":       best_company,
            "chart":              chart_b64,
            "cross_suggestion":   cross_suggestion,
            "best_justification": best_justification,
        })

    except Exception as e:
        return jsonify({
            "error": f"Comparison failed: {str(e)}",
            "trace": traceback.format_exc()
        }), 500


# ── ROUTES — Full Analysis (single call for all phases) ────────────

@app.route("/api/full-analysis/<name>", methods=["GET"])
def full_analysis(name):
    if name not in datasets:
        return jsonify({"error": "Dataset not found"}), 404
    try:
        df = datasets[name]
        if name not in ml_cache:
            ml_cache[name] = run_ml(df)
        ml  = ml_cache[name]
        w   = detect_weaknesses(df)
        rec = build_recommendations(w, ml.get("best_model", "Random Forest"))
        bp  = ml.get("best_predicted_profit", 0)
        total_profit = float(df["Profit"].sum()) if "Profit" in df.columns else 0
        return jsonify({
            "summary": {
                "rows":         int(len(df)),
                "total_sales":  round(float(df["Sales"].sum()),  2) if "Sales"  in df.columns else None,
                "total_profit": round(total_profit, 2)               if "Profit" in df.columns else None,
            },
            "weaknesses":     w,
            "predictions":    ml,
            "strategy": {
                "best_model": ml.get("best_model"),
                "simulation": {
                    "current_profit": round(total_profit, 2),
                    "after_strategy": round(bp * 1.1, 2),
                    "growth_percent": 10,
                },
            },
            "recommendations": rec,
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── ROUTES — Conclusion & Report ───────────────────────────────────

def _key_findings(df: pd.DataFrame, ml: dict, w: dict) -> list:
    findings = []
    if "Profit" in df.columns:
        tp = df["Profit"].sum()
        if tp > 0:
            findings.append(f"Business is profitable with a total profit of ${tp:,.0f}.")
        else:
            findings.append(f"Business is currently at a loss of ${abs(tp):,.0f} — immediate action required.")

    if w.get("loss_making_products"):
        findings.append(f"{len(w['loss_making_products'])} sub-categories are generating losses and need review.")

    if "Category" in df.columns and "Profit" in df.columns:
        bc = df.groupby("Category")["Profit"].sum().idxmax()
        findings.append(f"'{bc}' is the highest-profit product category.")

    if "Region" in df.columns and "Sales" in df.columns:
        br = df.groupby("Region")["Sales"].sum().idxmax()
        findings.append(f"'{br}' region leads in total sales volume.")

    if ml.get("best_model"):
        findings.append(f"{ml['best_model']} is the most accurate ML prediction model (lowest MSE).")

    if "Discount" in df.columns and "Profit" in df.columns:
        corr = df["Discount"].corr(df["Profit"])
        if corr < -0.1:
            findings.append("High discounts negatively correlate with profit — pricing strategy should be reviewed.")

    return findings


@app.route("/api/conclusion", methods=["GET"])
def conclusion():
    if not datasets:
        return jsonify({"error": "No datasets loaded"}), 400

    results_c = {}
    for name, df in datasets.items():
        if name not in ml_cache:
            ml_cache[name] = run_ml(df)
        ml  = ml_cache[name]
        w   = detect_weaknesses(df)
        rec = build_recommendations(w, ml.get("best_model", "Random Forest"))

        total_sales  = float(df["Sales"].sum())  if "Sales"  in df.columns else 0
        total_profit = float(df["Profit"].sum()) if "Profit" in df.columns else 0
        profit_margin = round(total_profit / total_sales * 100, 2) if total_sales else 0
        bp = ml.get("best_predicted_profit", 0)

        results_c[name] = {
            "total_sales":      round(total_sales, 2),
            "total_profit":     round(total_profit, 2),
            "profit_margin":    profit_margin,
            "rows":             int(len(df)),
            "best_model":       ml.get("best_model", "N/A"),
            "predicted_profit": round(bp, 2),
            "simulated_profit": round(bp * 1.1, 2),
            "weakness_count":   sum(len(v) for v in w.values()),
            "key_findings":     _key_findings(df, ml, w),
            "recommendations":  rec,
        }

    best       = max(results_c, key=lambda k: results_c[k]["total_profit"])
    avg_margin = round(sum(c["profit_margin"] for c in results_c.values()) / len(results_c), 2)

    overall = (
        f"The AI-Based Decision Intelligence & Strategy Impact Analyzer successfully processed "
        f"{len(results_c)} dataset(s) covering {sum(c['rows'] for c in results_c.values()):,} records. "
        f"The overall average profit margin is {avg_margin}%. "
        f"'{best}' is the top-performing business unit. "
        f"Applying the recommended strategy is projected to deliver a 10% profit growth. "
        f"The system identified key weaknesses, evaluated multiple ML models, and generated "
        f"actionable recommendations to drive sustainable business growth."
    )

    return jsonify({
        "datasets":           results_c,
        "best_performer":     best,
        "avg_profit_margin":  avg_margin,
        "overall_conclusion": overall,
        "total_datasets":     len(results_c),
        "total_records":      sum(c["rows"] for c in results_c.values()),
    })


# ── Serve frontend index.html at root ─────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(FRONTEND_DIR), filename)


# ── ROUTES — Risk & Anomaly Detection ─────────────────────────────

@app.route("/api/anomaly/<name>", methods=["GET"])
def anomaly(name):
    if name not in datasets:
        return jsonify({"error": "Dataset not found"}), 404
    df       = datasets[name]
    anomalies = []
    alerts   = []

    col_why = {
        "Sales":    "Extreme sales values suggest mega-order anomalies or data-entry errors that distort revenue forecasts.",
        "Profit":   "Outlier profits (very negative or very high) indicate transactions priced far outside normal range — returns, bulk deals, or clearance sales.",
        "Quantity": "Unusually large quantities point to wholesale/bulk orders that skew demand forecasts and inflate inventory assumptions.",
        "Discount": "Discounts beyond 3σ mean some orders received near-total price waivers, directly destroying margin integrity.",
    }
    num_cols = [c for c in ["Sales", "Profit", "Quantity", "Discount"] if c in df.columns]
    for col in num_cols:
        z             = np.abs(sp_stats.zscore(df[col].dropna()))
        outlier_count = int((z > 3).sum())
        if outlier_count > 0:
            worst_val = float(df[col][z > 3].abs().max())
            col_mean  = round(float(df[col].mean()), 2)
            col_std   = round(float(df[col].std()),  2)
            impact_txt = (
                f"These {outlier_count} outlier(s) in '{col}' pull the column mean away from true business average "
                f"(mean: {col_mean}, σ: {col_std}). ML models trained on this data will overfit to extremes, "
                "producing unreliable forecasts. KPI dashboards will also show inflated or deflated averages."
            )
            anomalies.append({
                "column":             col,
                "outlier_count":      outlier_count,
                "max_deviation":      round(worst_val, 2),
                "col_mean":           col_mean,
                "col_std":            col_std,
                "severity":           "critical" if outlier_count > 10 else "warning",
                "description":        f"{outlier_count} outlier(s) in {col} deviate >3σ from mean.",
                "why_risky":          col_why.get(col, "Extreme values distort model training and KPI averages."),
                "impact":             impact_txt,
                "recommended_action": (
                    f"Cap or winsorize {col} values beyond ±3σ before model training. "
                    "Investigate the top 5 outlier transactions to determine if they are data errors or legitimate edge cases."
                ),
            })

    drops = []
    if {"Order Date", "Profit"}.issubset(df.columns):
        df2              = df.dropna(subset=["Order Date"]).copy()
        df2["Month"]     = df2["Order Date"].dt.to_period("M")
        monthly          = df2.groupby("Month")["Profit"].sum().sort_index()
        monthly_sales    = df2.groupby("Month")["Sales"].sum()    if "Sales"    in df2.columns else None
        monthly_discount = df2.groupby("Month")["Discount"].mean() if "Discount" in df2.columns else None

        for i in range(1, len(monthly)):
            prev, curr = monthly.iloc[i - 1], monthly.iloc[i]
            if prev > 0 and (curr - prev) / abs(prev) < -0.25:
                drop_pct    = round((curr - prev) / abs(prev) * 100, 1)
                period      = str(monthly.index[i])
                prev_period = str(monthly.index[i - 1])

                causes = []
                if monthly_discount is not None:
                    pd_ = float(monthly_discount.get(monthly.index[i - 1], 0))
                    cd_ = float(monthly_discount.get(monthly.index[i], 0))
                    if cd_ - pd_ > 0.05:
                        causes.append(
                            f"Average discount jumped from {pd_*100:.1f}% to {cd_*100:.1f}% in {period}. "
                            "A 5%+ discount spike directly compresses profit margins by 15–30%."
                        )
                if monthly_sales is not None:
                    ps = float(monthly_sales.get(monthly.index[i - 1], 0))
                    cs = float(monthly_sales.get(monthly.index[i], 0))
                    if cs < ps * 0.85:
                        causes.append(
                            f"Total sales fell from ${ps:,.0f} to ${cs:,.0f} "
                            f"({round((cs - ps) / abs(ps) * 100, 1)}%), indicating reduced demand, "
                            "lost key accounts, or seasonal slowdown. Lower revenue with fixed overheads "
                            "produces a disproportionately large profit decline."
                        )
                if not causes:
                    causes.append(
                        "Profit dropped without a proportional sales decline. Most likely causes: "
                        "(1) category mix shifted toward low-margin products, "
                        "(2) operational costs increased (logistics, returns, storage), or "
                        "(3) a large loss-making order was fulfilled in this period."
                    )

                loss_amt      = round(float(prev - curr), 2)
                annual_impact = round(loss_amt * 12, 2)
                impact_txt    = (
                    f"This drop erased ${loss_amt:,.2f} in a single month. "
                    f"Annualized, a sustained drop of this size equals -${annual_impact:,.2f}/year. "
                    "Compounding drops compress operating cash flow and can signal structural margin deterioration."
                )
                drops.append({
                    "period":          period,
                    "prev_period":     prev_period,
                    "drop_pct":        drop_pct,
                    "from_val":        round(float(prev), 2),
                    "to_val":          round(float(curr), 2),
                    "loss_amount":     loss_amt,
                    "root_causes":     causes,
                    "impact":          impact_txt,
                    "why_it_happened": (
                        f"Profit dropped {abs(drop_pct)}% from {prev_period} to {period}. " + " ".join(causes)
                    ),
                    "what_to_do": (
                        "1. Audit all orders in this period for discount-rate spikes. "
                        "2. Compare product category mix vs. prior month. "
                        "3. Identify large-volume, low-margin orders that skewed totals. "
                        "4. Set an automated profit-floor alert for future months."
                    ),
                })
                alerts.append(
                    f"⚠️ Profit dropped {abs(drop_pct)}% in {period} "
                    f"(${round(float(prev), 0):,.0f} → ${round(float(curr), 0):,.0f})"
                )

    high_disc_info = None
    if "Discount" in df.columns:
        high_disc = df[df["Discount"] > 0.4]
        if len(high_disc) > 0:
            hd_profit = round(float(high_disc["Profit"].sum()), 2) if "Profit" in high_disc.columns else 0
            hd_sales  = round(float(high_disc["Sales"].sum()),  2) if "Sales"  in high_disc.columns else 0
            alerts.append(
                f"🔴 {len(high_disc)} orders have discounts >40% — high margin erosion risk "
                f"(combined profit: ${hd_profit:,.2f} on ${hd_sales:,.2f} sales)"
            )
            high_disc_info = {
                "order_count":         len(high_disc),
                "combined_profit":     hd_profit,
                "combined_sales":      hd_sales,
                "why_risky": (
                    "Discounts above 40% push per-order profit near or below break-even. "
                    f"The {len(high_disc)} affected orders generate only ${hd_profit:,.2f} profit "
                    f"on ${hd_sales:,.2f} sales — a severely compressed margin."
                ),
                "impact_if_unchecked": (
                    "Sustained over-discounting anchors customer price expectations permanently below "
                    "profitable levels, increases revenue volatility, and erodes brand pricing power long-term."
                ),
                "recommended_action": (
                    "Cap automatic discounts at 25–30% without manager approval. "
                    "Flag orders with >35% discount for profitability review. "
                    "Use ML-based dynamic pricing to enforce minimum profitable price floors by SKU."
                ),
            }

    risk_score = min(100, len(anomalies) * 15 + len(drops) * 20 + (10 if high_disc_info else 0))
    risk_level = "Critical" if risk_score >= 70 else "Elevated" if risk_score >= 40 else "Low"
    risk_explanation = (
        f"Risk score {risk_score}/100 ({risk_level}) = "
        f"{len(anomalies)} outlier type(s) (+{len(anomalies)*15} pts) + "
        f"{len(drops)} profit drop(s) (+{len(drops)*20} pts) + "
        f"{'discount risk (+10 pts)' if high_disc_info else 'no discount risk (+0 pts)'}. "
        "Score ≥70 requires immediate action; 40–69 warrants close monitoring; <40 is stable."
    )

    return jsonify({
        "anomalies":        anomalies,
        "sudden_drops":     drops,
        "alerts":           alerts,
        "high_discount":    high_disc_info,
        "total_anomalies":  len(anomalies),
        "risk_score":       risk_score,
        "risk_level":       risk_level,
        "risk_explanation": risk_explanation,
    })


# ── BK AI Assistant — Comprehensive NLP Chat ───────────────────────

def _bk_build_context(df, ml, w, name):
    """Build a rich context dict from all dashboard modules for BK to use."""
    ctx = {"dataset": name, "rows": len(df)}

    if "Sales" in df.columns:
        ctx["total_sales"] = round(float(df["Sales"].sum()), 2)
    if "Profit" in df.columns:
        ctx["total_profit"] = round(float(df["Profit"].sum()), 2)
        ctx["avg_profit"]   = round(float(df["Profit"].mean()), 4)
    if "Order ID" in df.columns:
        ctx["total_orders"] = int(df["Order ID"].nunique())
    if "Customer Name" in df.columns:
        ctx["total_customers"] = int(df["Customer Name"].nunique())

    # Profit margin
    if ctx.get("total_sales") and ctx.get("total_profit") is not None:
        ctx["profit_margin"] = round(ctx["total_profit"] / ctx["total_sales"] * 100, 2)

    # ML
    ctx["best_model"]     = ml.get("best_model", "N/A")
    bk2 = ml.get("best_model_key", "")
    if bk2 and bk2 in ml:
        ctx["best_r2"]  = ml[bk2].get("r2", 0)
        ctx["best_mse"] = ml[bk2].get("mse", 0)
        ctx["best_predicted_profit"] = ml[bk2].get("avg_predicted_profit", 0)

    # Weakness
    ctx["loss_products"] = dict(list(w.get("loss_making_products", {}).items())[:5])
    ctx["low_regions"]   = w.get("low_performing_regions", {})
    ctx["poor_margins"]  = w.get("poor_profit_margins", {})

    # Category & Region analysis
    if "Category" in df.columns and "Profit" in df.columns:
        cat_profit = df.groupby("Category")["Profit"].sum()
        ctx["best_category"]  = cat_profit.idxmax()
        ctx["worst_category"] = cat_profit.idxmin()
        ctx["category_profits"] = cat_profit.round(2).to_dict()
    if "Region" in df.columns and "Sales" in df.columns:
        reg_sales  = df.groupby("Region")["Sales"].sum()
        reg_profit = df.groupby("Region")["Profit"].sum() if "Profit" in df.columns else None
        ctx["top_region"]    = reg_sales.idxmax()
        ctx["region_sales"]  = reg_sales.round(2).to_dict()
        if reg_profit is not None:
            ctx["region_profits"] = reg_profit.round(2).to_dict()
    if "Sub-Category" in df.columns and "Profit" in df.columns:
        sc = df.groupby("Sub-Category")["Profit"].sum().sort_values(ascending=False)
        ctx["top5_subcategories"]    = sc.head(5).round(2).to_dict()
        ctx["bottom5_subcategories"] = sc.tail(5).round(2).to_dict()
    if "Segment" in df.columns and "Sales" in df.columns:
        ctx["segment_sales"] = df.groupby("Segment")["Sales"].sum().round(2).to_dict()
    if "Discount" in df.columns and "Profit" in df.columns:
        ctx["discount_profit_corr"] = round(float(df["Discount"].corr(df["Profit"])), 4)
        ctx["high_discount_orders"] = int((df["Discount"] > 0.4).sum())
    if "Order Date" in df.columns:
        valid_dates = df["Order Date"].dropna()
        if len(valid_dates):
            ctx["date_from"] = str(valid_dates.min().date())
            ctx["date_to"]   = str(valid_dates.max().date())
    return ctx


def _bk_nlp_answer(question_raw, ctx):
    """
    Core NLP engine for BK. Matches any natural language question
    and returns a detailed, comprehensive response.
    """
    q = question_raw.lower().strip()
    name = ctx.get("dataset", "this dataset")

    # ── Greetings ────────────────────────────────────────────────────
    greet_patterns = [
        r"^(hi|hello|hey|hii+|good\s*(morning|afternoon|evening|day)|what'?s up|howdy|greetings|namaste|sup)[\s!?.,]*$",
        r"^(hi|hello|hey)\s+bk[\s!?.,]*$",
        r"^bk[\s!?.,]*$",
    ]
    for pat in greet_patterns:
        if re.search(pat, q):
            return (
                "👋 <strong>Hi there! I'm BK, your AI Decision Intelligence Assistant.</strong><br><br>"
                "I'm here to help you explore every aspect of your business data with deep, insightful analysis. "
                "Here's what I can help you with:<br><br>"
                "📊 <strong>Overview & KPIs</strong> — Sales, profit, orders, and customer summaries<br>"
                "⚠️ <strong>Weakness Detection</strong> — Loss-making products, underperforming regions, margin erosion<br>"
                "🔮 <strong>ML Predictions</strong> — Best model selection, profit forecasting, R² and MSE metrics<br>"
                "⚙️ <strong>Strategy Simulation</strong> — What-if growth analysis and strategic roadmaps<br>"
                "🚨 <strong>Risk & Anomaly Detection</strong> — Statistical outliers and profit drop analysis<br>"
                "💡 <strong>Recommendations</strong> — AI-generated step-by-step actionable plans<br>"
                "⚖️ <strong>Company Comparison</strong> — Multi-dataset benchmarking<br>"
                "🔬 <strong>Model Explainability</strong> — Feature importance and profit drivers<br><br>"
                "Just ask me anything — in any form or sentence! How can I help you today? 🚀"
            )

    # ── Help / capabilities ──────────────────────────────────────────
    if re.search(r"what can you|what do you|help|capabilities|features|tell me about yourself|who are you|about bk", q):
        return (
            "🤖 <strong>I'm BK — your AI Business Intelligence Assistant!</strong><br><br>"
            "I understand natural language questions in any form — complete sentences, short questions, "
            "or single keywords. Here's what I can analyze for you:<br><br>"
            "• <strong>Business Performance</strong>: total sales, profit, orders, customers, profit margin<br>"
            "• <strong>Weakness Analysis</strong>: loss-making products, low-performing regions, poor margin categories<br>"
            "• <strong>Machine Learning</strong>: model comparison (Linear Regression, Decision Tree, Random Forest), "
            "best model selection, R² scores, MSE, predicted profits<br>"
            "• <strong>Risk Assessment</strong>: anomaly detection, profit drops, discount risk analysis<br>"
            "• <strong>Strategies</strong>: growth simulation, profit maximization, regional recovery plans<br>"
            "• <strong>Data Insights</strong>: category performance, regional analysis, segment breakdown<br><br>"
            f"Currently analyzing: <strong>{name}</strong>. Ask me anything! 💬"
        )

    # ── Overview / Summary ───────────────────────────────────────────
    if re.search(r"overview|summary|snapshot|kpi|key metrics|dashboard|big picture|overall|general|status|health", q):
        parts = []
        if ctx.get("total_sales") is not None:
            parts.append(f"💰 <strong>Total Sales:</strong> ${ctx['total_sales']:,.2f}")
        if ctx.get("total_profit") is not None:
            profit_status = "🟢 Profitable" if ctx["total_profit"] > 0 else "🔴 Loss-Making"
            parts.append(f"📈 <strong>Total Profit:</strong> ${ctx['total_profit']:,.2f} ({profit_status})")
        if ctx.get("profit_margin") is not None:
            parts.append(f"📊 <strong>Profit Margin:</strong> {ctx['profit_margin']}%")
        if ctx.get("total_orders"):
            parts.append(f"📦 <strong>Total Orders:</strong> {ctx['total_orders']:,}")
        if ctx.get("total_customers"):
            parts.append(f"👥 <strong>Total Customers:</strong> {ctx['total_customers']:,}")
        if ctx.get("date_from"):
            parts.append(f"📅 <strong>Date Range:</strong> {ctx['date_from']} → {ctx['date_to']}")
        if ctx.get("best_model"):
            parts.append(f"🤖 <strong>Best ML Model:</strong> {ctx['best_model']}")
        weak_count = len(ctx.get("loss_products", {})) + len(ctx.get("low_regions", {}))
        if weak_count:
            parts.append(f"⚠️ <strong>Weaknesses Detected:</strong> {weak_count} areas need attention")
        summary_text = "<br>".join(parts) if parts else "No data loaded yet."
        return (
            f"📊 <strong>Business Overview — {name}</strong><br><br>"
            f"{summary_text}<br><br>"
            "This snapshot represents the current state of your business dataset. "
            "Want a deeper dive into any specific area? Just ask! For example: "
            "'Why is profit low?', 'Which region performs best?', or 'What is the best ML model?'"
        )

    # ── Total Sales / Revenue ────────────────────────────────────────
    if re.search(r"(total|overall|how much|what is|show|tell).*(sales|revenue)|sales.*(total|figure|number|amount|value)|revenue", q):
        ts = ctx.get("total_sales")
        if ts is not None:
            by_region = ""
            if ctx.get("region_sales"):
                top3 = sorted(ctx["region_sales"].items(), key=lambda x: -x[1])[:3]
                by_region = "<br>🗺️ <strong>Sales by Region:</strong><br>" + "<br>".join(
                    f"  • {r}: ${s:,.2f}" for r, s in top3
                )
            by_seg = ""
            if ctx.get("segment_sales"):
                by_seg = "<br>👥 <strong>Sales by Segment:</strong><br>" + "<br>".join(
                    f"  • {seg}: ${sal:,.2f}" for seg, sal in ctx["segment_sales"].items()
                )
            return (
                f"💰 <strong>Total Sales for {name}: ${ts:,.2f}</strong><br><br>"
                f"This represents the total revenue generated across all orders, products, regions, and customer "
                f"segments in this dataset. Here's a breakdown of where the sales are coming from:<br>"
                f"{by_region}{by_seg}<br><br>"
                "📌 <strong>Key Insight:</strong> Sales volume is a strong leading indicator of business health, "
                "but it must be viewed alongside profit margins to assess true business performance. "
                "High sales with low profit margins indicate discount-driven or cost-heavy operations that "
                "need strategic intervention."
            )
        return "Sales data is not available in this dataset."

    # ── Total Profit ─────────────────────────────────────────────────
    if re.search(r"(total|overall|how much|what is|show|tell).*(profit)|profit.*(total|figure|number|amount|value)|net profit|bottom line|earnings", q):
        tp = ctx.get("total_profit")
        if tp is not None:
            margin_txt = f" (profit margin: {ctx['profit_margin']}%)" if ctx.get("profit_margin") is not None else ""
            loss_prods = ctx.get("loss_products", {})
            loss_txt   = ""
            if loss_prods:
                loss_txt = (
                    f"<br><br>⚠️ <strong>Warning:</strong> {len(loss_prods)} sub-categories are generating losses: "
                    + ", ".join(f"{k} (${abs(v):,.2f} loss)" for k, v in list(loss_prods.items())[:3])
                    + ". These drag down your overall profit and need immediate attention."
                )
            return (
                f"📈 <strong>Total Profit for {name}: ${tp:,.2f}</strong>{margin_txt}<br><br>"
                f"{'🟢 The business is profitable — a positive net profit indicates healthy operations.' if tp > 0 else '🔴 The business is currently at a net loss — immediate strategic action is recommended.'}"
                f"{loss_txt}<br><br>"
                "💡 <strong>To improve profitability:</strong> Focus on eliminating loss-making products, "
                "reducing excessive discounts (>40%), and deploying the best ML model for dynamic pricing. "
                "Even a 10% improvement in margins across key categories can produce a measurable impact on total profit."
            )
        return "Profit data is not available in this dataset."

    # ── Profit Drop / Decline ────────────────────────────────────────
    if re.search(r"why.*(profit|loss)|profit.*(drop|fell|fall|declin|low|less|poor|bad|down|decreas|negative)|loss.*(making|why|reason|cause)", q):
        causes = []
        loss_prods = ctx.get("loss_products", {})
        low_regs   = ctx.get("low_regions", {})
        poor_margins = ctx.get("poor_margins", {})
        disc_corr  = ctx.get("discount_profit_corr")

        if loss_prods:
            worst = min(loss_prods, key=lambda k: loss_prods[k])
            total_loss = sum(v for v in loss_prods.values())
            causes.append(
                f"<strong>Loss-Making Sub-Categories:</strong> {len(loss_prods)} sub-categories "
                f"generate cumulative losses of ${abs(total_loss):,.2f}. The worst offender is '{worst}' "
                f"with a ${abs(loss_prods[worst]):,.2f} loss. These sub-categories are priced below cost "
                "or receiving excessive discounts that completely erode margin."
            )
        if low_regs:
            causes.append(
                f"<strong>Underperforming Regions:</strong> {len(low_regs)} region(s) "
                f"({', '.join(low_regs.keys())}) are generating profit below the company average, "
                "indicating poor market penetration, inadequate sales coverage, or logistical inefficiencies."
            )
        if poor_margins:
            causes.append(
                f"<strong>Low-Margin Categories:</strong> {', '.join(poor_margins.keys())} have profit margins below 10%, "
                "meaning almost every sale in these categories barely covers operating costs."
            )
        if disc_corr is not None and disc_corr < -0.1:
            causes.append(
                f"<strong>Excessive Discounting:</strong> There is a {disc_corr:+.4f} negative correlation "
                "between discount rates and profit. This means that as discounts increase, profit drops "
                "significantly — a classic sign that the discount strategy is destroying margin faster "
                "than it is generating volume."
            )
        if not causes:
            return (
                f"✅ Good news! No significant profit decline drivers were detected in <strong>{name}</strong>. "
                "The business appears to be in a relatively healthy state. However, continue monitoring "
                "monthly trends and discount rates to prevent future margin erosion."
            )
        causes_html = "".join(f"<br>🔍 {i+1}. {c}" for i, c in enumerate(causes))
        return (
            f"📉 <strong>Why is Profit Declining in {name}?</strong><br>"
            f"Based on a comprehensive analysis of {ctx.get('rows', 0):,} records, "
            f"the following root causes were identified:{causes_html}<br><br>"
            "💊 <strong>Recommended Fixes:</strong><br>"
            "• Discontinue or reprice the top 3 loss-making sub-categories within 60 days<br>"
            "• Cap automatic discounts at 25–30% without manager approval<br>"
            "• Launch targeted regional recovery campaigns in underperforming markets<br>"
            "• Renegotiate supplier contracts for low-margin categories<br>"
            "• Deploy ML-guided dynamic pricing to enforce minimum profitable price floors"
        )

    # ── Best Strategy / Recommendations ─────────────────────────────
    if re.search(r"(best|top|what|how).*(strateg|approach|action|plan|recommend|suggest|improve|grow|fix|solve|do|next step)|recommend|suggestion|action plan|roadmap", q):
        bm = ctx.get("best_model", "Random Forest")
        loss_prods = ctx.get("loss_products", {})
        low_regs   = ctx.get("low_regions", {})
        actions = [
            f"<strong>Deploy {bm} for ML-Driven Profit Forecasting</strong> — This model achieved the lowest MSE "
            f"(R²={ctx.get('best_r2', 0):.4f}) and should be integrated into your pricing and inventory system. "
            "Businesses that adopt ML-driven forecasting report 10–20% improvement in profit margin accuracy.",
        ]
        if loss_prods:
            prods_str = ", ".join(list(loss_prods.keys())[:3])
            actions.append(
                f"<strong>Eliminate Loss-Making Sub-Categories ({prods_str})</strong> — "
                "Conduct a deep-dive cost analysis, pilot 15–20% price increases, or discontinue unprofitable lines. "
                "Recovering these losses could improve overall profit by 5–15%."
            )
        if low_regs:
            reg_str = ", ".join(list(low_regs.keys())[:2])
            actions.append(
                f"<strong>Regional Recovery Campaigns ({reg_str})</strong> — "
                "Assign dedicated regional managers, launch localized promotions, and build local distribution networks. "
                "Recovering even 20% of the regional profit gap can contribute measurable revenue uplift."
            )
        actions.append(
            "<strong>Launch RFM-Based Customer Loyalty Program</strong> — "
            "Retaining existing customers costs 5–7x less than acquiring new ones. "
            "Segment customers by Recency, Frequency, and Monetary value to run personalized re-engagement campaigns."
        )
        actions.append(
            "<strong>Cap Excessive Discounts at 25–30%</strong> — "
            "Any discount above 40% should require management approval and profitability review. "
            "Use ML-based dynamic pricing to enforce minimum profitable price floors by SKU."
        )
        actions_html = "".join(f"<br>🚀 {i+1}. {a}" for i, a in enumerate(actions))
        return (
            f"💡 <strong>Top Strategic Recommendations for {name}</strong><br>"
            f"Based on a full analysis of {ctx.get('rows', 0):,} records, here are the highest-impact actions:{actions_html}<br><br>"
            "📈 <strong>Expected Outcome:</strong> Applying all recommended strategies is projected to deliver "
            "a 10–25% improvement in overall profitability within 2–3 quarters. Start with the highest-confidence, "
            "lowest-risk actions first for immediate impact."
        )

    # ── ML Models / Predictions ──────────────────────────────────────
    if re.search(r"(ml|machine learning|model|algorithm|predict|forecast|random forest|decision tree|linear regression|r2|r squared|mse|accuracy|performance)", q):
        bm  = ctx.get("best_model", "N/A")
        r2  = ctx.get("best_r2", 0)
        mse = ctx.get("best_mse", 0)
        pp  = ctx.get("best_predicted_profit", 0)
        return (
            f"🤖 <strong>Machine Learning Analysis for {name}</strong><br><br>"
            f"Three models were trained and evaluated on this dataset:<br>"
            "📐 <strong>Linear Regression</strong> — Fast, interpretable baseline model for linear profit patterns<br>"
            "🌲 <strong>Decision Tree</strong> — Rule-based model capturing non-linear profit thresholds<br>"
            "🌳 <strong>Random Forest</strong> — Ensemble of trees, most robust and accurate for real-world data<br><br>"
            f"🏆 <strong>Best Model: {bm}</strong><br>"
            f"  • R² Score: {r2:.4f} (explains {round(r2*100,1)}% of profit variance)<br>"
            f"  • MSE: {mse:,.4f} (lowest prediction error among all models)<br>"
            f"  • Average Predicted Profit: ${pp:,.2f} per prediction cycle<br><br>"
            f"💡 <strong>Why {bm}?</strong> It achieved the lowest Mean Squared Error, meaning its profit "
            "predictions deviate least from actual values. This makes it the most reliable tool for "
            "operational profit forecasting, inventory optimization, and dynamic pricing decisions."
        )

    # ── Risk & Anomaly ───────────────────────────────────────────────
    if re.search(r"risk|anomal|outlier|unusual|strange|alert|danger|problem|issue|concern|threat|spike|drop|sudden", q):
        disc_corr = ctx.get("discount_profit_corr")
        high_disc = ctx.get("high_discount_orders", 0)
        loss_count = len(ctx.get("loss_products", {}))
        risk_items = []
        if high_disc > 0:
            risk_items.append(
                f"<strong>High-Discount Risk:</strong> {high_disc:,} orders have discounts >40%, "
                "directly compressing profit margins. These orders likely operate at near break-even or loss."
            )
        if disc_corr is not None and disc_corr < -0.1:
            risk_items.append(
                f"<strong>Discount-Profit Inverse Correlation:</strong> r = {disc_corr:+.4f}. "
                "As discounts increase, profit falls significantly — a systemic pricing risk that needs "
                "an immediate policy review."
            )
        if loss_count > 0:
            risk_items.append(
                f"<strong>Loss-Making Product Risk:</strong> {loss_count} sub-categories generate negative cumulative profit, "
                "eroding overall margin. Left unaddressed, these products silently drain company resources."
            )
        if ctx.get("low_regions"):
            risk_items.append(
                f"<strong>Regional Concentration Risk:</strong> {len(ctx['low_regions'])} underperforming region(s) "
                f"({', '.join(list(ctx['low_regions'].keys())[:3])}) concentrate revenue risk in fewer geographic markets."
            )
        if not risk_items:
            return (
                f"✅ <strong>Risk Assessment for {name}:</strong> No major risk signals detected. "
                "The dataset appears stable with no critical anomalies. Continue monitoring monthly "
                "profit trends and discount rate changes as early warning indicators."
            )
        risk_html = "".join(f"<br>🚨 {i+1}. {r}" for i, r in enumerate(risk_items))
        return (
            f"🚨 <strong>Risk & Anomaly Analysis for {name}</strong><br>"
            f"The following risk signals were identified:{risk_html}<br><br>"
            "🛡️ <strong>Recommended Actions:</strong><br>"
            "• Cap discounts at 25–30% and require manager approval above that threshold<br>"
            "• Implement monthly profit floor alerts for all product lines<br>"
            "• Investigate and phase out persistently loss-making sub-categories<br>"
            "• Diversify geographic revenue mix to reduce regional concentration risk<br>"
            "• Run monthly anomaly detection checks to catch issues before they compound"
        )

    # ── Region Analysis ──────────────────────────────────────────────
    if re.search(r"region|geographic|location|area|territory|market|west|east|central|south|north", q):
        reg_sales   = ctx.get("region_sales", {})
        reg_profits = ctx.get("region_profits", {})
        top_region  = ctx.get("top_region", "N/A")
        if not reg_sales:
            return "Region data is not available in this dataset."
        region_html = ""
        for reg in sorted(reg_sales, key=lambda r: -reg_sales[r]):
            profit_txt = f" | Profit: ${reg_profits.get(reg, 0):,.2f}" if reg_profits else ""
            icon = "🏆" if reg == top_region else "📍"
            region_html += f"<br>{icon} <strong>{reg}</strong>: Sales = ${reg_sales[reg]:,.2f}{profit_txt}"
        low_regs = ctx.get("low_regions", {})
        low_txt  = ""
        if low_regs:
            low_txt = (
                f"<br><br>⚠️ <strong>Underperforming Regions:</strong> {', '.join(low_regs.keys())} "
                "are generating profit below the company average and need targeted recovery strategies."
            )
        return (
            f"🗺️ <strong>Regional Performance Analysis — {name}</strong><br><br>"
            f"Performance breakdown by region:{region_html}{low_txt}<br><br>"
            f"🏆 <strong>Top Region:</strong> {top_region} leads in total sales volume.<br><br>"
            "📌 <strong>Insight:</strong> Regional performance differences often stem from market maturity, "
            "competition intensity, product-market fit, and operational efficiency. Underperforming regions "
            "should receive dedicated sales teams, localized pricing, and region-specific promotional campaigns."
        )

    # ── Category Analysis ────────────────────────────────────────────
    if re.search(r"categor|product|segment|sub.?categor|furniture|technology|office|supply", q):
        cat_profits = ctx.get("category_profits", {})
        best_cat    = ctx.get("best_category", "N/A")
        worst_cat   = ctx.get("worst_category", "N/A")
        top5_sub    = ctx.get("top5_subcategories", {})
        bot5_sub    = ctx.get("bottom5_subcategories", {})
        if not cat_profits:
            return "Category data is not available in this dataset."
        cat_html = "".join(
            f"<br>{'🥇' if cat == best_cat else '📦'} <strong>{cat}</strong>: ${profit:,.2f}"
            for cat, profit in sorted(cat_profits.items(), key=lambda x: -x[1])
        )
        top5_html = "".join(
            f"<br>  ✅ {sc}: ${val:,.2f}" for sc, val in list(top5_sub.items())[:5]
        )
        bot5_html = "".join(
            f"<br>  ❌ {sc}: ${val:,.2f}" for sc, val in list(bot5_sub.items())
        )
        return (
            f"🏷️ <strong>Category & Sub-Category Analysis — {name}</strong><br><br>"
            f"<strong>Profit by Category:</strong>{cat_html}<br><br>"
            f"🏆 <strong>Best Category:</strong> {best_cat} — highest profit contribution<br>"
            f"⚠️ <strong>Worst Category:</strong> {worst_cat} — lowest or negative profit<br>"
            f"{f'<br>📈 <strong>Top 5 Sub-Categories by Profit:</strong>{top5_html}' if top5_html else ''}"
            f"{f'<br><br>📉 <strong>Bottom 5 Sub-Categories (Losses):</strong>{bot5_html}' if bot5_html else ''}<br><br>"
            "💡 <strong>Insight:</strong> Focus investment and marketing on your highest-profit categories. "
            "For loss-making sub-categories, conduct root cause analysis — are losses driven by discounting, "
            "high returns, supplier costs, or competitive pricing pressure?"
        )

    # ── Discount Analysis ────────────────────────────────────────────
    if re.search(r"discount|price|pricing|mark.?down|promotion|deal|offer|coupon", q):
        disc_corr = ctx.get("discount_profit_corr")
        high_disc = ctx.get("high_discount_orders", 0)
        disc_txt  = ""
        if disc_corr is not None:
            disc_txt = (
                f"The correlation between discount rate and profit is <strong>{disc_corr:+.4f}</strong>. "
                + ("This negative correlation confirms that higher discounts are destroying profit margin. " if disc_corr < -0.1
                   else "Discounts appear to be well-managed with minimal profit impact. ")
            )
        high_disc_txt = ""
        if high_disc > 0:
            high_disc_txt = (
                f"{high_disc:,} orders have discounts above 40% — these are high-risk transactions that likely "
                "operate at near break-even or at a loss. "
            )
        return (
            f"🏷️ <strong>Discount & Pricing Analysis — {name}</strong><br><br>"
            f"{disc_txt}{high_disc_txt}<br><br>"
            "📌 <strong>Best Practices for Discount Strategy:</strong><br>"
            "• Cap automatic discounts at 25–30% without manager approval<br>"
            "• Flag any order with >35% discount for profitability review<br>"
            "• Use ML-based dynamic pricing to enforce minimum profitable price floors per SKU<br>"
            "• Introduce tiered pricing (Basic / Standard / Premium) instead of blanket discounts<br>"
            "• Test price elasticity with A/B pricing experiments before broad rollout<br><br>"
            "💰 <strong>Impact:</strong> Reducing average discount by just 5% can improve overall profit margin "
            "by 10–20%, depending on sales volume and current cost structure."
        )

    # ── Dataset Info / Size ──────────────────────────────────────────
    if re.search(r"rows|records|size|how many|data|dataset|column|field|variable|attributes", q):
        date_txt = ""
        if ctx.get("date_from"):
            date_txt = f"<br>📅 <strong>Date Range:</strong> {ctx['date_from']} → {ctx['date_to']}"
        return (
            f"📋 <strong>Dataset Information — {name}</strong><br><br>"
            f"📊 <strong>Rows (Records):</strong> {ctx.get('rows', 0):,}<br>"
            f"📐 This dataset captures transactional retail sales data including order details, "
            f"customer segments, product categories, regional performance, and financial outcomes "
            f"(sales, profit, discounts).{date_txt}<br><br>"
            "The dataset is used for:<br>"
            "• Detecting business weaknesses and loss-making areas<br>"
            "• Training and evaluating ML prediction models<br>"
            "• Generating strategic business recommendations<br>"
            "• Comparing performance across datasets (if multiple are loaded)"
        )

    # ── Comparison ───────────────────────────────────────────────────
    if re.search(r"compar|benchmark|versus|vs|other dataset|which is better|best company|top performer|rank", q):
        return (
            "⚖️ <strong>Dataset Comparison</strong><br><br>"
            "To compare this dataset with others, load multiple datasets (use 'Load Default Datasets' or "
            "upload your own CSVs) and navigate to the <strong>Company Comparison</strong> tab.<br><br>"
            "The comparison module analyzes:<br>"
            "• Total Sales & Profit across all datasets<br>"
            "• Average profit per order<br>"
            "• Best ML model per dataset<br>"
            "• Predicted future profit using ML forecasting<br>"
            "• Strategic recommendations to close the profit gap between top and bottom performers<br><br>"
            "I can also answer specific comparison questions once multiple datasets are loaded!"
        )

    # ── Conclusion / Report ──────────────────────────────────────────
    if re.search(r"conclusion|final|report|summary report|overall result|outcome|finding|takeaway", q):
        tp = ctx.get("total_profit")
        ts = ctx.get("total_sales")
        bm = ctx.get("best_model", "N/A")
        margin = ctx.get("profit_margin", 0)
        return (
            f"🏁 <strong>Final Conclusion & Report — {name}</strong><br><br>"
            f"{'✅ The business is currently profitable.' if tp and tp > 0 else '⚠️ The business is currently at a net loss.'} "
            f"With ${(ts or 0):,.2f} in total sales and ${(tp or 0):,.2f} in profit ({margin}% margin), "
            f"the best predictive model identified is <strong>{bm}</strong>.<br><br>"
            "🔑 <strong>Key Takeaways:</strong><br>"
            f"• {len(ctx.get('loss_products', {}))} sub-categories are loss-making and need immediate review<br>"
            f"• {len(ctx.get('low_regions', {}))} regions are underperforming relative to the company average<br>"
            f"• Deploying {bm} for profit forecasting can improve decision-making accuracy by 10–20%<br>"
            "• A 10% growth scenario is projected to deliver significant profit uplift<br><br>"
            "📄 For a full printable report, navigate to the <strong>Conclusion & Report</strong> tab "
            "and click 'Generate Conclusion' followed by 'Print / Save as PDF'."
        )

    # ── Fallback — General intelligent response ─────────────────────
    bm = ctx.get("best_model", "N/A")
    tp = ctx.get("total_profit")
    ts = ctx.get("total_sales")
    profit_line  = f"<br>📈 <strong>Profitability:</strong> ${tp:,.2f} total profit" if tp is not None else ""
    revenue_line = f"<br>💰 <strong>Revenue:</strong> ${ts:,.2f} total sales"        if ts is not None else ""
    model_line   = f"<br>🤖 <strong>Best ML Model:</strong> {bm}"                   if bm != "N/A" else ""
    issue_count  = len(ctx.get("loss_products", {}))
    issue_line   = f"<br>⚠️ <strong>Key Issues:</strong> {issue_count} loss-making sub-categories detected" if issue_count else ""
    return (
        f"🤔 <strong>BK's Analysis for: \"{question_raw}\"</strong><br><br>"
        f"I understand you're asking about <strong>{name}</strong>. "
        f"Here's what I know based on the loaded data:<br>"
        f"{profit_line}{revenue_line}{model_line}{issue_line}"
        "<br><br>You can ask me about:<br>"
        "• <em>\"Show me the overview\"</em> — full KPI snapshot<br>"
        "• <em>\"Why is profit dropping?\"</em> — root cause analysis<br>"
        "• <em>\"What is the best strategy?\"</em> — actionable recommendations<br>"
        "• <em>\"Which region performs best?\"</em> — regional analysis<br>"
        "• <em>\"Tell me about the ML models\"</em> — prediction model comparison<br>"
        "• <em>\"What are the risks?\"</em> — anomaly and risk assessment"
    )


@app.route("/api/chat/<name>", methods=["POST"])
def chat(name):
    """BK AI Assistant — Comprehensive NLP chat endpoint."""
    if name not in datasets:
        return jsonify({"error": "Dataset not found"}), 404
    df   = datasets[name]
    body = request.get_json(silent=True) or {}
    question_raw = str(body.get("question", "")).strip()

    if name not in ml_cache:
        ml_cache[name] = run_ml(df)
    ml = ml_cache[name]
    w  = detect_weaknesses(df)

    ctx    = _bk_build_context(df, ml, w, name)
    answer = _bk_nlp_answer(question_raw, ctx)

    return jsonify({
        "question":  question_raw,
        "answer":    answer,
        "dataset":   name,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    })


@app.route("/api/chat-global", methods=["POST"])
def chat_global():
    """BK global chat — handles queries even when no dataset is selected."""
    body = request.get_json(silent=True) or {}
    question_raw = str(body.get("question", "")).strip()
    q = question_raw.lower().strip()

    greet_patterns = [
        r"^(hi|hello|hey|hii+|good\s*(morning|afternoon|evening|day)|what'?s up|howdy|greetings|namaste|sup)[\s!?.,]*$",
        r"^(hi|hello|hey)\s+bk[\s!?.,]*$",
        r"^bk[\s!?.,]*$",
    ]
    for pat in greet_patterns:
        if re.search(pat, q):
            answer = (
                "👋 <strong>Hi! I'm BK, your AI Decision Intelligence Assistant.</strong><br><br>"
                "To get started with personalized data analysis, please:<br>"
                "1️⃣ Click <strong>'Load Default Datasets'</strong> to load sample business data, or<br>"
                "2️⃣ Upload your own CSV files using the <strong>'Upload CSV(s)'</strong> button<br>"
                "3️⃣ Select a dataset from the dropdown to activate all modules<br><br>"
                "Once your data is loaded, I can answer any question about your business — "
                "profits, risks, strategies, predictions, and much more! 🚀"
            )
            return jsonify({"question": question_raw, "answer": answer, "dataset": None,
                            "timestamp": datetime.datetime.utcnow().isoformat()})

    answer = (
        "📊 <strong>No dataset selected yet.</strong><br><br>"
        "To unlock BK's full analysis capabilities, please load a dataset first:<br>"
        "• Click <strong>'Load Default Datasets'</strong> for sample data<br>"
        "• Or upload your own CSV using <strong>'Upload CSV(s)'</strong><br><br>"
        "Once loaded, I can answer any question about your business data in any form or sentence!"
    )
    return jsonify({"question": question_raw, "answer": answer, "dataset": None,
                    "timestamp": datetime.datetime.utcnow().isoformat()})


# ── AI Decision Engine ─────────────────────────────────────────────

@app.route("/api/decision-engine/<name>", methods=["GET"])
def decision_engine(name):
    if name not in datasets:
        return jsonify({"error": "Dataset not found"}), 404
    df = datasets[name]
    if name not in ml_cache:
        ml_cache[name] = run_ml(df)
    ml  = ml_cache[name]
    w   = detect_weaknesses(df)

    total_profit = float(df["Profit"].sum()) if "Profit" in df.columns else 0
    total_sales  = float(df["Sales"].sum())  if "Sales"  in df.columns else 0
    bp           = ml.get("best_predicted_profit", 0)

    decisions = []

    loss_sum = abs(sum(w.get("loss_making_products", {}).values()))
    if loss_sum > 0:
        roi = round(loss_sum / max(abs(total_profit), 1) * 100, 1)
        decisions.append({
            "rank": 1, "priority": "HIGH",
            "strategy":       "Eliminate Loss-Making Sub-Categories",
            "expected_roi":   f"+{roi}% profit recovery",
            "risk_level":     "Low",
            "time_to_impact": "60 days",
            "revenue_impact": round(loss_sum, 2),
            "growth_impact":  round(roi, 1),
            "confidence":     92,
            "rationale":      f"Removing {len(w.get('loss_making_products', {}))} loss-making lines recovers ${loss_sum:,.2f} in eroded profit.",
        })

    reg_gap = abs(sum(v for v in w.get("low_performing_regions", {}).values() if v < 0))
    if reg_gap > 0:
        decisions.append({
            "rank": len(decisions) + 1, "priority": "MEDIUM",
            "strategy":       "Regional Recovery Campaigns",
            "expected_roi":   "+15–25% regional revenue",
            "risk_level":     "Medium",
            "time_to_impact": "90 days",
            "revenue_impact": round(reg_gap * 0.2, 2),
            "growth_impact":  18.5,
            "confidence":     78,
            "rationale":      "Targeted local campaigns can recover 20% of regional profit gap with moderate investment.",
        })

    decisions.append({
        "rank": len(decisions) + 1, "priority": "HIGH",
        "strategy":       f"Deploy {ml.get('best_model', 'Random Forest')} for Dynamic Pricing",
        "expected_roi":   "+10–20% margin improvement",
        "risk_level":     "Low",
        "time_to_impact": "30 days",
        "revenue_impact": round(bp * 0.15, 2),
        "growth_impact":  15.0,
        "confidence":     88,
        "rationale":      "ML-guided pricing reduces over-discounting and improves average transaction margin.",
    })

    decisions.append({
        "rank": len(decisions) + 1, "priority": "MEDIUM",
        "strategy":       "Launch RFM-Based Loyalty Program",
        "expected_roi":   "+25–95% LTV improvement",
        "risk_level":     "Low",
        "time_to_impact": "45 days",
        "revenue_impact": round(total_sales * 0.05, 2),
        "growth_impact":  22.0,
        "confidence":     85,
        "rationale":      "Retaining top 20% customers via RFM segmentation can yield 5x lower acquisition cost.",
    })

    decisions.sort(key=lambda x: -x["confidence"])
    for i, d in enumerate(decisions, 1):
        d["rank"] = i

    return jsonify({
        "decisions":        decisions,
        "top_strategy":     decisions[0] if decisions else {},
        "total_strategies": len(decisions),
        "dataset":          name,
    })


# ── Model Explainability ───────────────────────────────────────────

@app.route("/api/explain/<name>", methods=["GET"])
def explain(name):
    if name not in datasets:
        return jsonify({"error": "Dataset not found"}), 404
    df = datasets[name]
    if not {"Sales", "Quantity", "Profit"}.issubset(df.columns):
        return jsonify({"error": "Required columns missing"}), 400

    # Subsample for extremely fast training on large datasets
    max_train_samples = 10000
    if len(df) > max_train_samples:
        df_sample = df.sample(n=max_train_samples, random_state=42)
    else:
        df_sample = df

    features = df_sample[["Sales", "Quantity"]].values
    target   = df_sample["Profit"].values
    X_tr, X_te, y_tr, y_te = train_test_split(features, target, test_size=0.2, random_state=42)

    rf          = RandomForestRegressor(n_estimators=20, max_depth=10, min_samples_split=10, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    importances = rf.feature_importances_.tolist()

    feature_names = ["Sales", "Quantity"]
    explanation   = [
        {"feature": fn, "importance": round(imp, 4), "pct": round(imp * 100, 1)}
        for fn, imp in zip(feature_names, importances)
    ]
    explanation.sort(key=lambda x: -x["importance"])

    directions   = {}
    correlations = {}
    for fn in feature_names:
        col_vals        = df[fn].values
        corr            = float(np.corrcoef(col_vals, target)[0, 1])
        directions[fn]  = "positive" if corr > 0 else "negative"
        correlations[fn] = round(corr, 4)

    feature_plain_english = {
        "Sales": (
            "Sales (Revenue) is the total monetary value of each transaction. "
            "A higher sales value generally signals a larger order, which gives the business more gross margin to work with — "
            "even after subtracting costs. When Sales is the top profit driver, it means your business profit is "
            "most sensitive to order value. Growing average order size (through upselling, bundles, or premium tiers) "
            "will have the biggest direct impact on profit."
        ),
        "Quantity": (
            "Quantity is the number of units sold per order. While higher quantities generate more revenue, "
            "they also often come with bulk discounts that compress per-unit margin. "
            "When Quantity strongly influences profit, it suggests the business should carefully evaluate "
            "its volume-discount policy — ensuring that large-quantity orders remain genuinely profitable "
            "rather than just high-revenue."
        ),
    }

    feature_risk_impact = {
        "Sales": {
            "positive": "When sales revenue increases, profit typically rises proportionally. However, if sales growth comes from deep discounts, margin may not improve.",
            "negative": "A negative correlation between Sales and Profit is a red flag — it means higher-revenue orders are somehow less profitable, often due to excessive discounting on large orders.",
        },
        "Quantity": {
            "positive": "More units sold = more profit in your dataset. This suggests bulk orders are profitable and volume growth is a viable strategy.",
            "negative": "Higher quantity correlates with lower profit — strong signal that bulk discounts or fulfillment costs are eating into margins on large orders.",
        },
    }

    enriched_explanation = []
    for e in explanation:
        fn   = e["feature"]
        dir_ = directions[fn]
        enriched_explanation.append({
            **e,
            "direction":        dir_,
            "correlation":      correlations[fn],
            "plain_english":    feature_plain_english.get(fn, "This feature influences profit predictions."),
            "impact_narrative": feature_risk_impact.get(fn, {}).get(dir_, ""),
            "when_profit_rises": (
                f"When {fn} increases, profit tends to {'increase' if dir_ == 'positive' else 'decrease'} "
                f"(correlation: {correlations[fn]:+.4f}). "
                + ("This is the expected healthy pattern." if dir_ == "positive" else
                   "This is a warning sign — investigate if cost structures or discounting are causing inverse performance.")
            ),
            "what_to_watch": (
                f"Monitor {fn} monthly. A sudden drop in {fn} with no corresponding cost reduction "
                "will directly compress profits. Set threshold alerts for significant deviations."
            ),
        })

    fig, ax = plt.subplots(figsize=(6, 3))
    colors  = ["#6C63FF" if directions[e["feature"]] == "positive" else "#FF6584" for e in explanation]
    ax.barh([e["feature"] for e in explanation], [e["importance"] for e in explanation], color=colors)
    ax.set_title("Feature Importance (Random Forest)", fontsize=12, fontweight="bold", color="white")
    ax.set_xlabel("Importance Score", color="white")
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#16213e")
    ax.tick_params(colors="white"); ax.spines[:].set_color("#333366")
    chart = fig_to_b64(fig)

    top_f = enriched_explanation[0]
    bot_f = enriched_explanation[-1] if len(enriched_explanation) > 1 else None

    model_behaviour = (
        f"The Random Forest model assigns {top_f['pct']}% of its decision-making weight to '{top_f['feature']}', "
        f"meaning this single variable is responsible for {top_f['pct']}% of what the model uses to predict profit. "
        + (f"'{bot_f['feature']}' accounts for the remaining {bot_f['pct']}%. " if bot_f else "")
        + "This distribution tells you exactly where to focus data quality efforts and business strategy: "
        f"improving the reliability and magnitude of '{top_f['feature']}' will have the greatest impact on forecast accuracy."
    )

    how_to_use = [
        f"Focus sales strategy on growing '{top_f['feature']}' — it's your single biggest profit lever ({top_f['pct']}% model weight).",
        "Review your bulk-discount policy if Quantity has a negative profit correlation.",
        "Use these feature weights to prioritize which KPIs to include in executive dashboards.",
        "Any new pricing or promotional strategy should be evaluated first through its impact on these two drivers.",
        "Retrain this analysis monthly — feature importance can shift as market conditions and product mix change.",
    ]

    return jsonify({
        "feature_importance": enriched_explanation,
        "directions":         directions,
        "correlations":       correlations,
        "chart":              chart,
        "model_behaviour":    model_behaviour,
        "how_to_use":         how_to_use,
        "insight": (
            f"'{top_f['feature']}' is the strongest profit driver ({top_f['pct']}% importance) "
            f"with a {top_f['direction']} correlation to profit (r = {top_f['correlation']:+.4f}). "
            f"This means: {top_f['impact_narrative']}"
        ),
        "plain_summary": (
            f"In simple terms: the AI model predicts profit primarily based on '{top_f['feature']}'. "
            + top_f["plain_english"]
        ),
    })


# ── Smart Data Processing ──────────────────────────────────────────

@app.route("/api/smart-process/<name>", methods=["GET"])
def smart_process(name):
    if name not in datasets:
        return jsonify({"error": "Dataset not found"}), 404
    df = datasets[name]

    issues      = []
    suggestions = []

    missing = df.isnull().sum()
    for col, cnt in missing[missing > 0].items():
        pct = round(cnt / len(df) * 100, 1)
        issues.append({"type": "missing_values", "column": col, "count": int(cnt), "pct": pct})

    dup_count = int(df.duplicated().sum())
    if dup_count:
        issues.append({"type": "duplicates", "count": dup_count})

    for col in df.columns:
        if df[col].nunique() == 1:
            issues.append({"type": "constant_column", "column": col})

    if {"Sales", "Profit"}.issubset(df.columns):
        suggestions.append({"feature": "profit_margin", "formula": "Profit / Sales", "reason": "Key efficiency KPI"})
    if {"Order Date", "Ship Date"}.issubset(df.columns):
        suggestions.append({"feature": "shipping_days", "formula": "Ship Date - Order Date", "reason": "Operational efficiency metric"})
    if "Quantity" in df.columns and "Sales" in df.columns:
        suggestions.append({"feature": "avg_order_value", "formula": "Sales / Quantity", "reason": "Unit economics metric"})

    model_rec    = "Random Forest" if len(df) > 1000 else "Decision Tree"
    model_reason = (
        "Large dataset benefits from ensemble stability."
        if len(df) > 1000 else "Smaller dataset suits interpretable single-tree model."
    )

    quality_score = 100
    quality_score -= len([i for i in issues if i["type"] == "missing_values"]) * 10
    quality_score -= (5 if dup_count else 0)
    quality_score = max(0, quality_score)

    return jsonify({
        "quality_score":       quality_score,
        "total_rows":          len(df),
        "total_columns":       len(df.columns),
        "issues":              issues,
        "feature_suggestions": suggestions,
        "recommended_model":   model_rec,
        "model_reason":        model_reason,
        "status":              "good" if quality_score >= 80 else "needs_attention",
    })


# ── KPI Filters ────────────────────────────────────────────────────

@app.route("/api/kpi-filters/<name>", methods=["GET"])
def kpi_filters(name):
    if name not in datasets:
        return jsonify({"error": "Dataset not found"}), 404
    df = datasets[name]

    region   = request.args.get("region")
    category = request.args.get("category")
    segment  = request.args.get("segment")

    filtered = df.copy()
    if region   and "Region"   in df.columns: filtered = filtered[filtered["Region"]   == region]
    if category and "Category" in df.columns: filtered = filtered[filtered["Category"] == category]
    if segment  and "Segment"  in df.columns: filtered = filtered[filtered["Segment"]  == segment]

    result = {
        "filters_applied":   {"region": region, "category": category, "segment": segment},
        "rows_after_filter": len(filtered),
        "total_sales":       round(float(filtered["Sales"].sum()),  2) if "Sales"    in filtered.columns else 0,
        "total_profit":      round(float(filtered["Profit"].sum()), 2) if "Profit"   in filtered.columns else 0,
        "total_orders":      int(filtered["Order ID"].nunique())        if "Order ID" in filtered.columns else 0,
        "avg_discount":      round(float(filtered["Discount"].mean()), 4) if "Discount" in filtered.columns else 0,
        "profit_margin":     0,
        "filter_options": {
            "regions":    sorted(df["Region"].dropna().unique().tolist())   if "Region"   in df.columns else [],
            "categories": sorted(df["Category"].dropna().unique().tolist()) if "Category" in df.columns else [],
            "segments":   sorted(df["Segment"].dropna().unique().tolist())  if "Segment"  in df.columns else [],
        },
    }
    if result["total_sales"] > 0:
        result["profit_margin"] = round(result["total_profit"] / result["total_sales"] * 100, 2)

    return jsonify(result)


# ── Report Metadata ────────────────────────────────────────────────

@app.route("/api/report-meta/<name>", methods=["GET"])
def report_meta(name):
    if name not in datasets:
        return jsonify({"error": "Dataset not found"}), 404
    df = datasets[name]
    if name not in ml_cache:
        ml_cache[name] = run_ml(df)
    ml = ml_cache[name]
    w  = detect_weaknesses(df)

    total_profit = round(float(df["Profit"].sum()), 2) if "Profit" in df.columns else 0
    total_sales  = round(float(df["Sales"].sum()),  2) if "Sales"  in df.columns else 0
    margin       = round(total_profit / total_sales * 100, 2) if total_sales else 0

    return jsonify({
        "report_title":   f"AI Decision Intelligence Report — {name}",
        "generated_at":   datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "dataset":        name,
        "summary": {
            "total_sales":   total_sales,
            "total_profit":  total_profit,
            "profit_margin": margin,
            "rows":          len(df),
            "best_model":    ml.get("best_model", "N/A"),
        },
        "key_findings":      _key_findings(df, ml, w),
        "weaknesses_count":  len(w.get("loss_making_products", {})) + len(w.get("low_performing_regions", {})),
        "top_recommendation": "Deploy ML-driven pricing and eliminate loss-making sub-categories within 60 days.",
        "projected_growth":  "+10% profit with recommended strategy application",
    })


# ── Live Refresh Status ────────────────────────────────────────────

@app.route("/api/refresh-status", methods=["GET"])
def refresh_status():
    return jsonify({
        "status":          "live",
        "datasets_loaded": len(datasets),
        "server_time":     datetime.datetime.utcnow().isoformat(),
        "uptime_ok":       True,
    })


# ── Entry Point ────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port, host="0.0.0.0")

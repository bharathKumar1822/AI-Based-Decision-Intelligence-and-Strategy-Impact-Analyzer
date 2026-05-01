# main.py  — AI Decision Intelligence & Strategy Impact Analyzer
# Run: python src/main.py  (from project root)
# ============================================================
# PHASE 2 → PHASE 11
# ============================================================

import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — saves plots instead of showing

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

sns.set_style("whitegrid")

# ── Resolve paths relative to THIS file ──────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ============================================================
# PHASE 6: STRATEGY DEFINITION
# ============================================================

def strategy_profit(df):
    return float(df["Profit"].sum()) if "Profit" in df.columns else 0.0

def strategy_sales(df):
    return float(df["Sales"].sum()) if "Sales" in df.columns else 0.0

def strategy_loss(df):
    if "Profit" in df.columns:
        return float(df[df["Profit"] < 0]["Profit"].sum())
    return 0.0

# ============================================================
# PHASE 2 & PHASE 5: LOAD DATA
# ============================================================

datasets = {}

def add_dataset(name, path):
    if not os.path.exists(path):
        print(f"⚠️  File not found: {path}  — skipping {name}")
        return
    df = pd.read_csv(path, encoding="latin1")
    datasets[name] = df
    print(f"✅ {name} loaded — {len(df)} rows, {len(df.columns)} columns")

add_dataset("dataset_A", os.path.join(DATA_DIR, "superstore_dataset1.csv"))
add_dataset("dataset_B", os.path.join(DATA_DIR, "superstore_dataset2.csv"))

if not datasets:
    sys.exit("❌ No datasets found. Place CSV files in the data/ folder and re-run.")

# ============================================================
# PHASE 3: DATA CLEANING
# ============================================================

for name, df in datasets.items():
    print(f"\n🧹 Cleaning dataset: {name}")
    df.drop_duplicates(inplace=True)
    for col in ("Order Date", "Ship Date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)
    cat_cols = df.select_dtypes(exclude=["number"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    print(f"   Shape after cleaning: {df.shape}")

# ============================================================
# PHASE 4: EDA + GRAPHS
# ============================================================

for name, df in datasets.items():
    print(f"\n📊 EDA for {name}")
    print(df.describe())

    if "Category" in df.columns and "Profit" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        df.groupby("Category")["Profit"].sum().plot(kind="bar", ax=ax, color=["#6C63FF","#FF6584","#43CBFF"])
        ax.set_title(f"{name} — Profit by Category")
        ax.set_ylabel("Profit ($)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, f"{name}_profit_by_category.png"), dpi=100)
        plt.close()

    if "Sales" in df.columns and "Profit" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df.sample(min(2000, len(df)), random_state=1),
                        x="Sales", y="Profit", alpha=0.4, color="#6C63FF", ax=ax)
        ax.set_title(f"{name} — Sales vs Profit")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, f"{name}_sales_vs_profit.png"), dpi=100)
        plt.close()

    if "Customer Name" in df.columns and "Profit" in df.columns:
        top5 = df.groupby("Customer Name")["Profit"].sum().sort_values(ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(8, 5))
        top5.plot(kind="bar", ax=ax, color="#43CBFF")
        ax.set_title(f"{name} — Top 5 Customers by Profit")
        ax.set_ylabel("Profit ($)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, f"{name}_top_customers.png"), dpi=100)
        plt.close()
        print("\n  Top 5 Customers by Profit:")
        print(top5)

print(f"\n✅ Charts saved to: {OUTPUTS_DIR}")

# ============================================================
# PHASE 5: WEAKNESS DETECTION
# ============================================================

weaknesses = {}
for name, df in datasets.items():
    weaknesses[name] = {}
    if "Sub-Category" in df.columns and "Profit" in df.columns:
        lp = df.groupby("Sub-Category")["Profit"].sum()
        weaknesses[name]["loss_making_products"] = lp[lp < 0].to_dict()
    if "Region" in df.columns and "Profit" in df.columns:
        avg = df["Profit"].mean()
        rp  = df.groupby("Region")["Profit"].sum()
        weaknesses[name]["low_performing_regions"] = rp[rp < avg].to_dict()
    if "Category" in df.columns and "Profit" in df.columns and "Sales" in df.columns:
        df2 = df.copy()
        df2["margin"] = df2["Profit"] / df2["Sales"].replace(0, float("nan"))
        pm = df2.groupby("Category")["margin"].mean()
        weaknesses[name]["poor_profit_margins"] = pm[pm < 0.1].to_dict()

print("\n\n🔍 Weakness Detection")
print("-" * 40)
for name, weak in weaknesses.items():
    print(f"\n  {name}:")
    for key, val in weak.items():
        print(f"    {key}: {val}")

# ============================================================
# PHASE 6: STRATEGY EXECUTION
# ============================================================

results = {
    name: {
        "profit": strategy_profit(df),
        "sales":  strategy_sales(df),
        "loss":   strategy_loss(df),
    }
    for name, df in datasets.items()
}

# ============================================================
# PHASE 7: MACHINE LEARNING PREDICTIONS
# ============================================================

predictions = {}
for name, df in datasets.items():
    predictions[name] = {}
    if {"Sales", "Quantity", "Profit"}.issubset(df.columns):
        X = df[["Sales", "Quantity"]].values
        y = df["Profit"].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

        lr = LinearRegression().fit(Xtr, ytr)
        dt = DecisionTreeRegressor(random_state=42).fit(Xtr, ytr)
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr)

        for model, key in [(lr, "linear_regression"), (dt, "decision_tree"), (rf, "random_forest")]:
            pred = model.predict(Xte)
            predictions[name][f"{key}_mse"]              = float(mean_squared_error(yte, pred))
            predictions[name][f"predicted_profit_{key}"] = float(pred.mean())

print("\n\n🔮 ML Predictions")
print("-" * 40)
for name, pred in predictions.items():
    print(f"\n  {name}:")
    for key, val in pred.items():
        print(f"    {key}: {val:.4f}")

# ============================================================
# PHASE 8: BEST MODEL SELECTION
# ============================================================

best_strategies = {}
for name in datasets:
    pred = predictions[name]
    mses = {
        "Linear Regression": pred.get("linear_regression_mse", float("inf")),
        "Decision Tree":     pred.get("decision_tree_mse",     float("inf")),
        "Random Forest":     pred.get("random_forest_mse",     float("inf")),
    }
    best_model = min(mses, key=mses.get)
    key_map    = {"Linear Regression": "linear_regression",
                  "Decision Tree":     "decision_tree",
                  "Random Forest":     "random_forest"}
    best_strategies[name] = {
        "best_model":       best_model,
        "predicted_profit": pred.get(f"predicted_profit_{key_map[best_model]}", 0),
    }

print("\n\n⚙️  Best Strategy Selection")
print("-" * 40)
for name, strat in best_strategies.items():
    print(f"  {name} → Best Model: {strat['best_model']}  |  Predicted Profit: {strat['predicted_profit']:.2f}")

# ============================================================
# PHASE 9: STRATEGY SIMULATION
# ============================================================

simulations = {
    name: {
        "simulated_profit": best_strategies[name]["predicted_profit"] * 1.1,
        "simulated_growth": 10,
    }
    for name in datasets
}

print("\n\n🔥 Strategy Simulation")
print("-" * 40)
for name, sim in simulations.items():
    print(f"  {name} → Simulated Future Profit: {sim['simulated_profit']:.2f}  (+{sim['simulated_growth']}%)")

# ============================================================
# PHASE 10: RECOMMENDATIONS
# ============================================================

recommendations = {}
for name in datasets:
    weak = weaknesses[name]
    rec  = {"DOs": [], "DONTs": []}
    if weak.get("loss_making_products"):
        prods = ", ".join(list(weak["loss_making_products"].keys())[:5])
        rec["DONTs"].append(f"Avoid loss-making sub-categories: {prods}")
    if weak.get("low_performing_regions"):
        rec["DONTs"].append("Reduce focus on: " + ", ".join(weak["low_performing_regions"].keys()))
    if weak.get("poor_profit_margins"):
        rec["DONTs"].append("Fix pricing for: " + ", ".join(weak["poor_profit_margins"].keys()))
    rec["DOs"].append(f"Use {best_strategies[name]['best_model']} for future predictions")
    rec["DOs"].append("Expand in top-performing regions and categories")
    recommendations[name] = rec

print("\n\n💡 Recommendations")
print("-" * 40)
for name, rec in recommendations.items():
    print(f"\n  {name}:")
    for d in rec["DOs"]:   print(f"    ✅ DO:   {d}")
    for d in rec["DONTs"]: print(f"    ❌ DON'T: {d}")

# ============================================================
# PHASE 11: MULTI-DATASET COMPARISON
# ============================================================

profits = {name: results[name]["profit"] for name in results}
best_company = max(profits, key=profits.get)

print("\n\n⚖️  Multi-Dataset Comparison")
print("-" * 40)
for name, p in profits.items():
    marker = "🏆 BEST" if name == best_company else ""
    print(f"  {name}: Total Profit = ${p:,.2f}  {marker}")
print(f"\n  → Best Performer: {best_company}")
print("\n✅ Analysis complete.\n")
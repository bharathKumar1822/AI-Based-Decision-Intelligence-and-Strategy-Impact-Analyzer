import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

# Load data
datasets = {}
datasets["dataset_A"] = pd.read_csv("./data/superstore_dataset1.csv", encoding='latin1')
datasets["dataset_B"] = pd.read_csv("./data/superstore_dataset2.csv", encoding='latin1')

# Clean data (similar to main.py)
for name, df in datasets.items():
    df.drop_duplicates(inplace=True)
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%Y-%m-%d', errors='coerce')
    if 'Ship Date' in df.columns:
        df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%Y-%m-%d', errors='coerce')
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    cat_cols = df.select_dtypes(exclude=['number']).columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')

st.title("Decision Intelligence Dashboard")

dataset_choice = st.selectbox("Select Dataset", list(datasets.keys()))
df = datasets[dataset_choice]

st.header(f"Dataset: {dataset_choice}")
st.dataframe(df.head())

# EDA
st.header("Exploratory Data Analysis")
if 'Category' in df.columns and 'Profit' in df.columns:
    profit_by_category = df.groupby("Category")["Profit"].sum()
    st.bar_chart(profit_by_category)

if 'Sales' in df.columns and 'Profit' in df.columns:
    st.scatter_chart(df[['Sales', 'Profit']])

if 'Customer Name' in df.columns and 'Profit' in df.columns:
    top_customers = df.groupby("Customer Name")["Profit"].sum().sort_values(ascending=False).head(5)
    st.write("Top 5 Customers by Profit:")
    st.dataframe(top_customers)

# Weaknesses
st.header("Weaknesses")
weaknesses = {}
if 'Sub-Category' in df.columns and 'Profit' in df.columns:
    loss_products = df.groupby("Sub-Category")["Profit"].sum()
    loss_products = loss_products[loss_products < 0]
    weaknesses["loss_making_products"] = loss_products.to_dict()

if 'Region' in df.columns and 'Profit' in df.columns:
    avg_profit = df['Profit'].mean()
    region_profit = df.groupby("Region")["Profit"].sum()
    low_regions = region_profit[region_profit < avg_profit]
    weaknesses["low_performing_regions"] = low_regions.to_dict()

if 'Category' in df.columns and 'Profit' in df.columns and 'Sales' in df.columns:
    df['margin'] = df['Profit'] / df['Sales']
    poor_margins = df.groupby("Category")["margin"].mean()
    poor_margins = poor_margins[poor_margins < 0.1]
    weaknesses["poor_profit_margins"] = poor_margins.to_dict()

for key, val in weaknesses.items():
    st.write(f"{key}: {val}")

# ML Predictions
st.header("Machine Learning Predictions")
if 'Sales' in df.columns and 'Quantity' in df.columns and 'Profit' in df.columns:
    features = df[['Sales', 'Quantity']]
    target = df['Profit']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    lr = LinearRegression().fit(X_train, y_train)
    dt = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
    rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    dt_pred = dt.predict(X_test)
    rf_pred = rf.predict(X_test)
    st.write(f"Linear Regression MSE: {mean_squared_error(y_test, lr_pred)}")
    st.write(f"Decision Tree MSE: {mean_squared_error(y_test, dt_pred)}")
    st.write(f"Random Forest MSE: {mean_squared_error(y_test, rf_pred)}")
    mses = {
        "Linear Regression": mean_squared_error(y_test, lr_pred),
        "Decision Tree": mean_squared_error(y_test, dt_pred),
        "Random Forest": mean_squared_error(y_test, rf_pred)
    }
    best_model = min(mses, key=mses.get)
    st.write(f"Best Model: {best_model}")

# Recommendations
st.header("Recommendations")
rec = {"DOs": [], "DONTs": []}
if weaknesses.get("loss_making_products"):
    rec["DONTs"].append("Loss products: " + ", ".join(weaknesses["loss_making_products"].keys()))
if weaknesses.get("low_performing_regions"):
    rec["DONTs"].append("Low regions: " + ", ".join(weaknesses["low_performing_regions"].keys()))
if weaknesses.get("poor_profit_margins"):
    rec["DONTs"].append("Low margin: " + ", ".join(weaknesses["poor_profit_margins"].keys()))
rec["DOs"].append(f"Use {best_model}")
st.write(rec)
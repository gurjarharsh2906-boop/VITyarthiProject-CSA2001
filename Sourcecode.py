#SECTION 1: Imports and Configuration
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

#Configuration 
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "Retail_Sales_Data_Unlox (1).csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#SECTION 2: Load and Inspect Dataset
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Dataset not found: " + path + "\n"
            "Make sure Retail_Sales_Data.csv is in the same folder as this script."
        )
    df = pd.read_csv(path)
    print("=" * 52)
    print("  Dataset loaded successfully!")
    print("  Rows    : " + str(df.shape[0]))
    print("  Columns : " + str(df.shape[1]))
    print("  Fields  : " + ", ".join(df.columns.tolist()))
    print("=" * 52)
    return df

def inspect_data(df):
    print("\n-- Data Types --")
    print(df.dtypes)
    print("\n-- Null Counts --")
    print(df.isnull().sum())
    print("\n-- Summary Statistics --")
    print(df.describe())

#SECTION 3: Data Preprocessing
def preprocess(df):
    df = df.copy()
    df["Date"]       = pd.to_datetime(df["Date"])
    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.month_name()
    print("\nDate column parsed. Sample:")
    print(df[["Date", "Year", "Month", "Month_Name"]].head())
    return df

#SECTION 4: Exploratory Data Analysis
def plot_sales_over_time(df):
    data = df.groupby("Date")["Total_Sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data["Date"], data["Total_Sales"], color="steelblue", linewidth=1.5)
    ax.set_title("Total Sales Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Sales")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "sales_over_time.png")
    fig.savefig(out, dpi=150)
    plt.show()
    print("  Saved: " + out)

def plot_monthly_trend(df):
    data = df.groupby(["Year", "Month"])["Total_Sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=data, x="Month", y="Total_Sales", hue="Year", marker="o", ax=ax)
    ax.set_title("Monthly Sales Trend by Year", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    ax.set_xticks(range(1, 13))
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "monthly_sales_trend.png")
    fig.savefig(out, dpi=150)
    plt.show()
    print("  Saved: " + out)

def plot_category_sales(df):
    data = df.groupby("Product_Category")["Total_Sales"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    data.plot(kind="bar", ax=ax, color="coral", edgecolor="white")
    ax.set_title("Total Sales by Product Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Total Sales")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "category_sales.png")
    fig.savefig(out, dpi=150)
    plt.show()
    print("  Saved: " + out)

def plot_top_stores(df, top_n=10):
    data = df.groupby("Store_ID")["Total_Sales"].sum().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(9, 5))
    data.plot(kind="bar", ax=ax, color="mediumseagreen", edgecolor="white")
    ax.set_title("Top " + str(top_n) + " Stores by Total Sales", fontsize=14, fontweight="bold")
    ax.set_xlabel("Store ID")
    ax.set_ylabel("Total Sales")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "top_stores.png")
    fig.savefig(out, dpi=150)
    plt.show()
    print("  Saved: " + out)

#SECTION 5: Store Segmentation using KMeans
def segment_stores(df, n_clusters=3):
    store_perf = (
        df.groupby("Store_ID")
        .agg(Total_Sales=("Total_Sales", "sum"), Transactions=("Date", "count"))
        .reset_index()
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(store_perf[["Total_Sales", "Transactions"]])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    store_perf["Cluster"] = kmeans.fit_predict(scaled)

    print("\nStore Segmentation (" + str(n_clusters) + " clusters):")
    print(store_perf.groupby("Cluster")[["Total_Sales", "Transactions"]].mean().round(2))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=store_perf, x="Total_Sales", y="Transactions",
                    hue="Cluster", palette="Set2", s=80, ax=ax)
    ax.set_title("Store Performance Segmentation", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Sales")
    ax.set_ylabel("Transactions")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "store_segmentation.png")
    fig.savefig(out, dpi=150)
    plt.show()
    print("  Saved: " + out)
    return store_perf

#SECTION6: SARIMA Sales Forecasting
def forecast_sales(df, forecast_steps=6):
    monthly_ts = df.set_index("Date").resample("ME")["Total_Sales"].sum()

    if len(monthly_ts) < 24:
        print("Warning: Less than 24 months of data. SARIMA results may vary.")

    train = monthly_ts.iloc[:-forecast_steps]
    test  = monthly_ts.iloc[-forecast_steps:]

    print("\nFitting SARIMA on " + str(len(train)) + " months of data ...")
    model   = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)
    print(results.summary())

    forecast = results.forecast(steps=forecast_steps)

    mae  = np.mean(np.abs(test.values - forecast.values))
    mape = np.mean(np.abs((test.values - forecast.values) / test.values)) * 100
    print("\nForecast Evaluation:")
    print("  MAE  : " + str(round(mae, 2)))
    print("  MAPE : " + str(round(mape, 2)) + "%")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train.index,    train,    label="Train",    color="steelblue")
    ax.plot(test.index,     test,     label="Actual",   color="darkorange")
    ax.plot(forecast.index, forecast, label="Forecast", color="green",
            linestyle="--", linewidth=2)
    ax.set_title("Sales Forecast - Next " + str(forecast_steps) + " Months",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Sales")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "sales_forecast.png")
    fig.savefig(out, dpi=150)
    plt.show()
    print("  Saved: " + out)

#SECTION  7: Business Insights
def print_insights():
    print("")
    print("+----------------------------------------------------------+")
    print("|       Business Insights and Recommendations              |")
    print("+----------------------------------------------------------+")
    print("| * Sales show clear seasonal trends.                      |")
    print("|   Plan promotions around peak months.                    |")
    print("|                                                          |")
    print("| * Certain categories drive most of the revenue.          |")
    print("|   Prioritize their stock levels.                         |")
    print("|                                                          |")
    print("| * High-performing stores are benchmarks for best         |")
    print("|   practices across the retail network.                   |")
    print("|                                                          |")
    print("| * Low-performing clusters may need pricing reviews       |")
    print("|   or better inventory and marketing support.             |")
    print("|                                                          |")
    print("| * SARIMA forecasts support proactive demand planning     |")
    print("|   and smarter inventory control.                         |")
    print("+----------------------------------------------------------+")
    print("")

# MAIN
def main():
    df = load_data(DATA_PATH)
    inspect_data(df)
    df = preprocess(df)

    print("\n-- EDA Charts --")
    plot_sales_over_time(df)
    plot_monthly_trend(df)
    plot_category_sales(df)
    plot_top_stores(df, top_n=10)

    print("\n-- Store Segmentation --")
    segment_stores(df, n_clusters=3)

    print("\n-- SARIMA Forecasting --")
    forecast_sales(df, forecast_steps=6)

    print_insights()
    print("All outputs saved to: " + OUTPUT_DIR)

if __name__ == "__main__":
    main()

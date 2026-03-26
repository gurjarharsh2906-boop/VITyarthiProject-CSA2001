# VITyarthiProject-CSA2001
# Retail Analytics & AI-Powered Sales Forecasting
A Python-based Command Line Utility for Analyzing Retail Sales Trends, Segmenting Store Performance via K-Means Clustering, and Generating AI-Powered Revenue Forecasts.
##  Project Overview
An end-to-end data science pipeline that transforms raw retail transactions into actionable insights. It combines automated EDA, Machine Learning for store clustering, and SARIMA modeling for revenue forecasting.

##  Key Features
* **Automated Pipeline:** Cleans and aggregates CSV data into time-series and categorical metrics.
* **Visual Analytics:** Auto-generates trend reports, seasonality charts, and performance rankings.
* **Store Segmentation:** Uses **K-Means Clustering** to group stores into performance tiers.
* **Predictive Modeling:** Implements **SARIMA** to project sales while accounting for seasonal cycles.

##  Business Insights
* **Inventory:** 6-month forecasts enable proactive stock adjustments before peak seasons.
* **Benchmarking:** Top-tier clusters serve as operational models for underperforming stores.
* **Targeting:** Identifies high-revenue regions and categories to optimize marketing spend.
* **Seasonality:** Pinpoints specific months for high-impact holiday and promotion planning.

##  Setup & Execution
1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
    ```
2.  **Configuration:** Place `Retail_Sales_Data.csv` in the script's folder.
3.  **Run Project:**
    ```bash
    python "Sourcecode.py"
    ```

##  Verify Outputs
After execution, check the `/outputs` directory for:
* `sales_over_time.png` & `monthly_sales_trend.png`
* `category_sales.png` & `top_10_stores.png`
* `store_segments.png` & `sales_forecast.png`

---

##  Project Structure
* `Sourcecode.py` —> Analysis Engine
* `Retail_Sales_Data.csv` —> Raw Dataset
* `/outputs/` —> Visual reports

##  Evaluation Compliance
* **CLI-Only:** 100% terminal-executable; no GUI required.
* **Headless:** Runs on servers without displays via `Agg` backend.
* **Portable:** Uses relative paths for automatic dataset detection.

##  Future Enhancements
* **Auto-ML:** Implement Auto-ARIMA for dynamic parameter selection.
* **Real-time BI:** Add a Streamlit dashboard for interactive data filtering.
* **Deep Learning:** Integrate LSTM networks for improved non-linear forecasting.
## Technical Note
* **Data Source:** Please note that the dataset used in this project (Retail_Sales_Data.csv) consists of synthetic data generated entirely with the assistance of AI to simulate real-world retail environments for analytical testing.
##  Author
**HARSH GURJAR**\
RegNo.: 25BCE11195\
Date: 26th March 2026

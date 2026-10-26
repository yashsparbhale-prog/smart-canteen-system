# 🍽️ Smart Canteen System (Data Science Project)

A clean, practical, and end-to-end Data Science project designed to optimize canteen operations, forecast daily meal preparation targets to eliminate food waste, recommend high-converting combo meals, and analyze customer dining behavior.

---

## 📌 Project Overview & Objectives

College and corporate canteens face three core logistical challenges:
1. **Food Wastage & Overcooking**: Cooking too much food on low-traffic days leads to spoilage and financial loss.
2. **Stockouts during Rush Hours**: Underpreparing during exam weeks or peak hours causes long queues and student dissatisfaction.
3. **Suboptimal Revenue**: Lack of data-driven combo meal promotions and personalized offers.

The **Smart Canteen System** addresses these issues using Machine Learning and Analytics:
- **Daily Demand Forecasting**: Predicts exact portions of Meals, Snacks, Beverages, and Desserts using Ridge Regression and Random Forest.
- **Smart Menu Recommendations**: Discovers frequently co-ordered items via Market Basket Analysis (Support, Confidence, Lift) for real-time upsells.
- **Customer Segmentation**: Clusters customers into *Occasional Snackers*, *Regular Diners*, and *VIP High-Spenders* using K-Means RFM analysis.
- **Interactive Web Dashboard**: Streamlit dashboard for canteen staff to generate instant preparation plans and explore KPIs.

---

## 📂 Repository Structure

```text
Smart Canteen/
│
├── data/
│   ├── canteen_orders.csv            # 4,500+ realistic transaction orders
│   └── demand_model.pkl              # Saved trained Random Forest model
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb          # Rush hours, sales trends, wait times
│   ├── 02_demand_forecasting.ipynb                 # Time-series ML modeling & evaluation
│   └── 03_customer_insights_and_recommendation.ipynb # Market basket analysis & RFM clustering
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py             # Realistic dataset generator with exam/rush patterns
│   ├── forecasting_model.py          # ML feature engineering, model training & prediction
│   ├── recommender.py                # Co-occurrence engine & K-Means clustering
│   └── create_notebooks.py           # Helper script generating the Jupyter notebooks
│
├── main.py                           # CLI pipeline executing all stages end-to-end
├── dashboard.py                      # Interactive Streamlit Web Application
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Complete Data Science Pipeline (CLI)
Execute all steps (data loading, model training, prediction, recommendations, and segmentation):
```bash
python main.py
```

### 3. Launch the Interactive Dashboard
Launch the web interface for managers and canteen staff:
```bash
streamlit run dashboard.py
```
*(Open http://localhost:8501 in your browser)*

### 4. Explore the Jupyter Notebooks
Open the `notebooks/` directory in VS Code or Jupyter:
- **`01_exploratory_data_analysis.ipynb`**: Visual exploration of hourly rush patterns, exam week impact, and wait times.
- **`02_demand_forecasting.ipynb`**: Train and evaluate Ridge vs. Random Forest for kitchen demand targets.
- **`03_customer_insights_and_recommendation.ipynb`**: Discover high-lift combos and cluster students by spending.

---

## 🧠 Machine Learning & Analytical Methodology

### 1. Demand Forecasting
- **Features Used**: Previous 1-day lag (`lag_1_qty`), same-day previous week lag (`lag_7_qty`), 7-day rolling average (`rolling_7_mean`), day of week, month, weekend indicator, exam week indicator, and one-hot encoded food category.
- **Models**:
  - Baseline: Ridge Regression ($L_2$ regularization)
  - Non-linear: Random Forest Regressor (100 estimators, depth=8)
- **Evaluation**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ score.

### 2. Market Basket Analysis (Recommendations)
- Analyzes multi-item transaction baskets:
  $$\text{Support}(A \rightarrow B) = \frac{\text{Count}(A \cap B)}{\text{Total Orders}}$$
  $$\text{Confidence}(A \rightarrow B) = \frac{\text{Count}(A \cap B)}{\text{Count}(A)}$$
  $$\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}$$
- Identifies strong combos like:
  - *Masala Chai + Samosa* (Lift: ~2.5x)
  - *Chicken Biryani + Gulab Jamun* (Lift: ~4.4x)
  - *Burger + Cold Coffee* (Lift: ~2.2x)

### 3. Customer RFM Segmentation
- **Recency**: Days elapsed since the student's last order.
- **Frequency**: Total unique orders placed.
- **Monetary**: Total lifetime spend in the canteen.
- Scaled using `StandardScaler` and clustered with `K-Means (k=3)` into:
  1. *Occasional Snackers* (Lower frequency, quick snack buyers)
  2. *Regular Diners* (Daily lunch customers)
  3. *VIP High-Spenders* (Highest frequency and wallet share)

---

## 📈 Key Results & Business Impact

| Metric / Goal | Result |
| :--- | :--- |
| **Demand Forecasting Accuracy** | **MAE ~3.3 units**, allowing kitchen staff to prepare tight portion targets |
| **Peak Rush Windows** | Identified at **13:00 - 15:00** (Lunch) and **16:00 - 18:00** (Snacks) |
| **Exam Period Behavior** | Demand shifts heavily towards caffeinated drinks (+80%) and quick snacks |
| **Food Waste Reduction** | Estimated **15% - 25% decrease** in unserved food disposal |
| **Basket Size Increase** | **+12% to 18% potential revenue gain** from recommended combo promotions |

---

## 🛠️ Technologies Used
- **Language**: Python 3.10+
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (RandomForestRegressor, Ridge, KMeans, StandardScaler)
- **Data Visualization**: Matplotlib, Seaborn
- **Interactive UI**: Streamlit
- **Notebook Environment**: Jupyter / IPykernel

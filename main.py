"""
Smart Canteen System - Main Demonstration Pipeline
Runs the complete Data Science pipeline:
1. Dataset Verification & Summary Analytics
2. Demand Forecasting Model Training & Next-Day Prediction
3. Smart Menu Recommendation & Combo Optimization
4. Customer RFM Segmentation & Behavior Profiling
"""

import os
import pandas as pd
from datetime import datetime, timedelta

from src.data_generator import generate_canteen_dataset
from src.forecasting_model import train_demand_model, predict_next_day
from src.recommender import SmartCanteenRecommender, perform_customer_segmentation


def print_header(title):
    print("\n" + "=" * 65)
    print(f"  {title.upper()}")
    print("=" * 65)


def run_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "canteen_orders.csv")
    model_path = os.path.join(base_dir, "data", "demand_model.pkl")

    print_header("Smart Canteen System - Data Science Project")

    # Step 1: Dataset check
    if not os.path.exists(data_path):
        print("[1/5] Generating synthetic dataset...")
        df = generate_canteen_dataset(num_orders=5000, output_path=data_path)
    else:
        print("[1/5] Loading canteen dataset...")
        df = pd.read_csv(data_path)

    total_orders = df["order_id"].nunique()
    total_revenue = df["total_price"].sum()
    busiest_hour = df["hour"].value_counts().idxmax()
    avg_wait = df["wait_time_min"].mean()
    top_item = df["item_name"].value_counts().idxmax()

    print(f" -> Total Orders Processed : {total_orders:,}")
    print(f" -> Total Revenue Generated: Rs. {total_revenue:,.2f}")
    print(f" -> Most Popular Item      : {top_item}")
    print(f" -> Peak Rush Hour         : {busiest_hour}:00 - {busiest_hour + 1}:00")
    print(f" -> Average Customer Wait  : {avg_wait:.1f} minutes")

    # Step 2: Demand Forecasting Training
    print_header("Step 2: Training Demand Forecasting Models (ML)")
    print("Training Ridge Regression and Random Forest models on daily demand...")
    artifacts, metrics = train_demand_model(data_path, model_path)

    print("\nModel Evaluation (Test Set):")
    for model_name, score in metrics.items():
        print(f"  - {model_name:14s} | MAE: {score['MAE']:5.2f} items | RMSE: {score['RMSE']:5.2f} | R²: {score['R2']:.3f}")

    print("\nTop 3 Demand Predictors:")
    for feat, imp in artifacts["feature_importance"][:3]:
        print(f"  * {feat:20s}: {imp * 100:.1f}% importance")

    # Step 3: Kitchen Prep Forecast for Tomorrow
    print_header("Step 3: AI Kitchen Demand Forecast for Tomorrow")
    tomorrow = datetime.now() + timedelta(days=1)
    is_weekend = 1 if tomorrow.weekday() >= 5 else 0
    predictions = predict_next_day(
        artifacts,
        recent_history_df=df.tail(500),
        is_weekend=is_weekend,
        is_exam_week=0,
        day_num=tomorrow.weekday(),
        month=tomorrow.month,
        day_of_month=tomorrow.day
    )

    print(f"Forecast for: {tomorrow.strftime('%A, %d %B %Y')} (Weekend: {'Yes' if is_weekend else 'No'})")
    print("Recommended Portions to Prepare (Zero-Waste Target):")
    for cat, qty in predictions.items():
        print(f"  - {cat:12s} : {qty:3d} portions")

    # Step 4: Recommender & Combo Offers
    print_header("Step 4: Smart Menu Recommendations (Market Basket Analysis)")
    recommender = SmartCanteenRecommender(data_path)
    top_combos = recommender.get_top_combos(top_n=4)

    print("Top High-Converting Combo Deals for Canteen Display:")
    for idx, c in enumerate(top_combos, 1):
        print(f"  {idx}. {c['combo']} (Co-orders: {c['co_purchases']}, Lift: {c['lift']}x)")

    sample_item = "Cheese Burger"
    print(f"\nReal-Time Cross-Sell Recommendation for '{sample_item}':")
    recs = recommender.recommend_combos(sample_item, top_n=2)
    for r in recs:
        print(f"  -> Suggest: '{r['recommended_item']}' ({r['confidence_pct']}% of burger buyers order this)")

    # Step 5: Customer Segmentation
    print_header("Step 5: Customer Segmentation (K-Means Clustering)")
    _, summary = perform_customer_segmentation(df)
    print(summary.to_string(index=False))

    print_header("Pipeline Finished Successfully!")
    print("Next Steps:")
    print(" 1. Interactive Dashboard : Run 'streamlit run dashboard.py'")
    print(" 2. Detailed Data Science  : Open the notebooks in 'notebooks/'")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline()

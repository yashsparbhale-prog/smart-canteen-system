"""
Smart Canteen System - Interactive Streamlit Dashboard
A visual interface for canteen managers and data science demonstration.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.recommender import SmartCanteenRecommender, perform_customer_segmentation
from src.forecasting_model import predict_next_day

# Set page config
st.set_page_config(
    page_title="Smart Canteen Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
    }
    .highlight-text {
        color: #0284c7;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Data paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "canteen_orders.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "demand_model.pkl")


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        from src.data_generator import generate_canteen_dataset
        generate_canteen_dataset(output_path=DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    from src.forecasting_model import train_demand_model
    artifacts, _ = train_demand_model(DATA_PATH, MODEL_PATH)
    return artifacts


df = load_data()
model_artifacts = load_model()
recommender = SmartCanteenRecommender(DATA_PATH)

# Sidebar
st.sidebar.title("🍽️ Smart Canteen")
st.sidebar.markdown("**AI & Data Science Suite**")
st.sidebar.markdown("Optimizing kitchen preparation, reducing food waste, and recommending menu combos.")

st.sidebar.markdown("---")
view_mode = st.sidebar.radio(
    "Navigation",
    ["📊 Operations Overview", "🔮 AI Demand Forecaster", "🍔 Menu Recommender & Combos", "👥 Customer Segmentation"]
)

# ----------------- TAB 1: OPERATIONS OVERVIEW -----------------
if view_mode == "📊 Operations Overview":
    st.title("📊 Smart Canteen Operations Overview")
    st.markdown("Real-time operational KPIs and sales distribution analytics.")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    total_orders = df["order_id"].nunique()
    total_revenue = df["total_price"].sum()
    busiest_hour = df["hour"].value_counts().idxmax()
    avg_rating = df["rating"].mean()

    col1.metric("Total Orders", f"{total_orders:,}")
    col2.metric("Total Revenue", f"₹ {total_revenue:,.0f}")
    col3.metric("Peak Rush Hour", f"{busiest_hour}:00 - {busiest_hour+1}:00")
    col4.metric("Avg Satisfaction", f"⭐ {avg_rating:.2f} / 5.0")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Hourly Footfall & Order Volume")
        hourly_counts = df.groupby("hour")["order_id"].nunique().reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=hourly_counts, x="hour", y="order_id", color="#3b82f6", ax=ax)
        ax.set_title("Orders by Hour of the Day", fontsize=12, fontweight="bold")
        ax.set_xlabel("Hour (24h)")
        ax.set_ylabel("Order Count")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        st.pyplot(fig)

    with c2:
        st.subheader("Sales by Food Category")
        cat_sales = df.groupby("category")["total_price"].sum().reset_index()
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        colors = ["#10b981", "#f59e0b", "#6366f1", "#ec4899"]
        ax2.pie(cat_sales["total_price"], labels=cat_sales["category"], autopct="%1.1f%%", colors=colors, startangle=140)
        ax2.set_title("Revenue Contribution by Category", fontsize=12, fontweight="bold")
        st.pyplot(fig2)

    st.subheader("Top 8 Best Selling Menu Items")
    top_items = df.groupby(["item_name", "category"]).agg(
        total_quantity=("quantity", "sum"),
        revenue=("total_price", "sum"),
        avg_wait_min=("wait_time_min", "mean")
    ).reset_index().sort_values(by="total_quantity", reverse=True).head(8)

    st.dataframe(
        top_items.rename(columns={
            "item_name": "Menu Item",
            "category": "Category",
            "total_quantity": "Units Sold",
            "revenue": "Revenue (₹)",
            "avg_wait_min": "Avg Wait (mins)"
        }).style.format({"Revenue (₹)": "₹{:,.0f}", "Avg Wait (mins)": "{:.1f}"}),
        use_container_width=True
    )

# ----------------- TAB 2: AI DEMAND FORECASTER -----------------
elif view_mode == "🔮 AI Demand Forecaster":
    st.title("🔮 AI Kitchen Demand Forecaster")
    st.markdown("Predict tomorrow's food portion requirements to prevent overcooking and stockouts.")

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        sel_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=1)
    with col_in2:
        is_exam = st.checkbox("Exam / Test Week (High stress & quick snacks)", value=False)
    with col_in3:
        sel_month = st.slider("Month", 1, 12, 10)

    day_num_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    is_weekend = 1 if sel_day in ["Saturday", "Sunday"] else 0
    day_num = day_num_map[sel_day]

    if st.button("Generate Kitchen Preparation Plan", type="primary"):
        preds = predict_next_day(
            model_artifacts,
            recent_history_df=df.tail(600),
            is_weekend=is_weekend,
            is_exam_week=1 if is_exam else 0,
            day_num=day_num,
            month=sel_month,
            day_of_month=15
        )

        st.success(f"Portion Target Generated for **{sel_day}** ({'Weekend' if is_weekend else 'Weekday'})")

        q_cols = st.columns(4)
        for idx, (cat, qty) in enumerate(preds.items()):
            q_cols[idx].metric(f"{cat} Target", f"{qty} portions", f"Zero-waste guide")

        st.markdown("---")
        st.subheader("Machine Learning Feature Drivers")
        feats = model_artifacts.get("feature_importance", [])[:5]
        if feats:
            feat_df = pd.DataFrame(feats, columns=["Feature", "Relative Importance"])
            fig_f, ax_f = plt.subplots(figsize=(8, 3.5))
            sns.barplot(data=feat_df, x="Relative Importance", y="Feature", palette="Blues_r", ax=ax_f)
            ax_f.set_title("Top 5 Factors Dictating Canteen Demand", fontsize=11, fontweight="bold")
            st.pyplot(fig_f)

# ----------------- TAB 3: MENU RECOMMENDER & COMBOS -----------------
elif view_mode == "🍔 Menu Recommender & Combos":
    st.title("🍔 Smart Menu Recommender & Combos")
    st.markdown("Association Rule & Market Basket Analysis for canteen billing screens.")

    st.subheader("1. Add-on Recommendation for Cashier / Kiosk")
    all_items = sorted(df["item_name"].unique())
    selected_item = st.selectbox("Customer just ordered:", all_items, index=all_items.index("Cheese Burger") if "Cheese Burger" in all_items else 0)

    recs = recommender.recommend_combos(selected_item, top_n=3)
    if recs:
        st.markdown(f"**Recommended Cross-sells for '{selected_item}':**")
        rec_cols = st.columns(len(recs))
        for i, r in enumerate(recs):
            with rec_cols[i]:
                st.info(f"✨ **{r['recommended_item']}**\n\n"
                        f"• Frequency: {r['co_occurrence_count']} times\n\n"
                        f"• Match Confidence: **{r['confidence_pct']}%**\n\n"
                        f"• Lift: **{r['lift']}x** above baseline")
    else:
        st.write("No strong pairings found for this item.")

    st.markdown("---")
    st.subheader("2. Top Promotional Combo Packs for Digital Menu Board")
    top_c = recommender.get_top_combos(5)
    combo_df = pd.DataFrame(top_c)
    st.dataframe(
        combo_df.rename(columns={
            "combo": "Promotional Combo",
            "co_purchases": "Historical Co-Orders",
            "support_pct": "Basket Share (%)",
            "lift": "Affinity Lift"
        }),
        use_container_width=True
    )

# ----------------- TAB 4: CUSTOMER SEGMENTATION -----------------
elif view_mode == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation (K-Means RFM)")
    st.markdown("Cluster customers by **Recency**, **Frequency**, and **Monetary** spending.")

    rfm_df, summary_df = perform_customer_segmentation(df)

    st.subheader("Customer Cluster Profiles")
    st.dataframe(
        summary_df.rename(columns={
            "segment_name": "Segment",
            "customer_count": "Customers",
            "avg_recency_days": "Avg Inactivity (Days)",
            "avg_orders": "Avg Orders Placed",
            "avg_spend": "Avg Total Spend (₹)"
        }).style.format({"Avg Total Spend (₹)": "₹{:,.1f}"}),
        use_container_width=True
    )

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        fig_rfm, ax_rfm = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=rfm_df,
            x="frequency",
            y="monetary",
            hue="segment_name",
            palette="Set2",
            alpha=0.8,
            ax=ax_rfm
        )
        ax_rfm.set_title("Customer Segments: Frequency vs Monetary", fontsize=11, fontweight="bold")
        ax_rfm.set_xlabel("Order Frequency (Count)")
        ax_rfm.set_ylabel("Monetary Spend (₹)")
        st.pyplot(fig_rfm)

    with col_s2:
        st.subheader("Targeted Marketing Strategy")
        st.markdown("""
        - 🌟 **VIP High-Spenders**: Offer VIP priority pickup lines and monthly subscription passes.
        - 🍽️ **Regular Diners**: Offer bundle coupons (e.g. 10% off on 5th lunch order).
        - ☕ **Occasional Snackers**: Send notification prompts during afternoon tea hours (4:00 PM) with snack discounts.
        """)

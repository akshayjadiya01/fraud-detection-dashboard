import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# ---------- Setup ----------

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Database connection
conn = sqlite3.connect("fraud_paysim_sample.db")
cursor = conn.cursor()

# Create necessary views if they don't exist
cursor.execute("""
CREATE VIEW IF NOT EXISTS v_fraud_by_type AS
SELECT type, COUNT(*) AS count, SUM(amount) AS total_loss
FROM transactions
WHERE isFraud = 1
GROUP BY type;
""")

cursor.execute("""
CREATE VIEW IF NOT EXISTS v_fraud_by_date AS
SELECT step, COUNT(*) AS count, SUM(amount) AS total_loss
FROM transactions
WHERE isFraud = 1
GROUP BY step;
""")

cursor.execute("""
CREATE VIEW IF NOT EXISTS v_top_senders AS
SELECT nameOrig AS sender, COUNT(*) AS count, SUM(amount) AS total_loss
FROM transactions
WHERE isFraud = 1
GROUP BY nameOrig
ORDER BY total_loss DESC
LIMIT 10;
""")

conn.commit()

# ---------- Functions ----------

@st.cache_data
def load_view(query):
    return pd.read_sql(query, conn)

@st.cache_data
def get_data():
    df_type = load_view("SELECT * FROM v_fraud_by_type;")
    df_daily = load_view("SELECT * FROM v_fraud_by_date;")
    df_top = load_view("SELECT * FROM v_top_senders;")
    df_loss = load_view("SELECT type, SUM(amount) AS total_loss FROM transactions WHERE isFraud = 1 GROUP BY type;")
    df_full = pd.read_sql("SELECT * FROM transactions;", conn)
    return df_type, df_daily, df_top, df_loss, df_full

def plot_fraud_by_type(df_type):
    fig = px.bar(df_type, x="type", y="count", title="Fraud Count by Transaction Type")
    st.plotly_chart(fig, use_container_width=True)

def plot_daily_trends(df_daily):
    fig = px.line(df_daily, x="step", y="count", title="Daily Fraud Counts")
    st.plotly_chart(fig, use_container_width=True)

def plot_loss_by_type(df_loss):
    fig = px.pie(df_loss, names="type", values="total_loss", title="Fraud Loss by Type")
    st.plotly_chart(fig, use_container_width=True)

def plot_top_senders(df_top):
    fig = px.bar(df_top, x="sender", y="total_loss", title="Top Fraudulent Senders")
    st.plotly_chart(fig, use_container_width=True)

def run_prediction(df_full):
    st.header("Fraud Prediction")

    # Prepare data
    X = df_full.drop(columns=["isFraud"])
    y = df_full["isFraud"]

    # Encoding categorical features if necessary
    X_encoded = pd.get_dummies(X)

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)

    st.write(f"ROC AUC Score: {auc:.4f}")

    # Feature importance
    importance = model.feature_importances_
    features = X_encoded.columns
    fi_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values(by="Importance", ascending=False).head(10)

    st.subheader("Top Features")
    st.dataframe(fi_df)

    fig, ax = plt.subplots()
    ax.barh(fi_df["Feature"], fi_df["Importance"])
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# ---------- Main App ----------

st.title("📊 Fraud Detection Dashboard")

# Navigation
page = st.sidebar.selectbox("Navigate", ["Dashboard", "Prediction", "Help"])

df_type, df_daily, df_top, df_loss, df_full = get_data()

if page == "Dashboard":
    st.header("Fraud Analytics Overview")
    plot_fraud_by_type(df_type)
    plot_daily_trends(df_daily)
    plot_loss_by_type(df_loss)
    plot_top_senders(df_top)

elif page == "Prediction":
    run_prediction(df_full)

elif page == "Help":
    st.header("Help Documentation")
    st.markdown("""
    ### How to Use the Dashboard
    - **Dashboard**: Explore fraud data through visualizations.
    - **Prediction**: Run a machine learning model to predict fraudulent transactions.
    - **Help**: Guidance on how to use the app.
    
    ### Notes
    - Data is from the PaySim dataset sample.
    - Model uses XGBoost for predictions.
    - Database views are created automatically on startup.
    """)

# ---------- Footer ----------

st.markdown("---")
st.write("Built with ❤️ using Streamlit, XGBoost, and SQLite.")

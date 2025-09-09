import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# ---------- Page Setup ----------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Database Setup ----------
conn = sqlite3.connect("fraud_paysim_sample.db")
cursor = conn.cursor()

# Create views if they don't exist
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
    st.subheader("📊 Fraud Distribution by Transaction Type")
    fig = px.bar(df_type, x="type", y="count", color="type", title="Fraud Count per Type",
                 labels={"count": "Number of Frauds", "type": "Transaction Type"})
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def plot_daily_trends(df_daily):
    st.subheader("📈 Daily Fraud Trends")
    fig = px.line(df_daily, x="step", y="count", title="Fraudulent Transactions Over Time",
                  labels={"step": "Time Step", "count": "Fraud Count"})
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def plot_loss_by_type(df_loss):
    st.subheader("💰 Fraud Loss by Type")
    fig = px.pie(df_loss, names="type", values="total_loss", title="Total Fraud Loss Distribution",
                 labels={"type": "Transaction Type", "total_loss": "Total Loss"})
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def plot_top_senders(df_top):
    st.subheader("🚨 Top Fraudulent Senders")
    fig = px.bar(df_top, x="sender", y="total_loss", color="total_loss",
                 title="Top 10 Fraudulent Senders by Total Loss", labels={"sender": "Sender", "total_loss": "Total Loss"})
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def run_prediction(df_full):
    st.header("🔮 Fraud Prediction")
    st.markdown("Train and evaluate a fraud detection model. Explore which features are most impactful.")

    X = df_full.drop(columns=["isFraud"])
    y = df_full["isFraud"]

    X_encoded = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)

    st.success(f"✅ Model Trained Successfully! ROC AUC Score: {auc:.4f}")

    importance = model.feature_importances_
    features = X_encoded.columns
    fi_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values(by="Importance", ascending=False).head(10)

    st.subheader("📌 Top Features Impacting Fraud")
    st.dataframe(fi_df)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=fi_df["Feature"],
        x=fi_df["Importance"],
        orientation='h',
        marker=dict(color='orange')
    ))
    fig.update_layout(title="Feature Importance", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Main App ----------

st.title("🚨 Fraud Detection Dashboard")
st.markdown("Analyze fraud patterns, visualize insights, and predict fraudulent activities with advanced tools.")

# Sidebar Navigation
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Go to", ["📊 Dashboard", "🔮 Prediction", "📖 Help"], index=0)

# Load Data
df_type, df_daily, df_top, df_loss, df_full = get_data()

# Pages
if page == "📊 Dashboard":
    st.header("Dashboard Overview")
    plot_fraud_by_type(df_type)
    plot_daily_trends(df_daily)
    plot_loss_by_type(df_loss)
    plot_top_senders(df_top)

elif page == "🔮 Prediction":
    run_prediction(df_full)

elif page == "📖 Help":
    st.header("📖 Help & Instructions")
    st.markdown("""
    Welcome to the Fraud Detection Dashboard!

    **Dashboard**  
    View interactive charts analyzing fraud by transaction type, daily trends, losses, and top senders.

    **Prediction**  
    Train an XGBoost model to predict fraudulent transactions and explore feature importance.

    **Help**  
    Guidance on how to use the dashboard and interpret results.

    **Notes**  
    - Data is from PaySim sample transactions.  
    - The app uses SQLite for data storage and XGBoost for machine learning.  
    - Visualizations are powered by Plotly for smooth interaction.

    Explore, learn, and showcase your data analytics skills!
    """)

# Footer
st.markdown("---")
st.write("Built with ❤️ by Akshay using Streamlit, Plotly, XGBoost, and SQLite.")

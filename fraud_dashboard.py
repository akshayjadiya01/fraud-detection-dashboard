import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb

# ---------- Page Setup ----------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Database Setup ----------
DB_FILE = os.path.join(os.path.dirname(__file__), "fraud_paysim_sample.db")
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Create views if not exist
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
    df_full = pd.read_sql("SELECT * FROM transactions LIMIT 50000;", conn)  # Limit to avoid memory issues
    return df_type, df_daily, df_top, df_loss, df_full

def plot_fraud_by_type(df_type):
    st.subheader("📊 Fraud Distribution by Transaction Type")
    fig = px.bar(df_type, x="type", y="count", color="type",
                 title="Number of Fraudulent Transactions by Type")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def plot_daily_trends(df_daily):
    st.subheader("📈 Fraud Trends Over Time")
    fig = px.line(df_daily, x="step", y="count", title="Daily Fraudulent Transactions")
    fig.update_traces(mode="lines+markers")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def plot_loss_by_type(df_loss):
    st.subheader("💰 Fraud Loss by Transaction Type")
    fig = px.pie(df_loss, names="type", values="total_loss",
                 title="Total Loss Distribution by Type")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def plot_top_senders(df_top):
    st.subheader("🚨 Top Fraudulent Senders")
    fig = px.bar(df_top, x="sender", y="total_loss", color="total_loss",
                 title="Top 10 Fraudulent Senders by Total Loss")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def run_prediction(df_full):
    st.header("🔮 Fraud Prediction")
    st.markdown("Train a model to predict fraudulent transactions and explore its performance.")

    X = df_full.drop(columns=["isFraud"])
    y = df_full["isFraud"]

    X_encoded = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    st.success(f"✅ Model Trained Successfully! ROC AUC Score: {auc:.4f}")

    # Feature Importance
    importance = model.feature_importances_
    features = X_encoded.columns
    fi_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values(by="Importance", ascending=False).head(15)
    st.subheader("📌 Top Features Impacting Fraud")
    st.dataframe(fi_df)

    fig = go.Figure()
    fig.add_trace(go.Bar(y=fi_df["Feature"], x=fi_df["Importance"], orientation='h', marker=dict(color='orange')))
    fig.update_layout(title="Feature Importance", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Classification Report
    st.subheader("📄 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion Matrix
    st.subheader("🧮 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

def show_help():
    st.header("📖 Help & Instructions")
    st.markdown("""
    **Welcome to the Fraud Detection Dashboard!**
    
    ➤ **Dashboard Tab**: Explore visualizations of fraud patterns.  
    ➤ **Prediction Tab**: Train a model and evaluate fraud predictions.  
    ➤ **Help Tab**: Guidance on using the dashboard and interpreting charts.  

    Technical Notes:  
    - SQLite used for database.  
    - XGBoost for predictions.  
    - Plotly and Matplotlib for visualizations.  
    """)

# ---------- Main Application ----------
st.title("🚨 Fraud Detection Dashboard")
st.markdown("Explore fraud patterns and predict fraudulent transactions with ease.")

tabs = st.tabs(["📊 Dashboard", "🔮 Prediction", "📖 Help"])

df_type, df_daily, df_top, df_loss, df_full = get_data()

with tabs[0]:
    plot_fraud_by_type(df_type)
    plot_daily_trends(df_daily)
    plot_loss_by_type(df_loss)
    plot_top_senders(df_top)

with tabs[1]:
    run_prediction(df_full)

with tabs[2]:
    show_help()

st.markdown("---")
st.write("Built with ❤️ by Akshay using Streamlit, Plotly, Matplotlib, XGBoost, and SQLite.")

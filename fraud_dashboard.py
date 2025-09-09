import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# ==============================
# Page setup and styling
# ==============================
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
h1 { font-size: 2.5rem; color: #2C3E50; }
h2 { color: #34495E; }
.sidebar .sidebar-content { background-color: #f0f2f6; }
.stDownloadButton button { background-color: #2C3E50; color: white; }
</style>
""", unsafe_allow_html=True)

# ==============================
# Database connection
# ==============================
DB_PATH = "fraud_paysim.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

def load_view(query):
    return pd.read_sql(query, conn)

# ==============================
# Load data with caching
# ==============================
@st.cache_data
def get_data():
    df_type = load_view("SELECT * FROM v_fraud_by_type;")
    df_daily = load_view("SELECT * FROM v_daily_fraud;")
    df_top = load_view("SELECT * FROM v_top_fraud_senders;")
    df_loss = load_view("SELECT * FROM v_fraud_loss_type;")
    df_full = load_view("SELECT * FROM transactions LIMIT 10000;")
    return df_type, df_daily, df_top, df_loss, df_full

df_type, df_daily, df_top, df_loss, df_full = get_data()

# ==============================
# Header
# ==============================
st.title("🚨 Fraud Detection Dashboard")
st.markdown("Explore fraud patterns and predictions with interactive visualizations.")

# ==============================
# Navigation Tabs
# ==============================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview 📊", "Trends 📈", "Top Senders 🚨", "Losses 💸", "Prediction 🤖"])

# ==============================
# Overview
# ==============================
with tab1:
    st.header("Fraud Rate by Transaction Type")
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = px.bar(df_type, x="type", y="fraud_rate",
                     color="fraud_rate", color_continuous_scale="Viridis",
                     labels={"fraud_rate": "Fraud Rate (%)"})
        fig.update_layout(title="Fraud Rate by Type",
                          xaxis_title="Transaction Type", yaxis_title="Rate (%)",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Total Types", df_type.shape[0])
        st.metric("Max Fraud Rate", f"{df_type['fraud_rate'].max():.2f}%")

    st.dataframe(df_type.style.highlight_max(subset=["fraud_rate"], color="lightgreen"), use_container_width=True)

# ==============================
# Trends
# ==============================
with tab2:
    st.header("Daily Fraud Trends")
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = px.line(df_daily, x="day", y="fraud_rate", markers=True,
                      labels={"day": "Day", "fraud_rate": "Fraud Rate (%)"})
        fig.update_layout(title="Daily Fraud Trends",
                          xaxis_title="Day", yaxis_title="Rate (%)",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Total Days", df_daily.shape[0])
        st.metric("Peak Rate", f"{df_daily['fraud_rate'].max():.2f}%")

    st.dataframe(df_daily, use_container_width=True)

# ==============================
# Top Senders
# ==============================
with tab3:
    st.header("Top Fraudulent Senders")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(df_top.style.format({"total_fraud_amount": "${:,.2f}"}), use_container_width=True)
    with col2:
        st.metric("Senders Listed", df_top.shape[0])
        st.metric("Max Fraud Amount", f"${df_top['total_fraud_amount'].max():,.2f}")

# ==============================
# Losses
# ==============================
with tab4:
    st.header("Fraud Loss by Transaction Type")
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = px.bar(df_loss, x="type", y="fraud_loss",
                     color="fraud_loss", color_continuous_scale="Magma",
                     labels={"fraud_loss": "Fraud Loss ($)"})
        fig.update_layout(title="Fraud Loss by Type",
                          xaxis_title="Transaction Type", yaxis_title="Loss ($)",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Types of Losses", df_loss.shape[0])
        st.metric("Maximum Loss", f"${df_loss['fraud_loss'].max():,.2f}")

    st.dataframe(df_loss.style.format({"fraud_loss": "${:,.2f}"}), use_container_width=True)

# ==============================
# Prediction Page (Improved)
# ==============================
with tab5:
    st.header("Machine Learning Prediction with XGBoost")
    st.markdown("""
    This section allows you to predict whether a transaction is fraudulent or not based on transaction details.
    Enter the transaction details below and see the prediction instantly.
    """)

    df = df_full.copy()
    if df.empty:
        st.error("Sample data not available for training.")
    else:
        le = LabelEncoder()
        df['type_encoded'] = le.fit_transform(df['type'])
        features = ['amount', 'type_encoded', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        target = 'isFraud'

        X = df[features]
        y = df[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        st.subheader("Model Evaluation")
        st.metric("ROC AUC Score", f"{auc:.2f}")
        st.text("Classification Report:")
        st.json(report)

        st.subheader("Make a Prediction")
        with st.form("predict_form"):
            amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0, step=100.0)
            type_input = st.selectbox("Transaction Type", df['type'].unique())
            oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, value=5000.0, step=100.0)
            newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, value=4000.0, step=100.0)
            oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=10000.0, step=100.0)
            newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=9000.0, step=100.0)
            submit = st.form_submit_button("Predict Fraud")

        if submit:
            type_encoded = le.transform([type_input])[0]
            input_data = pd.DataFrame({
                'amount': [amount],
                'type_encoded': [type_encoded],
                'oldbalanceOrg': [oldbalanceOrg],
                'newbalanceOrig': [newbalanceOrig],
                'oldbalanceDest': [oldbalanceDest],
                'newbalanceDest': [newbalanceDest]
            })
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            result = "Fraudulent Transaction ❗️" if prediction == 1 else "Legitimate Transaction ✅"
            st.subheader("Prediction Result")
            st.success(result)

        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        fig = px.bar(importance, x="Feature", y="Importance",
                     title="Feature Importance", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# Download Section
# ==============================
st.markdown("---")
st.header("📥 Download Reports")

cols = st.columns(4)
with cols[0]:
    csv_type = df_type.to_csv(index=False).encode()
    st.download_button("Download Fraud Rate", csv_type, "fraud_rate.csv", "text/csv")
with cols[1]:
    csv_daily = df_daily.to_csv(index=False).encode()
    st.download_button("Download Daily Trends", csv_daily, "daily_trends.csv", "text/csv")
with cols[2]:
    csv_top = df_top.to_csv(index=False).encode()
    st.download_button("Download Top Senders", csv_top, "top_senders.csv", "text/csv")
with cols[3]:
    csv_loss = df_loss.to_csv(index=False).encode()
    st.download_button("Download Loss Report", csv_loss, "fraud_loss.csv", "text/csv")

# ==============================
# Help Section at the Bottom
# ==============================
st.markdown("---")
with st.expander("ℹ️ Help / Documentation"):
    st.markdown("""
    **Welcome to the Fraud Detection Dashboard!**  

    ✅ **Overview:** Explore fraud rates by transaction type with metrics and charts.  
    ✅ **Trends:** Analyze daily fraud patterns with line graphs.  
    ✅ **Top Senders:** Identify accounts with suspicious activity.  
    ✅ **Losses:** Review financial loss by transaction type.  
    ✅ **Prediction:** Enter transaction details and get real-time fraud predictions using an XGBoost model.

    **Instructions:**  
    1. Navigate through the tabs to explore insights.  
    2. In the Prediction tab, enter transaction details and click 'Predict Fraud'.  
    3. Download reports for further analysis.

    **Technical Info:**  
    - Built with Python, Streamlit, Plotly, SQLite, and XGBoost.  
    - Features are scaled and encoded before training.  
    - Performance evaluated with ROC AUC and classification metrics.

    This dashboard is designed to be portfolio-ready, showcasing data analysis, visualization, and machine learning deployment skills.
    """)

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("Built with ❤️ using **Streamlit**, **Plotly**, **SQLite**, **XGBoost**, and **Python**")

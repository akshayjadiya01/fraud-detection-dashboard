# 🚨 Fraud Detection Dashboard

![Streamlit](https://img.shields.io/badge/Made_with-Streamlit-green) 
![Python](https://img.shields.io/badge/Made_with-Python-blue) 
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange) 
![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey)

Welcome to the **Fraud Detection Dashboard**, an interactive web application that provides insights into fraudulent transactions and allows users to predict fraud using machine learning—all in a clean, professional interface!  

This project demonstrates **data analysis, SQL integration, advanced visualization, and machine learning**, making it perfect for showcasing in your portfolio as a data analyst project.  

---

## 📊 Features

✅ **Interactive Dashboard**  
- Visualize fraud counts by transaction type  
- Explore daily fraud trends with dual-axis charts (fraud count + total loss)  
- Review fraud loss distribution and top fraudulent senders  

✅ **Machine Learning Prediction**  
- Predict fraudulent transactions using XGBoost  
- View ROC AUC score, confusion matrix, and feature importance  
- Make your analysis actionable and understandable  

✅ **Professional UI & UX**  
- Clean dark-themed layout  
- Tab navigation (Dashboard, Prediction, Help)  
- Help section with clear instructions  

---

## 📂 Screenshots

### Dashboard Overview
<img width="1770" height="707" alt="dashboard_overview" src="https://github.com/user-attachments/assets/b83e6b3d-15c4-46b2-b36a-a603a3dc5f8a" />


### Fraud Trends Over Time
<img width="1750" height="519" alt="fraud_trends" src="https://github.com/user-attachments/assets/a05c0be7-c28a-47a0-adc5-bf00a18d13e7" />


### Fraud by Type & Loss
![Fraud Type & Loss](screenshots/fraud_type_loss.png)

### Top Fraudulent Senders
![Top Fraudulent Senders](screenshots/top_senders.png)

### Prediction Tab
![Prediction Tab](screenshots/prediction_tab.png)

### Help / Instructions Tab
![Help Tab](screenshots/help_tab.png)

> _Tip: Place all screenshots in a `screenshots/` folder for easy linking._  

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/USERNAME/fraud-detection-dashboard.git
cd fraud-detection-dashboard


```
### 2️⃣ Create a virtual environment and install dependencies
```bash
python -m venv venv
venv\Scripts\activate    # On Windows
# OR
source venv/bin/activate # On Mac/Linux

pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app
```bash
streamlit run fraud_dashboard.py
```

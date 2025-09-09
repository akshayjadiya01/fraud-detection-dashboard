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
<img width="1382" height="547" alt="Fraud by Type   Loss" src="https://github.com/user-attachments/assets/facdbf22-b0e4-4e27-8995-3c70e6aedd0a" />


### Top Fraudulent Senders
<img width="1311" height="556" alt="Top Fraudulent Senders" src="https://github.com/user-attachments/assets/2167188b-3f3d-4158-b19d-9d64b6e63933" />


### Prediction Tab
<img width="1857" height="885" alt="Prediction Tab" src="https://github.com/user-attachments/assets/3e2285b8-f27c-45a8-be8c-d880b6d12a6c" />


### Help / Instructions Tab
<img width="1372" height="626" alt="help_tab" src="https://github.com/user-attachments/assets/ae306764-6f79-4743-a4af-ecb1740b3e7c" />




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

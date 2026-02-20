# 💳 Credit Card Fraud Detection

A Machine Learning project to detect fraudulent credit card transactions using **Logistic Regression** and **Random Forest**, deployed with **Streamlit**.

---

## 🚀 Project Overview

Fraud transactions are rare but critical.  
This is an **imbalanced classification problem** where detecting fraud (Recall) is more important than overall accuracy.

---

## 🖥️ Project Demo

### 🏠 Home Interface

![Home UI](assets/ui_home.png)

---

### 📊 Prediction Output

![Prediction Output 1](assets/prediction_output1.png)

![Prediction Output 2](assets/prediction_output2.png)

---

## 🧠 Problem Statement

Fraud transactions occur very rarely compared to legitimate transactions.  
Traditional accuracy is misleading in such cases.

> 🎯 Catching fraud is more important than overall accuracy.

Therefore, **Recall** is prioritized as the primary evaluation metric.

---

## ⚙️ Tech Stack

- Python
- Scikit-learn
- SMOTE (Imbalanced-learn)
- Pandas & NumPy
- Streamlit
- Matplotlib

---

## 📂 Dataset

- Kaggle Credit Card Fraud Dataset
- 284,807 transactions
- Only 0.17% are fraudulent

---

## 📊 Evaluation Metrics

- Recall (Primary Focus)
- F1-Score
- Confusion Matrix

---

## ▶️ Run Locally

```bash
# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

👨‍💻 Author  
**Kshitij Kamble**

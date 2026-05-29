# 🏦 Banking Risk Intelligence System

AI-powered web application for real-time fraud detection 
and loan default prediction using Machine Learning.

## 🔗 Live Demo
[Click here to view the app](YOUR_STREAMLIT_URL)

## 🎯 Features
- **Fraud Detection** — XGBoost model with ROC-AUC: 0.9739
- **Loan Default Prediction** — Risk scoring with LOW/MEDIUM/HIGH classification
- **SHAP Explainability** — Model decision transparency
- **Interactive Dashboard** — Real-time predictions via Streamlit

## 🛠️ Tech Stack
Python | XGBoost | Scikit-learn | SHAP | Streamlit | Pandas | NumPy

## 📊 Model Performance
| Model | ROC-AUC | Recall |
|-------|---------|--------|
| Fraud Detection | 0.9739 | 88% |
| Loan Default | ~0.82 | ~76% |

## 🚀 Run Locally
pip install -r requirements.txt
streamlit run app.py
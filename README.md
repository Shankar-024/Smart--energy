# ⚡ Smart Energy Consumption Prediction & Optimization System

> **AIML Department Internal Hackathon 2025**  
> A full-stack ML web application for predicting, classifying, and optimizing household energy consumption.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📁 Project Structure

```
smart_energy/
├── app.py                 # Flask web application
├── train.py               # ML training pipeline
├── requirements.txt       # Python dependencies
├── Procfile               # Gunicorn entry for Render
├── render.yaml            # Render deployment config
├── models/                # Saved ML models (auto-generated)
│   ├── linear_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── kmeans.pkl
│   ├── scaler.pkl
│   ├── label_encoder_eff.pkl
│   ├── feature_cols.pkl
│   └── metrics.json
├── data/
│   ├── synthetic_energy_data.csv   # Generated dataset
│   └── uploads/                    # User-uploaded CSVs
├── static/
│   └── css/
│       └── style.css
└── templates/
    ├── base.html
    ├── index.html
    ├── data.html
    ├── predict.html
    ├── visualize.html
    └── metrics.html
```

---

## 🧠 Machine Learning Models

| Model | Task | Target | Evaluation |
|-------|------|--------|------------|
| Linear Regression | Regression | Energy Consumption (kWh) | R², MAE, RMSE |
| Decision Tree | Binary Classification | High Usage (Yes/No) | Accuracy, Precision, Recall, F1 |
| KNN (k=7) | Multi-class Classification | Efficiency Category | Accuracy, Precision, Recall, F1 |
| K-Means (k=4) | Clustering | Household Segments | Inertia, Cluster Profiles |

---

## 📊 Synthetic Data Logic

**Formula:**
```
Energy_Consumption = (
    2.0 × Household_Size +
    0.8 × Appliance_Count +
    0.15 × Avg_Temperature +
    0.5 × Working_Hours
) × weekend_boost − 0.9 × Solar_Usage + noise
```

**Key correlations engineered:**
- Household size & appliance count → baseline load
- Temperature → HVAC demand (AC/Heating)
- Solar usage → negative offset (reduces grid consumption)
- Weekend → +10% boost in usage
- Previous consumption → autocorrelation (~90% correlated)
- Gaussian noise (σ=2.5) → realistic variability

**Derived labels:**
- `High_Usage` = top 40% of consumption threshold
- `Efficiency_Category` = Low/Medium/High based on kWh per person ratio

---

## 🚀 Local Setup

```bash
# 1. Clone / unzip project
cd smart_energy

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models (generates dataset + saves models)
python train.py

# 5. Run Flask app
python app.py
# → Open http://localhost:5000
```

---

## ☁️ Deployment on Render

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit — SmartEnergy AI"
git remote add origin https://github.com/YOUR_USERNAME/smart-energy-ai.git
git push -u origin main
```

### Step 2: Deploy on Render
1. Go to [render.com](https://render.com) → **New Web Service**
2. Connect your GitHub repo
3. Configure:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt && python train.py`
   - **Start Command**: `gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT --timeout 120`
4. Click **Deploy** — Render auto-detects `render.yaml`

> ✅ The build step runs `train.py` automatically — models are ready before the app starts!

---

## 🌐 Application Pages

| Route | Description |
|-------|-------------|
| `/` | Home — project overview, team, features |
| `/data` | Dataset — generate synthetic or upload CSV |
| `/predict` | Prediction form — all 4 models in real-time |
| `/visualize` | Charts — distribution, correlations, clusters, confusion matrices |
| `/metrics` | Model performance — R², Accuracy, F1, confusion matrices |

---

## 💡 Innovation Ideas (Judge Impressors)

1. **Explainable AI**: Add SHAP values to show which feature drove each prediction
2. **Time-series forecasting**: Add LSTM/Prophet for 7-day consumption forecast
3. **What-if simulator**: Slider UI to see how changing solar panels affects bill
4. **Anomaly detection**: Isolation Forest to flag unusual consumption spikes
5. **Cost optimizer**: LP solver to recommend appliance schedules minimizing bill
6. **Carbon footprint**: Map kWh → CO₂ emissions with offset suggestions
7. **API endpoint**: `/api/quick_predict` already built — can integrate with IoT devices

---

## 👥 Team

| Name | Role |
|------|------|
| Alex Kumar | ML Engineer |
| Priya Sharma | Data Scientist |
| Rahul Verma | Full Stack Developer |
| Neha Singh | AI Researcher |

---

## 📄 License

MIT License — Free to use and modify for educational purposes.

---

*Built with ❤️ for the AIML Hackathon 2025*

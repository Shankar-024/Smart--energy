"""
Smart Energy Consumption Prediction & Optimization System
==========================================================
Training Script - Generates synthetic data and trains all ML models.
Models: Linear Regression, Decision Tree, KNN, K-Means
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def generate_synthetic_data(n=1500, seed=42):
    """
    Generate realistic energy-consumption dataset with meaningful correlations:
    - Larger households, more appliances → more consumption
    - Higher temperature → more AC usage → more consumption
    - Solar usage → reduces net consumption
    - Weekends tend to have different usage patterns
    - Previous consumption has strong autocorrelation
    """
    rng = np.random.default_rng(seed)

    # ── Independent features ──────────────────────────────────────────────────
    household_size    = rng.integers(1, 8,   size=n).astype(float)
    appliance_count   = rng.integers(2, 20,  size=n).astype(float)
    avg_temperature   = rng.uniform(10, 45,  size=n)          # Celsius
    working_hours     = rng.uniform(0, 16,   size=n)           # hrs/day active usage
    solar_usage       = rng.uniform(0, 10,   size=n)           # kWh generated
    electricity_tariff= rng.uniform(3.5, 12, size=n)           # ₹/kWh
    day_type          = rng.choice(["Weekday", "Weekend"], size=n, p=[0.71, 0.29])

    # ── Derived / realistic consumption formula ────────────────────────────────
    # Base load depends on household size and appliances
    base_load = (
        2.0 * household_size +
        0.8 * appliance_count +
        0.15 * avg_temperature +          # cooling/heating demand
        0.5  * working_hours
    )
    # Solar offsets consumption; weekends add ~10%
    weekend_boost = np.where(day_type == "Weekend", 1.10, 1.0)
    # Add Gaussian noise for realism
    noise = rng.normal(0, 2.5, size=n)

    energy_consumption = (base_load * weekend_boost) - (0.9 * solar_usage) + noise
    energy_consumption = np.clip(energy_consumption, 1, 120)  # physical bounds

    # ── Previous-day consumption (strong autocorrelation) ─────────────────────
    prev_noise = rng.normal(0, 3, size=n)
    previous_consumption = np.clip(energy_consumption * 0.9 + prev_noise, 1, 120)

    df = pd.DataFrame({
        "Household_Size":       household_size.astype(int),
        "Appliance_Count":      appliance_count.astype(int),
        "Avg_Temperature":      avg_temperature.round(2),
        "Working_Hours":        working_hours.round(2),
        "Solar_Usage":          solar_usage.round(2),
        "Electricity_Tariff":   electricity_tariff.round(2),
        "Day_Type":             day_type,
        "Previous_Consumption": previous_consumption.round(2),
        "Energy_Consumption":   energy_consumption.round(2),
    })

    # ── Derived labels ─────────────────────────────────────────────────────────
    # High Usage: top 40 % of consumption
    threshold = df["Energy_Consumption"].quantile(0.60)
    df["High_Usage"] = (df["Energy_Consumption"] >= threshold).astype(int)

    # Efficiency Category based on consumption-per-person ratio
    ratio = df["Energy_Consumption"] / df["Household_Size"]
    df["Efficiency_Category"] = pd.cut(
        ratio,
        bins=[0, 8, 16, np.inf],
        labels=["High", "Medium", "Low"]   # Low efficiency → High consumption/person
    ).astype(str)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def encode_and_scale(df):
    """Return feature matrix X (scaled) and raw X for tree models."""
    df = df.copy()
    df["Day_Type_Enc"] = (df["Day_Type"] == "Weekend").astype(int)

    feature_cols = [
        "Household_Size", "Appliance_Count", "Avg_Temperature",
        "Working_Hours", "Solar_Usage", "Electricity_Tariff",
        "Day_Type_Enc", "Previous_Consumption"
    ]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, scaler, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════
def train_all(df):
    metrics = {}

    X_raw, X_scaled, scaler, feature_cols = encode_and_scale(df)

    # ── Save scaler ────────────────────────────────────────────────────────────
    joblib.dump(scaler,       os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))

    # ─────────────────────────────────────────────────────────────────────────
    # 3-A  LINEAR REGRESSION  →  Predict Energy_Consumption
    # ─────────────────────────────────────────────────────────────────────────
    y_reg = df["Energy_Consumption"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    y_pred_lr = lr.predict(X_te)

    metrics["regression"] = {
        "r2"  : round(r2_score(y_te, y_pred_lr), 4),
        "mae" : round(mean_absolute_error(y_te, y_pred_lr), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_te, y_pred_lr)), 4),
    }
    joblib.dump(lr, os.path.join(MODEL_DIR, "linear_regression.pkl"))
    print(f"[Regression]  R²={metrics['regression']['r2']}  MAE={metrics['regression']['mae']}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3-B  DECISION TREE CLASSIFIER  →  High Usage (0/1)
    # ─────────────────────────────────────────────────────────────────────────
    y_clf = df["High_Usage"].values
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_raw, y_clf, test_size=0.2, random_state=42)

    dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)
    dt.fit(X_tr2, y_tr2)
    y_pred_dt = dt.predict(X_te2)

    cm_dt = confusion_matrix(y_te2, y_pred_dt).tolist()
    metrics["decision_tree"] = {
        "accuracy" : round(accuracy_score(y_te2,  y_pred_dt), 4),
        "precision": round(precision_score(y_te2, y_pred_dt, zero_division=0), 4),
        "recall"   : round(recall_score(y_te2,    y_pred_dt, zero_division=0), 4),
        "f1"       : round(f1_score(y_te2,        y_pred_dt, zero_division=0), 4),
        "confusion_matrix": cm_dt,
    }
    joblib.dump(dt, os.path.join(MODEL_DIR, "decision_tree.pkl"))
    print(f"[DecisionTree] Acc={metrics['decision_tree']['accuracy']}  F1={metrics['decision_tree']['f1']}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3-C  KNN CLASSIFIER  →  Efficiency_Category (Low/Medium/High)
    # ─────────────────────────────────────────────────────────────────────────
    le = LabelEncoder()
    y_eff = le.fit_transform(df["Efficiency_Category"].values)
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder_eff.pkl"))

    X_tr3, X_te3, y_tr3, y_te3 = train_test_split(X_scaled, y_eff, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn.fit(X_tr3, y_tr3)
    y_pred_knn = knn.predict(X_te3)

    cm_knn = confusion_matrix(y_te3, y_pred_knn).tolist()
    metrics["knn"] = {
        "accuracy" : round(accuracy_score(y_te3,  y_pred_knn), 4),
        "precision": round(precision_score(y_te3, y_pred_knn, average="weighted", zero_division=0), 4),
        "recall"   : round(recall_score(y_te3,    y_pred_knn, average="weighted", zero_division=0), 4),
        "f1"       : round(f1_score(y_te3,        y_pred_knn, average="weighted", zero_division=0), 4),
        "confusion_matrix": cm_knn,
        "classes"  : le.classes_.tolist(),
    }
    joblib.dump(knn, os.path.join(MODEL_DIR, "knn.pkl"))
    print(f"[KNN]          Acc={metrics['knn']['accuracy']}  F1={metrics['knn']['f1']}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3-D  K-MEANS CLUSTERING  →  Consumption segments
    # ─────────────────────────────────────────────────────────────────────────
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)

    df["Cluster"] = cluster_labels
    cluster_summary = (
        df.groupby("Cluster")["Energy_Consumption"]
          .agg(["mean", "min", "max", "count"])
          .round(2)
          .reset_index()
    )
    cluster_names = {
        int(row["Cluster"]): _name_cluster(row["mean"])
        for _, row in cluster_summary.iterrows()
    }

    metrics["kmeans"] = {
        "inertia"       : round(km.inertia_, 2),
        "n_clusters"    : 4,
        "cluster_summary": cluster_summary.to_dict(orient="records"),
        "cluster_names" : cluster_names,
    }
    joblib.dump(km,            os.path.join(MODEL_DIR, "kmeans.pkl"))
    print(f"[KMeans]       Inertia={metrics['kmeans']['inertia']}")

    # Save dataset with cluster labels
    df.to_csv(os.path.join(DATA_DIR, "synthetic_energy_data.csv"), index=False)

    # Save all metrics for the Flask app
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ All models saved to ./models/")
    print("✅ Dataset saved to ./data/synthetic_energy_data.csv")
    return metrics


def _name_cluster(mean_consumption):
    if mean_consumption < 20:
        return "Eco Saver 🌱"
    elif mean_consumption < 40:
        return "Moderate User ⚡"
    elif mean_consumption < 60:
        return "High Consumer 🔥"
    else:
        return "Energy Intensive 🏭"


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Smart Energy Prediction — Training Pipeline")
    print("=" * 60)
    df = generate_synthetic_data(n=1500)
    print(f"\nDataset shape : {df.shape}")
    print(df.describe().to_string())
    print()
    train_all(df)

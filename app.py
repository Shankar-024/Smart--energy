"""
Smart Energy Consumption Prediction & Optimization System
==========================================================
Flask Web Application
"""

import os
import io
import json
import base64
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "energy_hack_2025_secret"
app.config["UPLOAD_FOLDER"]  = "data/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ── Load models ────────────────────────────────────────────────────────────────
MODEL_DIR = "models"

def load_models():
    models = {}
    try:
        models["lr"]      = joblib.load(os.path.join(MODEL_DIR, "linear_regression.pkl"))
        models["dt"]      = joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl"))
        models["knn"]     = joblib.load(os.path.join(MODEL_DIR, "knn.pkl"))
        models["km"]      = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
        models["scaler"]  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        models["le_eff"]  = joblib.load(os.path.join(MODEL_DIR, "label_encoder_eff.pkl"))
        models["f_cols"]  = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
        with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
            models["metrics"] = json.load(f)
        print("✅ All models loaded.")
    except FileNotFoundError as e:
        print(f"⚠️  Model not found: {e}. Run train.py first.")
    return models

MODELS = load_models()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def fig_to_b64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64


def preprocess_input(form):
    """Convert form data → scaled numpy array for prediction."""
    vals = {
        "Household_Size"      : float(form["household_size"]),
        "Appliance_Count"     : float(form["appliance_count"]),
        "Avg_Temperature"     : float(form["avg_temperature"]),
        "Working_Hours"       : float(form["working_hours"]),
        "Solar_Usage"         : float(form["solar_usage"]),
        "Electricity_Tariff"  : float(form["electricity_tariff"]),
        "Day_Type_Enc"        : 1.0 if form["day_type"] == "Weekend" else 0.0,
        "Previous_Consumption": float(form["previous_consumption"]),
    }
    X_raw    = np.array([[vals[c] for c in MODELS["f_cols"]]]).astype(float)
    X_scaled = MODELS["scaler"].transform(X_raw)
    return X_raw, X_scaled, vals


def energy_tips(consumption, solar, tariff):
    tips = []
    if consumption > 50:
        tips.append("🔴 High consumption detected — consider scheduling heavy appliances off-peak.")
    if solar < 3:
        tips.append("☀️  Low solar usage — installing more panels can cut bills by 20-40%.")
    if tariff > 8:
        tips.append("💡 High tariff zone — shift usage to nights/weekends for savings.")
    if consumption < 20:
        tips.append("🌱 Great efficiency! You're in the top 15% of eco-friendly households.")
    tips.append("🧊 Setting AC to 24°C instead of 18°C saves ~6% electricity.")
    tips.append("⚡ LED bulbs use 75% less energy than incandescent lights.")
    return tips


def cluster_name(cid, summary):
    names = {0: "Eco Saver 🌱", 1: "Moderate User ⚡", 2: "High Consumer 🔥", 3: "Energy Intensive 🏭"}
    # Re-sort by mean consumption so labels are consistent
    sorted_s = sorted(summary, key=lambda x: x["mean"])
    name_map = {s["Cluster"]: list(names.values())[i] for i, s in enumerate(sorted_s)}
    return name_map.get(int(cid), f"Cluster {cid}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════
DARK_BG = "#0d1117"
ACCENT  = "#00e5ff"
GRID_C  = "#1e2a38"

def set_dark_style(fig, ax_list):
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax_list if isinstance(ax_list, (list, np.ndarray)) else [ax_list]):
        ax.set_facecolor("#111927")
        ax.tick_params(colors="#aab4be", labelsize=9)
        ax.xaxis.label.set_color("#aab4be")
        ax.yaxis.label.set_color("#aab4be")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)
        ax.grid(color=GRID_C, linestyle="--", linewidth=0.5)


def chart_consumption_dist(df):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    set_dark_style(fig, ax)
    ax.hist(df["Energy_Consumption"], bins=40, color=ACCENT,
            edgecolor="#005f73", alpha=0.85)
    ax.set_xlabel("Energy Consumption (kWh)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Energy Consumption")
    return fig_to_b64(fig)


def chart_corr_heatmap(df):
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    set_dark_style(fig, ax)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                annot_kws={"size": 7}, linewidths=0.5, linecolor=GRID_C,
                cbar_kws={"shrink": 0.7})
    ax.set_title("Feature Correlation Heatmap")
    ax.tick_params(labelsize=7, colors="#aab4be")
    fig.patch.set_facecolor(DARK_BG)
    return fig_to_b64(fig)


def chart_scatter_solar(df):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    set_dark_style(fig, ax)
    scatter = ax.scatter(df["Solar_Usage"], df["Energy_Consumption"],
                         c=df["Energy_Consumption"], cmap="plasma",
                         alpha=0.5, s=10, edgecolors="none")
    plt.colorbar(scatter, ax=ax, label="kWh").ax.yaxis.label.set_color("#aab4be")
    ax.set_xlabel("Solar Usage (kWh)")
    ax.set_ylabel("Net Consumption (kWh)")
    ax.set_title("Solar Usage vs Energy Consumption")
    return fig_to_b64(fig)


def chart_cluster_scatter(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    set_dark_style(fig, ax)
    palette = ["#00e5ff", "#ff6b6b", "#ffd166", "#06d6a0"]
    for cid in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == cid]
        ax.scatter(sub["Appliance_Count"], sub["Energy_Consumption"],
                   color=palette[cid % 4], alpha=0.55, s=18, label=f"Cluster {cid}")
    ax.set_xlabel("Appliance Count")
    ax.set_ylabel("Energy Consumption (kWh)")
    ax.set_title("K-Means Cluster Plot")
    ax.legend(fontsize=8, facecolor="#111927", labelcolor="white", framealpha=0.7)
    return fig_to_b64(fig)


def chart_daytype_box(df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    set_dark_style(fig, ax)
    wkday = df[df["Day_Type"] == "Weekday"]["Energy_Consumption"]
    wkend = df[df["Day_Type"] == "Weekend"]["Energy_Consumption"]
    bp = ax.boxplot([wkday, wkend], patch_artist=True,
                    medianprops=dict(color="white", linewidth=2))
    colors = [ACCENT, "#ff6b6b"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticklabels(["Weekday", "Weekend"])
    ax.set_ylabel("Energy Consumption (kWh)")
    ax.set_title("Consumption by Day Type")
    return fig_to_b64(fig)


def chart_confusion_matrix(cm, labels, title):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    set_dark_style(fig, ax)
    cmap = plt.cm.Blues
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, color="#aab4be")
    ax.set_yticklabels(labels, fontsize=8, color="#aab4be")
    ax.set_xlabel("Predicted", color="#aab4be")
    ax.set_ylabel("Actual", color="#aab4be")
    ax.set_title(title, color="white", fontsize=10)
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax)
    return fig_to_b64(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data", methods=["GET", "POST"])
def data_page():
    """Show dataset — either synthetic or uploaded CSV."""
    df_path = "data/synthetic_energy_data.csv"
    preview_html = None
    stats_html   = None

    if request.method == "POST":
        # CSV Upload
        if "csv_file" in request.files:
            f = request.files["csv_file"]
            if f.filename.endswith(".csv"):
                fname = secure_filename(f.filename)
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
                f.save(save_path)
                df_path = save_path
                session["dataset_ready"] = True   # ← unlock visualizations
                flash("✅ CSV uploaded successfully!", "success")
            else:
                flash("❌ Please upload a valid CSV file.", "danger")
                return redirect(url_for("data_page"))

        # Generate synthetic
        elif "generate" in request.form:
            from train import generate_synthetic_data, train_all
            n = int(request.form.get("n_records", 1500))
            n = max(500, min(n, 5000))
            df = generate_synthetic_data(n=n)
            train_all(df)
            global MODELS
            MODELS = load_models()
            session["dataset_ready"] = True   # ← unlock visualizations
            flash(f"✅ Generated {n} records & retrained all models!", "success")

    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        preview_html = df.head(20).to_html(
            classes="table table-dark table-hover table-sm", index=False,
            border=0
        )
        stats_html = df.describe().round(2).to_html(
            classes="table table-dark table-hover table-sm", border=0
        )

    return render_template("data.html", preview=preview_html, stats=stats_html)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    tips   = []

    if request.method == "POST":
        if not MODELS:
            flash("⚠️  Models not loaded. Please generate data first.", "warning")
            return redirect(url_for("data_page"))

        try:
            X_raw, X_scaled, vals = preprocess_input(request.form)

            # Predictions
            consumption   = round(float(MODELS["lr"].predict(X_scaled)[0]), 2)
            high_usage    = int(MODELS["dt"].predict(X_raw)[0])
            eff_enc       = int(MODELS["knn"].predict(X_scaled)[0])
            efficiency    = MODELS["le_eff"].inverse_transform([eff_enc])[0]
            cluster_id    = int(MODELS["km"].predict(X_scaled)[0])
            c_summary     = MODELS["metrics"]["kmeans"]["cluster_summary"]
            cluster_label = cluster_name(cluster_id, c_summary)

            tips = energy_tips(consumption, vals["Solar_Usage"], vals["Electricity_Tariff"])
            monthly_cost = round(consumption * 30 * vals["Electricity_Tariff"], 2)

            result = {
                "consumption"  : consumption,
                "high_usage"   : "Yes 🔴" if high_usage else "No 🟢",
                "high_usage_raw": high_usage,
                "efficiency"   : efficiency,
                "cluster_label": cluster_label,
                "cluster_id"   : cluster_id,
                "monthly_cost" : monthly_cost,
            }
        except Exception as e:
            flash(f"❌ Prediction error: {e}", "danger")

    return render_template("predict.html", result=result, tips=tips)


@app.route("/visualize")
def visualize():
    # Only show charts after user has explicitly generated or uploaded a dataset
    if not session.get("dataset_ready"):
        return render_template("visualize.html", charts={}, cm_charts={}, metrics={}, locked=True)

    df_path = "data/synthetic_energy_data.csv"
    if not os.path.exists(df_path):
        flash("⚠️  Dataset not found. Generate it first!", "warning")
        return redirect(url_for("data_page"))

    df = pd.read_csv(df_path)

    charts = {
        "dist"    : chart_consumption_dist(df),
        "corr"    : chart_corr_heatmap(df),
        "solar"   : chart_scatter_solar(df),
        "daytype" : chart_daytype_box(df),
    }

    if "Cluster" in df.columns:
        charts["cluster"] = chart_cluster_scatter(df)

    # Confusion matrices
    m = MODELS.get("metrics", {})
    cm_charts = {}
    if "decision_tree" in m:
        cm_dt = np.array(m["decision_tree"]["confusion_matrix"])
        cm_charts["dt"] = chart_confusion_matrix(cm_dt, ["Low", "High"], "Decision Tree CM")
    if "knn" in m:
        cm_knn = np.array(m["knn"]["confusion_matrix"])
        labels = m["knn"].get("classes", ["High", "Low", "Medium"])
        cm_charts["knn"] = chart_confusion_matrix(cm_knn, labels, "KNN CM")

    return render_template("visualize.html", charts=charts, cm_charts=cm_charts,
                           metrics=m)


@app.route("/metrics")
def metrics_page():
    m = MODELS.get("metrics", {})
    return render_template("metrics.html", metrics=m)


@app.route("/api/quick_predict", methods=["POST"])
def api_predict():
    """JSON API for quick predictions (optional frontend use)."""
    data = request.json or {}
    try:
        X_raw, X_scaled, vals = preprocess_input(data)
        return jsonify({
            "consumption": round(float(MODELS["lr"].predict(X_scaled)[0]), 2),
            "high_usage" : int(MODELS["dt"].predict(X_raw)[0]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

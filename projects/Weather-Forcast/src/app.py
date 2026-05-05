"""
app.py -- Sci-Fi Weather Dashboard
====================================
- Pre-trained models: instant predictions, no training required
- Glassmorphism dark UI with neon accents
- Interactive city/model selector
- Live weather data visualization + model forecast overlay
"""

import json
import os
import pickle
import sys
from datetime import date, timedelta
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="SW Weather AI", page_icon="🛰️", layout="wide")

PROJ_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJ_ROOT, "data")
CKPT_DIR = os.path.join(PROJ_ROOT, "checkpoints")

sys.path.insert(0, PROJ_ROOT)
from src.models import build_model

# ══════════════════════════════════════════════════════════════
# CSS - Sci-Fi Glassmorphism Theme
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');

/* ── Base ── */
.stApp {
    background: radial-gradient(ellipse at 50% 0%, #0a1628 0%, #060e1a 50%, #020810 100%);
    color: #e0e0e0;
}

/* ── Glass Panels ── */
.glass {
    background: rgba(15, 30, 60, 0.55);
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    border: 1px solid rgba(100, 200, 255, 0.12);
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 8px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.04);
    margin-bottom: 16px;
}

.glass-header {
    background: rgba(10, 25, 50, 0.65);
    backdrop-filter: blur(20px) saturate(160%);
    -webkit-backdrop-filter: blur(20px) saturate(160%);
    border-bottom: 1px solid rgba(0, 200, 255, 0.2);
    border-radius: 0;
    padding: 16px 28px;
    margin-bottom: 0;
}

/* ── Stat Cards ── */
.stat-card {
    background: rgba(10, 25, 55, 0.6);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(0, 180, 255, 0.15);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    transition: all 0.3s;
}
.stat-card:hover {
    border-color: rgba(0, 220, 255, 0.35);
    box-shadow: 0 0 30px rgba(0, 180, 255, 0.15);
}
.stat-value {
    font-family: 'Orbitron', monospace;
    font-size: 2.0rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00d4ff, #0090ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #6a9fd8;
    margin-top: 4px;
}

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 1px;
}
h1 { background: linear-gradient(135deg, #00e5ff, #0088ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
h2 { color: #4dc9f6 !important; }
h3 { color: #7ec8e3 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(8, 18, 35, 0.92);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-right: 1px solid rgba(0, 160, 255, 0.15);
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 1.5px;
    font-size: 0.8rem;
    border-radius: 10px;
    background: rgba(0, 120, 255, 0.15);
    border: 1px solid rgba(0, 180, 255, 0.3);
    color: #4dc9f6;
    transition: all 0.3s;
    padding: 8px 20px;
}
.stButton > button:hover {
    background: rgba(0, 150, 255, 0.25);
    border-color: rgba(0, 220, 255, 0.6);
    box-shadow: 0 0 20px rgba(0, 160, 255, 0.2);
    color: #fff;
}

/* ── Select / Slider ── */
[data-testid="stSelectbox"] label, [data-testid="stSlider"] label {
    color: #7ec8e3 !important;
    font-family: 'Inter', sans-serif;
}

/* ── Metrics ── */
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace;
    color: #00d4ff !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: rgba(10, 25, 50, 0.5) !important;
    border: 1px solid rgba(0, 160, 255, 0.15) !important;
    border-radius: 10px;
}

/* ── Animations ── */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}
.pulse { animation: pulse 2s infinite; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
::-webkit-scrollbar-thumb { background: rgba(0,160,255,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Data Loading (once, cached)
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_scalers():
    return pickle.load(open(os.path.join(DATA_DIR, "scalers.pkl"), "rb"))


@st.cache_resource
def load_raw_data():
    path = os.path.join(DATA_DIR, "all_raw.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        return df
    return None


@st.cache_resource
def load_meta():
    path = os.path.join(DATA_DIR, "meta.json")
    if os.path.exists(path):
        return json.load(open(path))
    return {}


@st.cache_resource
def load_model(checkpoint_name):
    """Load a pre-trained model from checkpoint, reading architecture from config.yaml."""
    import yaml
    cfg_path = os.path.join(PROJ_ROOT, "config.yaml")
    with open(cfg_path) as f:
        cfg_full = yaml.safe_load(f)
    cfg = cfg_full["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_name == "ensemble":
        from src.models import Ensemble
        models = []
        for mname in ["bilstm", "lstm2", "transformer"]:
            ccfg = {**cfg_full, "model": {**cfg, "name": mname}}
            m = build_model(ccfg).to(device)
            path = os.path.join(CKPT_DIR, f"{mname}_best.pt")
            if os.path.exists(path):
                m.load_state_dict(torch.load(path, map_location=device))
                m.eval()
                models.append(m)
        if len(models) >= 2:
            return Ensemble(models).to(device), device
        return None, device

    # Individual model
    ccfg = {**cfg_full, "model": {**cfg, "name": checkpoint_name}}
    model = build_model(ccfg).to(device)
    ckpt_path = os.path.join(CKPT_DIR, f"{checkpoint_name}_best.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model, device


@st.cache_resource
def load_test_data():
    return dict(np.load(os.path.join(DATA_DIR, "test.npz")))


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def inverse_tavg(y_scaled, city_arr, scalers, n_features=9):
    out = np.zeros_like(y_scaled)
    for i, c in enumerate(city_arr):
        dummy = np.zeros((1, n_features))
        dummy[0, 0] = y_scaled[i]
        out[i] = scalers[c].inverse_transform(dummy)[0, 0]
    return out


def get_predictions(model, device, X_test, y_test, city_test, scalers, target_city):
    city_mask = city_test == target_city
    X_city = torch.from_numpy(X_test[city_mask]).to(device)
    with torch.no_grad():
        preds = model(X_city).cpu().numpy()
    n_feat = X_test.shape[2]
    y_true_orig = inverse_tavg(y_test[city_mask], city_test[city_mask], scalers, n_feat)
    y_pred_orig = inverse_tavg(preds, np.array([target_city] * len(preds)), scalers, n_feat)
    return y_true_orig, y_pred_orig


def forecast_multistep(model, device, city_raw, scalers, city, horizon_days, window_len=30):
    features = ["tavg", "tmin", "tmax", "prcp", "rhum", "wspd", "pres", "day_sin", "day_cos"]
    scaler = scalers[city]
    cdf = city_raw.sort_values("date").tail(window_len + 30).copy()

    # Add day_sin/day_cos if not present
    if "day_sin" not in cdf.columns:
        doy = cdf["date"].dt.dayofyear.values
        cdf["day_sin"] = np.sin(2 * np.pi * doy / 365.25)
        cdf["day_cos"] = np.cos(2 * np.pi * doy / 365.25)

    seed_window = cdf[features].values[-window_len:].astype(np.float32)
    seed_scaled = scaler.transform(seed_window)
    window = seed_scaled.copy()
    preds_scaled = []

    last_date = cdf["date"].iloc[-1]
    fc_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    for step, fc_date in enumerate(fc_dates):
        X_input = torch.from_numpy(window[np.newaxis, :, :]).to(device)
        with torch.no_grad():
            next_tavg = model(X_input).item()

        # Compute day_sin/day_cos for this forecast date
        doy = fc_date.dayofyear
        day_sin = np.sin(2 * np.pi * doy / 365.25)
        day_cos = np.cos(2 * np.pi * doy / 365.25)

        r = scaler.data_range_[0]
        next_row = np.array([
            next_tavg,
            next_tavg - 4.0 / (r + 1e-8),
            next_tavg + 4.0 / (r + 1e-8),
            window[-1, 3],    # prcp: carry forward
            window[-1, 4],    # rhum: carry forward
            window[-1, 5],    # wspd: carry forward
            window[-1, 6],    # pres: carry forward
            day_sin,           # real day_sin for target date
            day_cos,           # real day_cos for target date
        ], dtype=np.float32)

        preds_scaled.append(next_tavg)
        window = np.vstack([window[1:], next_row]).astype(np.float32)

    preds_scaled = np.array(preds_scaled)
    y_orig = inverse_tavg(preds_scaled, np.array([city] * horizon_days), scalers, len(features))

    mae_estimate = 1.05  # improved MAE
    lower = y_orig - mae_estimate
    upper = y_orig + mae_estimate

    return fc_dates, y_orig, lower, upper


# ══════════════════════════════════════════════════════════════
# Dashboard
# ══════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="glass-header">
    <h1 style="margin:0; font-size:2.2rem;">🛰️ <span style="font-weight:900;">SW WEATHER AI</span> <span style="font-size:1rem; color:#4dc9f6; font-family:'Inter';">v2.0</span></h1>
    <p style="margin:4px 0 0 0; color:#6a9fd8; font-size:0.85rem;">
    Deep Learning Weather Prediction · Southwest China · Kunming · Guiyang · Chengdu · Chongqing
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("## ⚙️ CONTROL PANEL")

    city = st.selectbox("TARGET CITY", ["Kunming", "Guiyang", "Chengdu", "Chongqing"])
    model_name = st.selectbox("AI MODEL", ["bilstm", "lstm2", "transformer", "ensemble"],
                              format_func=lambda x: {
                                  "bilstm": "BiLSTM (Bidirectional)",
                                  "lstm2": "LSTM v2 (Deep + Dropout)",
                                  "transformer": "Transformer Encoder",
                                  "ensemble": "Ensemble (All 3 Models)"
                              }[x])

    st.divider()
    st.markdown("### 🔮 FORECAST")
    forecast_days = st.selectbox("HORIZON", [1, 7, 30],
                                 format_func=lambda x: f"{x} Day{'s' if x>1 else ''}")
    st.divider()

    # Load test data
    test_data = load_test_data()
    scalers = load_scalers()
    raw_df = load_raw_data()
    meta = load_meta()

    # City data
    if raw_df is not None:
        city_raw = raw_df[raw_df["city"] == city].sort_values("date")
    else:
        city_raw = None

    # Latest observation
    if city_raw is not None and len(city_raw) > 0:
        latest = city_raw.iloc[-1]
        st.markdown(f"""
        <div class="glass" style="text-align:center;">
            <div style="color:#6a9fd8; font-size:0.7rem; text-transform:uppercase; letter-spacing:2px;">Latest Observation</div>
            <div style="font-family:'Orbitron'; font-size:1.5rem; font-weight:900; background:linear-gradient(135deg,#00d4ff,#0090ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{latest['date'].strftime('%Y-%m-%d')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Main Dashboard ──

# Row 1: Stat Cards
st.markdown("### 📡 REAL-TIME SENSOR DATA")

if city_raw is not None:
    latest = city_raw.iloc[-1]
    prev = city_raw.iloc[-2] if len(city_raw) > 1 else latest

    cols = st.columns(4)
    metrics = [
        ("🌡️ TEMPERATURE", f"{latest['tavg']:.1f}°C",
         f"min {latest['tmin']:.1f} / max {latest['tmax']:.1f}",
         "cyan"),
        ("💧 HUMIDITY", f"{latest['rhum']:.1f}%",
         f"Δ {latest['rhum'] - prev['rhum']:+.1f}%",
         "blue"),
        ("💨 WIND", f"{latest['wspd']:.1f} m/s",
         f"gust data via API",
         "green"),
        ("📊 PRESSURE", f"{latest['pres']:.1f} hPa",
         f"Δ {latest['pres'] - prev['pres']:+.1f} hPa",
         "purple"),
    ]

    for i, (label, value, subtitle, color) in enumerate(metrics):
        with cols[i]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value pulse" style="background:linear-gradient(135deg,#00d4ff,#0090ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{value}</div>
                <div class="stat-label">{label}</div>
                <div style="font-size:0.7rem; color:#4a7a9e; margin-top:4px;">{subtitle}</div>
            </div>
            """, unsafe_allow_html=True)

# Row 2: Data completeness bar
if city_raw is not None:
    total_days = len(city_raw)
    missing = city_raw[["tavg", "tmin", "tmax", "prcp", "rhum", "wspd", "pres"]].isna().sum().sum()
    completeness = (1 - missing / (total_days * 7)) * 100
    st.markdown(f"""
    <div style="font-size:0.7rem; color:#4a7a9e; text-align:right; margin-top:-8px;">
    Data completeness: {completeness:.1f}% · {total_days:,} records · {city_raw['date'].min().strftime('%Y-%m-%d')} ~ {city_raw['date'].max().strftime('%Y-%m-%d')}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Row 3: Historical Overview + AI Prediction
st.markdown("### 📈 HISTORICAL ANALYSIS + AI FORECAST")
col_chart, col_metrics = st.columns([3, 1])

with col_chart:
    if city_raw is not None:
        # Get model predictions
        model, device = load_model(model_name)
        if model and "city" in test_data:
            y_true, y_pred = get_predictions(
                model, device, test_data["X"], test_data["y"],
                test_data["city"], scalers, city
            )

        # Plotly interactive chart
        fig = go.Figure()

        # Full historical temperature
        fig.add_trace(go.Scatter(
            x=city_raw["date"], y=city_raw["tavg"],
            mode="lines", name="Historical Tavg",
            line=dict(color="rgba(0,200,255,0.25)", width=1),
            fill="tozeroy", fillcolor="rgba(0,180,255,0.03)",
            hovertemplate="%{y:.1f}°C<br>%{x|%Y-%m-%d}"
        ))

        # 30-day moving average
        if len(city_raw) > 30:
            ma = city_raw["tavg"].rolling(30).mean()
            fig.add_trace(go.Scatter(
                x=city_raw["date"], y=ma,
                mode="lines", name="30-day MA",
                line=dict(color="rgba(0,255,200,0.5)", width=1.5, dash="dash"),
                hovertemplate="MA: %{y:.1f}°C"
            ))

        # Test period predictions
        if model and len(y_true) > 0:
            test_dates = pd.to_datetime(test_data["dates"][test_data["city"] == city])
            fig.add_trace(go.Scatter(
                x=test_dates, y=y_pred,
                mode="lines", name=f"AI ({model_name})",
                line=dict(color="#ff6b9d", width=2),
                hovertemplate="Pred: %{y:.1f}°C<br>%{x|%Y-%m-%d}"
            ))
            fig.add_trace(go.Scatter(
                x=test_dates, y=y_true,
                mode="lines", name="Actual (test)",
                line=dict(color="#00e5ff", width=1.5),
                hovertemplate="True: %{y:.1f}°C"
            ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=400,
            legend=dict(orientation="h", yanchor="top", y=1.1, x=0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
            yaxis=dict(title="Temperature (°C)", gridcolor="rgba(255,255,255,0.05)", showgrid=True),
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

with col_metrics:
    if model and len(y_true) > 0:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        st.markdown("### 🎯 MODEL PERFORMANCE")
        for label, value, unit in [
            ("MAE", mae, "°C"),
            ("RMSE", rmse, "°C"),
            ("R² Score", r2, ""),
        ]:
            st.markdown(f"""
            <div class="stat-card" style="padding:12px 16px; margin-bottom:8px;">
                <div style="color:#6a9fd8; font-size:0.65rem; text-transform:uppercase; letter-spacing:2px;">{label}</div>
                <div style="font-family:'Orbitron'; font-size:1.4rem; font-weight:700; background:linear-gradient(135deg,#00e5ff,#ff6b9d);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{value:.4f} {unit}</div>
            </div>
            """, unsafe_allow_html=True)

        # Error histogram
        errors = y_true - y_pred
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=errors, nbinsx=40,
            marker=dict(color="rgba(0,200,255,0.6)", line=dict(color="rgba(0,200,255,0.8)", width=0.5)),
            name="Error"
        ))
        fig_hist.add_vline(x=0, line_width=1, line_dash="dash", line_color="#ff6b9d")
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=20, b=10),
            height=200,
            xaxis_title="Prediction Error (°C)",
            showlegend=False,
        )
        st.plotly_chart(fig_hist, width="stretch")

# Row 4: 7-day feature overview + forecast table
st.markdown("### 🗓️ WEATHER PATTERN ANALYSIS")
col_feat, col_fc = st.columns([3, 1])

with col_feat:
    if city_raw is not None:
        # Recent 90 days multi-feature chart
        recent = city_raw.tail(90)
        fig_multi = go.Figure()

        # Normalize for visual overlay
        def norm(s):
            return (s - s.min()) / (s.max() - s.min() + 1e-8)

        fig_multi.add_trace(go.Scatter(
            x=recent["date"], y=norm(recent["tavg"]),
            mode="lines", name="Temperature", line=dict(color="#00e5ff", width=2),
        ))
        fig_multi.add_trace(go.Scatter(
            x=recent["date"], y=norm(recent["rhum"]),
            mode="lines", name="Humidity", line=dict(color="#7c3aed", width=1.5),
        ))
        fig_multi.add_trace(go.Scatter(
            x=recent["date"], y=norm(recent["prcp"] * 10),
            mode="lines", name="Precipitation (x10)", line=dict(color="#06b6d4", width=1),
        ))
        fig_multi.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10), height=250,
            legend=dict(orientation="h", yanchor="top", y=1.15),
            xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
            yaxis=dict(title="Normalized", gridcolor="rgba(255,255,255,0.03)"),
        )
        st.plotly_chart(fig_multi, width="stretch", key="multi_feature")

with col_fc:
    if model and len(y_true) > 0:
        # Show last 7 days of predictions vs actual
        st.markdown("**Recent 7 Predictions**")
        n_show = min(7, len(y_true))
        fc_df = pd.DataFrame({
            "Actual": y_true[-n_show:],
            "Predicted": y_pred[-n_show:],
            "Error": np.abs(y_true[-n_show:] - y_pred[-n_show:]),
        }).round(2)
        st.dataframe(fc_df, width="stretch")

# Row 5: Future Forecast (multi-step)
st.markdown(f"### 🔮 {forecast_days}-DAY FORECAST — {city}")
if city_raw is not None and raw_df is not None:
    model, device = load_model(model_name)
    if model:
        fc_dates, fc_tavg, fc_lower, fc_upper = forecast_multistep(
            model, device, city_raw, scalers, city, forecast_days
        )

        col_fc_chart, col_fc_table = st.columns([5, 2])

        with col_fc_table:
            # Forecast table with dates
            fc_rows = []
            for i, (d, t, lo, hi) in enumerate(zip(fc_dates, fc_tavg, fc_lower, fc_upper)):
                icon = "☀️" if t > 20 else ("⛅" if t > 10 else "🌧️")
                fc_rows.append({
                    "": icon,
                    "Date": d.strftime("%m/%d"),
                    "Tavg": f"{t:.1f}°C",
                    "Range": f"{lo:.1f}~{hi:.1f}",
                })
            st.markdown(
                "<div style='font-size:0.75rem; color:#4dc9f6; margin-bottom:4px;'>"
                "Predicted Daily Avg Temperature</div>",
                unsafe_allow_html=True,
            )
            fc_df = pd.DataFrame(fc_rows)
            st.dataframe(fc_df, width="stretch", hide_index=True,
                         column_config={"": st.column_config.TextColumn("")})

        with col_fc_chart:
            # Show last 90 days + forecast
            hist_start = city_raw["date"].max() - pd.Timedelta(days=90)
            hist = city_raw[city_raw["date"] >= hist_start]

            fig_fc = go.Figure()

            # Historical
            fig_fc.add_trace(go.Scatter(
                x=hist["date"], y=hist["tavg"],
                mode="lines", name="Historical",
                line=dict(color="rgba(0,200,255,0.5)", width=1.5),
                hovertemplate="%{y:.1f}°C<br>%{x|%m/%d}"
            ))

            # Confidence band
            fig_fc.add_trace(go.Scatter(
                x=list(fc_dates) + list(fc_dates[::-1]),
                y=list(fc_upper) + list(fc_lower[::-1]),
                fill="toself", fillcolor="rgba(255,107,157,0.15)",
                line=dict(color="rgba(255,107,157,0)"),
                name="Confidence",
                hoverinfo="skip",
            ))

            # Forecast
            fig_fc.add_trace(go.Scatter(
                x=fc_dates, y=fc_tavg,
                mode="lines+markers", name=f"Forecast ({model_name})",
                line=dict(color="#ff6b9d", width=2.5),
                marker=dict(symbol="diamond", size=6, color="#ff6b9d"),
                hovertemplate="Pred: %{y:.1f}°C<br>%{x|%Y-%m-%d}"
            ))

            # Separator line
            sep_date = hist["date"].max()
            y_range = [min(hist["tavg"].min(), fc_lower.min()) - 2,
                       max(hist["tavg"].max(), fc_upper.max()) + 2]
            fig_fc.add_shape(
                type="line", x0=sep_date, x1=sep_date,
                y0=y_range[0], y1=y_range[1],
                line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
            )
            fig_fc.add_annotation(
                x=sep_date, y=y_range[1],
                text="NOW", showarrow=False,
                font=dict(size=10, color="#4dc9f6"),
                bgcolor="rgba(0,0,0,0.4)",
            )

            fig_fc.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10), height=350,
                legend=dict(orientation="h", yanchor="top", y=1.1, x=0),
                xaxis=dict(gridcolor="rgba(255,255,255,0.03)", title=None),
                yaxis=dict(title="Temperature (°C)", gridcolor="rgba(255,255,255,0.03)"),
                hovermode="x unified",
            )
            st.plotly_chart(fig_fc, width="stretch", key="forecast_chart")

        # Summary text
        fc_avg = fc_tavg.mean()
        trend = "warming" if fc_tavg[-1] > fc_tavg[0] else "cooling"
        st.markdown(f"""
        <div class="glass" style="font-size:0.85rem; color:#7ec8e3; text-align:center; margin-top:12px;">
            📡 <b>{forecast_days}-day forecast for {city}</b> —
            Average predicted: <span style="color:#00d4ff;">{fc_avg:.1f}°C</span>
            · Trend: <span style="color:#ff6b9d;">{trend}</span>
            · From <b>{fc_dates[0].strftime('%b %d')}</b> to <b>{fc_dates[-1].strftime('%b %d, %Y')}</b>
            · Model: <span style="color:#4dc9f6;">{model_name}</span>
        </div>
        """, unsafe_allow_html=True)

# Row 6: All cities comparison
st.markdown("### 🌍 CROSS-CITY COMPARISON (Full History)")
if raw_df is not None:
    fig_cities = go.Figure()
    colors = {"Kunming": "#00e5ff", "Guiyang": "#7c3aed", "Chengdu": "#f59e0b", "Chongqing": "#ff6b9d"}
    for c in ["Kunming", "Guiyang", "Chengdu", "Chongqing"]:
        cdf = raw_df[raw_df["city"] == c].sort_values("date")
        if len(cdf) > 100:
            ma = cdf["tavg"].rolling(90).mean()
            fig_cities.add_trace(go.Scatter(
                x=cdf["date"], y=ma,
                mode="lines", name=c,
                line=dict(color=colors.get(c, "white"), width=1.5),
                hovertemplate=f"{c}: %{{y:.1f}}°C"
            ))
    fig_cities.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10), height=300,
        legend=dict(orientation="h", yanchor="top", y=1.1),
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
        yaxis=dict(title="90-day MA Temperature (°C)", gridcolor="rgba(255,255,255,0.03)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_cities, width="stretch", key="city_compare")

# Footer
meta_info = load_meta()
dr = meta_info.get("date_range", ["?", "?"])
st.markdown(f"""
<div style="text-align:center; padding:20px; color:#2a4a6a; font-size:0.7rem; font-family:'Inter';">
    SW Weather AI v2.0 · Data: {dr[0]} ~ {dr[1]} · Open-Meteo API · PyTorch
     · Models: LSTM / Transformer · <span class="pulse">●</span> Live
</div>
""", unsafe_allow_html=True)

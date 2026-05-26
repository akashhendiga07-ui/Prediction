# ============================================================
# AI-BASED PREDICTIVE MAINTENANCE SYSTEM
# STREAMLIT DEPLOYMENT UI
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

with open("best_random_forest_model.pkl", "rb") as f:
    best_rf = pickle.load(f)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.big-title {
    font-size: 42px;
    font-weight: bold;
    color: #1f3c88;
    text-align: center;
}

.sub-title {
    font-size: 18px;
    color: #555;
    text-align: center;
}

.low-risk {
    background-color: #d4edda;
    padding: 20px;
    border-radius: 12px;
    color: #155724;
    font-size: 24px;
    font-weight: bold;
}

.medium-risk {
    background-color: #fff3cd;
    padding: 20px;
    border-radius: 12px;
    color: #856404;
    font-size: 24px;
    font-weight: bold;
}

.high-risk {
    background-color: #f8d7da;
    padding: 20px;
    border-radius: 12px;
    color: #721c24;
    font-size: 24px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="big-title">AI-Based Predictive Maintenance System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Server Failure Prediction using Machine Learning</div>',
    unsafe_allow_html=True
)

st.write("---")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("System Information")

st.sidebar.info("""
This system predicts:
- Server Failure Probability
- Infrastructure Risk Level
- Operational Health Status

Best Model:
Optimized Random Forest
""")

# ============================================================
# INPUT SECTION
# ============================================================

st.header("Enter Server Telemetry Metrics")

col1, col2 = st.columns(2)

# ============================================================
# COLUMN 1
# ============================================================

with col1:

    cpu = st.slider(
        "CPU Usage (%)",
        0,
        150,
        50
    )

    memory = st.slider(
        "Memory Usage (%)",
        0,
        150,
        60
    )

    disk_io = st.slider(
        "Disk IO (MB/s)",
        0,
        500,
        150
    )

# ============================================================
# COLUMN 2
# ============================================================

with col2:

    latency = st.slider(
        "Network Latency (ms)",
        0,
        250,
        50
    )

    error_rate = st.slider(
        "Error Rate",
        0.000,
        0.150,
        0.020
    )

    hour = st.slider(
        "Hour",
        0,
        23,
        12
    )

# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

sample_input = pd.DataFrame({

    "CPU_Usage(%)": [cpu],

    "Memory_Usage(%)": [memory],

    "Disk_IO(MB/s)": [disk_io],

    "Network_Latency(ms)": [latency],

    "Error_Rate": [error_rate],

    "Hour": [hour]
})

# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button("Predict System Risk"):

    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = best_rf.predict(sample_input)[0]

    probability = best_rf.predict_proba(
        sample_input
    )[0][1]

    # ========================================================
    # HYBRID RISK ENGINE
    # ========================================================

    if (
        cpu > 90 and
        memory > 90 and
        error_rate > 0.08
    ):

        risk = "High Risk"
        css_class = "high-risk"
        message = "Critical Infrastructure Failure Risk"

    elif probability < 0.35:

        risk = "Low Risk"
        css_class = "low-risk"
        message = "System Operating Normally"

    elif probability < 0.65:

        risk = "Medium Risk"
        css_class = "medium-risk"
        message = "Infrastructure Requires Monitoring"

    else:

        risk = "High Risk"
        css_class = "high-risk"
        message = "Potential Server Failure Detected"

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.write("---")

    st.subheader("Prediction Results")

    st.markdown(
        f'<div class="{css_class}">{message}</div>',
        unsafe_allow_html=True
    )

    st.write("")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            "Prediction",
            int(prediction)
        )

    with metric_col2:
        st.metric(
            "Failure Probability",
            f"{probability:.4f}"
        )

    with metric_col3:
        st.metric(
            "Risk Level",
            risk
        )

    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.subheader("Failure Probability Visualization")

    fig, ax = plt.subplots(figsize=(8,2))

    ax.barh(
        ["Failure Probability"],
        [probability]
    )

    ax.set_xlim(0,1)

    ax.set_xlabel("Probability")

    st.pyplot(fig)

    # ========================================================
    # TELEMETRY SUMMARY
    # ========================================================

    st.subheader("Input Telemetry")

    st.dataframe(sample_input)

# ============================================================
# FOOTER
# ============================================================

st.write("---")

st.caption(
    "AI-Based Predictive Maintenance Framework for Server Failure Prediction"
)
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Iris ANN Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CUSTOM CSS — gives the app a polished, non-default look
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(160deg, #fdf2f8 0%, #f0f9ff 50%, #f5f3ff 100%);
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #a855f7, #ec4899, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .card {
        background: rgba(255, 255, 255, 0.75);
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.10);
        border: 1px solid rgba(255,255,255,0.6);
        margin-bottom: 1rem;
    }
    .result-box {
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    .footer-note {
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 2.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #a855f7, #ec4899);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-weight: 700;
        font-size: 1.05rem;
        width: 100%;
        transition: transform 0.15s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown('<div class="main-title">🌸 Iris ANN Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A neural network trained to classify Iris flowers '
    'into Setosa, Versicolor, or Virginica</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# LOAD MODEL + SCALER (cached so it only loads once)
# ----------------------------------------------------------------------------
CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]
CLASS_COLORS = {"Setosa": "#a855f7", "Versicolor": "#ec4899", "Virginica": "#6366f1"}
CLASS_IMAGES = {
    "Setosa": "🌸",
    "Versicolor": "🌺",
    "Virginica": "🪻",
}

@st.cache_resource
def load_artifacts():
    model = load_model("iris_model.h5")
    try:
        scaler = joblib.load("scaler(1).pkl")
    except FileNotFoundError:
        scaler = None
    return model, scaler

model_load_error = None
try:
    model, scaler = load_artifacts()
except Exception as e:
    model, scaler = None, None
    model_load_error = str(e)

if model_load_error:
    st.error(
        "⚠️ Couldn't load `iris_model.h5` (and optionally `scaler.pkl`). "
        "Make sure both files sit in the same folder as this app.py.\n\n"
        f"Details: {model_load_error}"
    )

# ----------------------------------------------------------------------------
# SIDEBAR — INPUT CONTROLS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌿 Flower Measurements")
    st.caption("Adjust the sliders to describe the flower.")

    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.7, 0.1)
    petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)

    st.markdown("---")
    st.markdown("### 🎲 Quick Presets")
    preset = st.radio(
        "Load a typical example:",
        ["Custom", "Typical Setosa", "Typical Versicolor", "Typical Virginica"],
        index=0,
    )

    presets = {
        "Typical Setosa":     (5.0, 3.4, 1.5, 0.2),
        "Typical Versicolor": (5.9, 2.8, 4.3, 1.3),
        "Typical Virginica":  (6.6, 3.0, 5.6, 2.1),
    }
    if preset in presets:
        sepal_length, sepal_width, petal_length, petal_width = presets[preset]
        st.info(f"Loaded preset: **{preset}**")

    st.markdown("---")
    predict_clicked = st.button("🔮 Predict Species")

# ----------------------------------------------------------------------------
# MAIN LAYOUT
# ----------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📋 Current Input")
    input_df = pd.DataFrame(
        {
            "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
            "Value (cm)": [sepal_length, sepal_width, petal_length, petal_width],
        }
    )
    st.dataframe(input_df, hide_index=True, use_container_width=True)

    # Simple radar/shape visual of the flower's proportions
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[sepal_length, sepal_width, petal_length, petal_width, sepal_length],
        theta=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Sepal Length"],
        fill="toself",
        line=dict(color="#a855f7"),
        fillcolor="rgba(168, 85, 247, 0.25)",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 8])),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔮 Prediction")

    if predict_clicked:
        if model is None:
            st.warning("Model isn't loaded yet — add `iris_model.h5` next to this app.py.")
        else:
            raw_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            model_input = scaler.transform(raw_input) if scaler is not None else raw_input

            probs = model.predict(model_input, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx]) * 100

            st.markdown(
                f"""
                <div class="result-box" style="background: linear-gradient(135deg, {CLASS_COLORS[pred_class]}, #6366f1);">
                    {CLASS_IMAGES[pred_class]} Predicted Species: {pred_class}<br>
                    <span style="font-size:1rem; font-weight:400;">Confidence: {confidence:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Confidence Breakdown")
            fig_bar = go.Figure(go.Bar(
                x=[p * 100 for p in probs],
                y=CLASS_NAMES,
                orientation="h",
                marker_color=[CLASS_COLORS[c] for c in CLASS_NAMES],
                text=[f"{p*100:.1f}%" for p in probs],
                textposition="auto",
            ))
            fig_bar.update_layout(
                xaxis_title="Probability (%)",
                xaxis=dict(range=[0, 100]),
                margin=dict(l=10, r=10, t=10, b=10),
                height=250,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("👈 Set your measurements in the sidebar, then click **Predict Species**.")

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# ABOUT SECTION
# ----------------------------------------------------------------------------
with st.expander("ℹ️ About this project"):
    st.write(
        """
        This app runs a simple Artificial Neural Network (ANN) trained on the
        classic **Iris dataset** (150 samples, 4 features, 3 species).

        **Pipeline:** measurements → scaler (if used during training) → ANN → softmax probabilities → predicted class.

        Built with **TensorFlow/Keras**, **Streamlit**, and **Plotly**.
        """
    )

st.markdown('<div class="footer-note">Iris_ANN_Classifier · Built with Streamlit 🌱</div>', unsafe_allow_html=True)
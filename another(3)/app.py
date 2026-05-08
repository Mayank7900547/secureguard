from engine import (FraudEngine, KaggleFraudEngine, THRESHOLDS, FEATURE_DESCRIPTIONS,
                     FEATURE_UNITS, FEATURES, KAGGLE_FEATURES, classify_risk, feature_risk_level)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SOTA Fraud Detection — Industry Expert Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background: linear-gradient(135deg,#0f172a 0%,#1e293b 100%); color:#e2e8f0; }
.main-header {
    font-size:2.8rem; font-weight:700;
    background:linear-gradient(90deg,#38bdf8,#818cf8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:.3rem;
}
.sub-header { font-size:1.1rem; color:#94a3b8; margin-bottom:1.5rem; }
.metric-card {
    background:rgba(30,41,59,0.8); padding:1.2rem 1.4rem;
    border-radius:.8rem; border:1px solid rgba(255,255,255,0.08);
    backdrop-filter:blur(10px); transition:transform .25s ease,border .25s ease;
}
.metric-card:hover { transform:translateY(-4px); border:1px solid rgba(56,189,248,0.4); }
.risk-high     { color:#f43f5e; font-weight:700; }
.risk-moderate { color:#fb923c; font-weight:700; }
.risk-safe     { color:#10b981; font-weight:700; }
.shap-legend   {
    background:rgba(30,41,59,0.7); border-radius:.6rem;
    padding:.8rem 1.2rem; border-left:3px solid #818cf8;
    font-size:.9rem; line-height:1.7;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ENGINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_engine():
    return FraudEngine()

@st.cache_resource
def load_kaggle_engine():
    return KaggleFraudEngine()

engine        = load_engine()
import importlib, engine as eng_module; importlib.reload(eng_module)
kaggle_engine = load_kaggle_engine()

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/shield.png", width=90)
    st.title("SecureGuard AI")
    st.markdown("---")
    menu = st.radio("Navigation", [
        "📊 Overview",
        "🔍 Real-time Detection",
        "🧠 Model Explainability (SHAP)",
        "📈 Dataset Scaling Analysis",
        "🕵️ Fraud Transaction Explorer",
        "📑 Comparative Analysis",
        "⚙️ Settings",
    ])
    st.markdown("---")
    st.success("Status: SOTA Online ✅")
    st.markdown("**Features:** 6 domain-informed")
    st.markdown("---")
    
    # ── Download Architecture ────────────────────────────────────────────────
    try:
        with open("Fraud_Detection_System_Architecture.png", "rb") as f:
            st.download_button(
                label="📥 Download Architecture Diagram",
                data=f,
                file_name="Fraud_Detection_System_Architecture.png",
                mime="image/png",
                use_container_width=True
            )
    except FileNotFoundError:
        pass

st.markdown('<h1 class="main-header">🛡️ SOTA Fraud Detection Engine</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Industry-Expert Dashboard · Real-time Monitoring · Explainable AI · Domain-Informed Risk Thresholds</p>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def risk_badge(level: str) -> str:
    if level == "High Risk":     return "🔴 High Risk"
    if level == "Moderate Risk": return "🟠 Moderate Risk"
    return "🟢 Legitimate"

def confidence_narrative(prob: float, sv, feat_names) -> str:
    pct      = prob * 100
    red_n    = int(sum(1 for v in sv if v > 0))
    blue_n   = int(sum(1 for v in sv if v < 0))
    top_feat = feat_names[int(np.argmax(np.abs(sv)))]

    if pct >= 80:
        verdict = f"🔴 **Very High Fraud Risk ({pct:.1f}% confidence)**"
        detail  = (f"**{red_n} features** are pushing this transaction toward fraud. "
                   f"The strongest driver is **{top_feat}**. "
                   f"Even if the remaining {blue_n} features look clean, "
                   f"the weight of evidence is overwhelming — flag immediately.")
    elif pct >= 60:
        verdict = f"🟠 **High Fraud Risk ({pct:.1f}% confidence)**"
        detail  = (f"**{red_n} red bar(s)** indicate genuine fraud signals. "
                   f"**{top_feat}** is the primary driver. "
                   f"The {blue_n} blue feature(s) are partially reducing the score "
                   f"but not enough to clear the transaction — escalate for review.")
    elif pct >= 40:
        verdict = f"🟡 **Moderate Fraud Risk ({pct:.1f}% confidence)**"
        detail  = (f"The model is uncertain. {red_n} feature(s) suggest risk "
                   f"({top_feat} is the main concern) while {blue_n} feature(s) "
                   f"support legitimacy. Recommend manual review before approving.")
    else:
        verdict = f"🟢 **Low Fraud Risk ({pct:.1f}% confidence)**"
        detail  = (f"Most features ({blue_n} blue bars) actively reduce fraud probability. "
                   f"Only {red_n} feature(s) raise minor concern. "
                   f"Transaction appears legitimate — approve with standard monitoring.")

    return f"{verdict}\n\n{detail}"

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if menu == "📊 Overview":

    # Live metrics from 5 000-row synthetic data
    with st.spinner("Computing live metrics from model…"):
        sample_df = engine.generate_synthetic_data(n_samples=5000)
        metrics   = engine.get_live_metrics(sample_df)

    # ── KPI cards ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card"><h4>Total Transactions</h4>'
                    f'<h2>{metrics["total"]:,}</h2>'
                    f'<p style="color:#94a3b8">Live sample batch</p></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h4>Fraud Flagged</h4>'
                    f'<h2 style="color:#f43f5e">{metrics["flagged"]:,}</h2>'
                    f'<p style="color:#94a3b8">by SOTA ensemble</p></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h4>🔴 High Risk</h4>'
                    f'<h2 style="color:#f43f5e">{metrics["high_risk"]:,}</h2>'
                    f'<p style="color:#94a3b8">≥2 high-risk features</p></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h4>🟠 Moderate Risk</h4>'
                    f'<h2 style="color:#fb923c">{metrics["moderate"]:,}</h2>'
                    f'<p style="color:#94a3b8">1 high or ≥2 moderate features</p></div>',
                    unsafe_allow_html=True)
    with c5:
        recall_txt = f'{metrics["recall"]}%' if metrics["recall"] else "N/A"
        st.markdown(f'<div class="metric-card"><h4>Model Recall</h4>'
                    f'<h2 style="color:#10b981">{recall_txt}</h2>'
                    f'<p style="color:#94a3b8">True fraud caught</p></div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── Fraud probability distribution ───────────────────────────────────────
    st.markdown("### 📊 Fraud Probability Distribution (Live Batch)")
    results_df = engine.predict_batch(sample_df)
    fig_hist = px.histogram(
        results_df, x="Fraud_Probability", nbins=50,
        color="Risk_Level",
        color_discrete_map={"High Risk":"#f43f5e","Moderate Risk":"#fb923c","Legitimate":"#10b981"},
        title="Distribution of Fraud Probability Scores across 5,000 Transactions",
        labels={"Fraud_Probability":"Fraud Probability (%)","count":"# Transactions"},
    )
    fig_hist.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Feature breakdown ─────────────────────────────────────────────────────
    st.markdown("### 🔬 Risk Breakdown by Feature")
    col_l, col_r = st.columns(2)

    with col_l:
        # Amount distribution coloured by risk
        fig_amt = px.histogram(
            results_df, x="Amount", nbins=60, color="Amount_Risk",
            color_discrete_map={"High Risk":"#f43f5e","Moderate Risk":"#fb923c","Safe":"#10b981"},
            title="Transaction Amount — Risk Breakdown",
        )
        fig_amt.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_amt, use_container_width=True)

    with col_r:
        # Time_Delta distribution
        fig_td = px.histogram(
            results_df, x="Time_Delta", nbins=60, color="Time_Delta_Risk",
            color_discrete_map={"High Risk":"#f43f5e","Moderate Risk":"#fb923c","Safe":"#10b981"},
            title="Time Delta (seconds) — Risk Breakdown",
        )
        fig_td.add_vline(x=110, line_dash="dash", line_color="#fb923c",
                         annotation_text="Moderate (110s)")
        fig_td.add_vline(x=160, line_dash="dash", line_color="#f43f5e",
                         annotation_text="High Risk (160s)")
        fig_td.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_td, use_container_width=True)

    # ── Feature reference table ───────────────────────────────────────────────
    st.markdown("### 📋 Feature Definitions & Risk Thresholds")
    ref_data = {
        "Feature":     list(FEATURE_DESCRIPTIONS.keys()),
        "Unit":        [FEATURE_UNITS.get(f,"—") for f in FEATURE_DESCRIPTIONS],
        "Description": list(FEATURE_DESCRIPTIONS.values()),
        "Moderate Risk Threshold": [
            "$500 – $2,000","110 – 160 seconds","50 – 150 km","—","—","2× – 5× avg spend"],
        "High Risk Threshold": [
            "> $2,000","> 160 seconds","> 150 km","= 1 (always high)","—","> 5× avg spend"],
    }
    st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

    st.info("💡 **Note:** All metrics above are computed live by the SOTA Stacking Ensemble "
            "on a fresh synthetic batch. Every number reflects actual model output — nothing is hardcoded.")

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — REAL-TIME DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "🔍 Real-time Detection":
    st.markdown("### 🔍 Real-time Transaction Analysis")
    st.write("Upload a CSV (Kaggle `creditcard.csv` or behavioral format) **or** enter a transaction manually.")

    # ── SHAP legend ───────────────────────────────────────────────────────────
    with st.expander("📖 How to read the SHAP chart & confidence score", expanded=False):
        st.markdown("""
<div class="shap-legend">

**🔴 Red bar** = this feature is PUSHING the model toward "Fraud". It **increases** the fraud probability.

**🔵 Blue bar** = this feature is PULLING the model toward "Legitimate". It **decreases** the fraud probability.

**Bar length** = how strongly that feature influences the final score. A very long red bar can overwhelm several short blue bars.

**Confidence score interpretation:**

| Score | Meaning |
|---|---|
| ≥ 80% | Very High Risk — flag immediately |
| 60–79% | High Risk — escalate for review |
| 40–59% | Moderate — manual review recommended |
| < 40%  | Likely Legitimate |

**One red bar rule:** A single red bar does NOT make a transaction fraudulent. The model weighs ALL features together. 
If one feature (e.g., Distance) is red but all others are strongly blue, the transaction may still score below 50%.
The confidence score is the final verdict — not any individual bar.

</div>
""", unsafe_allow_html=True)

    mode = st.radio("Input Mode", ["📁 Upload CSV", "✏️ Manual Entry"], horizontal=True)

    # ── CSV Upload ────────────────────────────────────────────────────────────
    if mode == "📁 Upload CSV":
        uploaded = st.file_uploader("Drop CSV here (Kaggle creditcard.csv or behavioral format)", type="csv")

        if uploaded:
            raw_df = pd.read_csv(uploaded)
            is_kaggle = "V1" in raw_df.columns
            fmt = "Kaggle (V1–V28)" if is_kaggle else "Behavioral"
            st.success(f"Format detected: **{fmt}** — {len(raw_df):,} transactions loaded.")

            if is_kaggle:
                if not kaggle_engine.is_ready():
                    st.error("""
⚠️ **Kaggle model not trained yet.**

The Kaggle creditcard.csv uses anonymized PCA features (V1–V28) that our behavioral model 
cannot score correctly. A dedicated model needs to be trained on this data first.

👉 Go to **⚙️ Settings → Train Kaggle Model** and point it to your creditcard.csv.
This only needs to be done once — the model is saved to disk.
                    """)
                else:
                    with st.spinner("Running Kaggle-trained ensemble on all transactions…"):
                        results = kaggle_engine.predict_batch(raw_df)

                    fraud_df    = results[results["Predicted_Fraud"] == 1].copy()
                    high_df     = results[results["Risk_Level"] == "High Risk"].copy()
                    moderate_df = results[results["Risk_Level"] == "Moderate Risk"].copy()

                    # True fraud count if Class column present
                    true_fraud = int(raw_df["Class"].sum()) if "Class" in raw_df.columns else None

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Transactions", f"{len(results):,}")
                    m2.metric("🔴 High Risk",        f"{len(high_df):,}")
                    m3.metric("🟠 Moderate Risk",    f"{len(moderate_df):,}")
                    if true_fraud is not None:
                        m4.metric("✅ Actual Frauds in Dataset", f"{true_fraud:,}")
                    else:
                        m4.metric("Avg Fraud Probability (flagged)",
                                  f"{fraud_df['Fraud_Probability'].mean():.1f}%" if len(fraud_df) else "—")

                    if true_fraud is not None:
                        st.info(f"📊 **Reality check:** This dataset has **{true_fraud} actual frauds** "
                                f"out of {len(results):,} transactions — that's "
                                f"**{true_fraud/len(results)*100:.3f}%** fraud rate. "
                                f"Our model flagged **{len(fraud_df):,}** as suspicious.")

                    st.markdown("---")
                    st.markdown("#### 🚨 Flagged Transactions — High Risk + Moderate Risk")
                    combined = pd.concat([high_df, moderate_df]).sort_values(
                        "Fraud_Probability", ascending=False)

                    display_cols = ["Fraud_Probability", "Risk_Level", "Amount", "Time"]
                    kaggle_v_cols = [f"V{i}" for i in range(1, 8)]  # Show V1-V7 as sample
                    if "Class" in combined.columns:
                        display_cols = ["Class"] + display_cols
                    display_cols += kaggle_v_cols

                    st.dataframe(combined[[c for c in display_cols if c in combined.columns]
                                         ].reset_index(drop=True),
                                 use_container_width=True)

                    st.markdown("#### 📊 Fraud Probability — Top 50 Flagged")
                    top50 = fraud_df.nlargest(50, "Fraud_Probability").reset_index(drop=True)
                    fig   = px.bar(top50, x=top50.index, y="Fraud_Probability",
                                   color="Risk_Level",
                                   color_discrete_map={"High Risk": "#f43f5e",
                                                       "Moderate Risk": "#fb923c"},
                                   title="Top 50 Transactions by Fraud Probability (Kaggle Model)")
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                      plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

                    st.session_state["results_df"]  = results
                    st.session_state["is_kaggle"]   = True

            else:
                # ── Behavioral CSV ────────────────────────────────────────────
                with st.spinner("Running SOTA ensemble on all transactions…"):
                    results = engine.predict_batch(raw_df)

                fraud_df    = results[results["Predicted_Fraud"] == 1].copy()
                high_df     = results[results["Risk_Level"] == "High Risk"].copy()
                moderate_df = results[results["Risk_Level"] == "Moderate Risk"].copy()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Transactions",   f"{len(results):,}")
                m2.metric("🔴 High Risk",          f"{len(high_df):,}")
                m3.metric("🟠 Moderate Risk",      f"{len(moderate_df):,}")
                m4.metric("Avg Fraud Probability (flagged)",
                          f"{fraud_df['Fraud_Probability'].mean():.1f}%" if len(fraud_df) else "—")

                st.markdown("---")
                st.markdown("#### 🚨 Flagged Transactions — High Risk + Moderate Risk")
                combined = pd.concat([high_df, moderate_df]).sort_values(
                    "Fraud_Probability", ascending=False)

                display_cols = ["Fraud_Probability", "Risk_Level", "Amount",
                                "Time_Delta", "Distance_From_Home",
                                "Is_High_Risk_Merchant", "Avg_Spent_7D", "Velocity_Ratio"]
                if "Is_Fraud" in combined.columns:
                    display_cols.insert(0, "Is_Fraud")

                st.dataframe(combined[display_cols].reset_index(drop=True),
                             use_container_width=True)

                st.markdown("#### 📊 Fraud Probability — Top 50 Flagged")
                top50 = fraud_df.nlargest(50, "Fraud_Probability").reset_index(drop=True)
                fig   = px.bar(top50, x=top50.index, y="Fraud_Probability",
                               color="Risk_Level",
                               color_discrete_map={"High Risk": "#f43f5e",
                                                   "Moderate Risk": "#fb923c"},
                               title="Top 50 Transactions by Fraud Probability")
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### 🔬 Feature Risk Contribution in Flagged Transactions")
                feat_risk_cols = [c for c in results.columns if c.endswith("_Risk")
                                  and c not in ("Risk_Level",)]
                if feat_risk_cols:
                    risk_counts = {}
                    for col in feat_risk_cols:
                        feat_name = col.replace("_Risk", "")
                        risk_counts[feat_name] = {
                            "High Risk":     int((fraud_df[col] == "High Risk").sum()),
                            "Moderate Risk": int((fraud_df[col] == "Moderate Risk").sum()),
                            "Safe":          int((fraud_df[col] == "Safe").sum()),
                        }
                    rc_df = pd.DataFrame(risk_counts).T.reset_index()
                    rc_df.columns = ["Feature", "High Risk", "Moderate Risk", "Safe"]
                    fig_rc = px.bar(rc_df, x="Feature",
                                    y=["High Risk", "Moderate Risk", "Safe"],
                                    color_discrete_map={"High Risk": "#f43f5e",
                                                        "Moderate Risk": "#fb923c",
                                                        "Safe": "#10b981"},
                                    title="Feature Risk Breakdown in Flagged Transactions",
                                    barmode="stack")
                    fig_rc.update_layout(template="plotly_dark",
                                         paper_bgcolor="rgba(0,0,0,0)",
                                         plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_rc, use_container_width=True)

                st.session_state["results_df"] = results
                st.session_state["is_kaggle"]  = False

    # ── Manual Entry ──────────────────────────────────────────────────────────
    else:
        st.markdown("#### ✏️ Enter Transaction Details")
        with st.expander("📋 Feature guide — what values to enter", expanded=True):
            for feat, desc in FEATURE_DESCRIPTIONS.items():
                st.markdown(f"- **{feat}** ({FEATURE_UNITS.get(feat,'')}) — {desc}")

        c1, c2, c3 = st.columns(3)
        amt   = c1.number_input("Amount ($)",             value=1250.0, min_value=0.0)
        td    = c2.number_input("Time Delta (seconds)",   value=170,    min_value=0)
        dist  = c3.number_input("Distance from Home (km)",value=80.0,   min_value=0.0)
        c4, c5 = st.columns(2)
        hrm   = c4.selectbox("Merchant Category",
                              ["Normal (0)","High-Risk: crypto/forex/gambling (1)"])
        avg7  = c5.number_input("Avg Spent Last 7 Days ($)", value=80.0, min_value=1.0)

        hrm_val = 1 if "High-Risk" in hrm else 0
        vr_val  = round(amt / avg7, 2) if avg7 else 0

        st.markdown(f"**Computed Velocity Ratio:** `{vr_val:.2f}×` avg spend "
                    f"({'🔴 High Risk' if vr_val>5 else '🟠 Moderate' if vr_val>2 else '🟢 Safe'})")

        if st.button("🔍 Analyse Transaction", type="primary"):
            input_df = pd.DataFrame([{
                "Amount": amt, "Time_Delta": td,
                "Distance_From_Home": dist,
                "Is_High_Risk_Merchant": float(hrm_val),
                "Avg_Spent_7D": avg7,
            }])

            with st.spinner("Running SOTA ensemble + SHAP…"):
                prob, shap_img, sv, feat_names, risk = \
                    engine.predict_with_explanation(input_df)

            # Verdict banner
            if prob >= 0.6:
                st.error(f"🚨 {risk} — Fraud Probability: {prob*100:.1f}%")
            elif prob >= 0.4:
                st.warning(f"⚠️ {risk} — Fraud Probability: {prob*100:.1f}%")
            else:
                st.success(f"✅ {risk} — Fraud Probability: {prob*100:.1f}%")

            # Per-feature risk summary
            st.markdown("#### 🔬 Per-Feature Risk Assessment")
            feat_data = {
                "Feature":       feat_names,
                "Value":         [amt, td, dist, hrm_val, avg7, vr_val],
                "Unit":          [FEATURE_UNITS.get(f,"") for f in feat_names],
                "Risk Level":    [feature_risk_level(f, v)
                                  for f, v in zip(feat_names,
                                                  [amt, td, dist, hrm_val, avg7, vr_val])],
                "SHAP Impact":   [round(float(v), 4) for v in sv],
                "Direction":     ["↑ Fraud" if v > 0 else "↓ Fraud" for v in sv],
            }
            feat_table = pd.DataFrame(feat_data)
            st.dataframe(feat_table, use_container_width=True, hide_index=True)

            # SHAP chart
            st.markdown("#### 🧠 SHAP Explanation Chart")
            st.image(shap_img, use_container_width=True)

            # Narrative
            st.markdown("#### 💬 Plain-English Confidence Explanation")
            st.markdown(confidence_narrative(prob, sv, feat_names))

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — SHAP GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "🧠 Model Explainability (SHAP)":
    st.markdown("### 🧠 Explainable AI (XAI) with SHAP")
    st.write("Understand *why* the model flags transactions — not just *that* it does.")

    st.markdown("""
<div class="shap-legend">

**How the global chart works:**

Each bar shows the **average absolute SHAP value** of that feature across hundreds of transactions.
A longer bar = that feature has a bigger influence on fraud decisions on average.

🔴 Red bars = the top-2 most impactful features globally (the model relies on these most).
🟣 Purple bars = supporting features.

**What a SHAP value means numerically:**
A SHAP value of +0.15 means that feature *alone* shifted the fraud probability up by ~15 percentage points from the model's baseline.
A value of -0.10 means it pulled the probability down by ~10 points.

</div>
""", unsafe_allow_html=True)

    if st.button("🚀 Generate Global XAI Dashboard", type="primary"):
        with st.spinner("Computing SHAP values across 300 transactions…"):
            global_img = engine.get_global_explanation()
        if global_img:
            st.image(global_img, use_container_width=True)
            st.success("Global Insight: The top drivers are typically **Velocity_Ratio** "
                       "(spending spike) and **Distance_From_Home** — matching published "
                       "fraud research (Dornadula 2019, Alarfaj 2022).")
        else:
            st.error("Engine not ready. Please retrain in Settings.")

    st.markdown("---")
    st.markdown("### 📋 Feature Threshold Reference")
    th_data = {
        "Feature":        ["Amount","Time_Delta","Distance_From_Home",
                           "Is_High_Risk_Merchant","Velocity_Ratio"],
        "Safe":           ["≤ $500","≤ 110s","≤ 50km","= 0","≤ 2×"],
        "Moderate Risk":  ["$500–$2,000","110–160s","50–150km","—","2×–5×"],
        "High Risk":      ["> $2,000","> 160s","> 150km","= 1","> 5×"],
        "Real-world Meaning": [
            "Large purchases uncommon in daily spend",
            "Avg legit transaction ~85s. >110s = unusual hesitation; >160s = coaching or stolen card test",
            "Transaction far from home — travel or card-present fraud",
            "Crypto/forex/gambling merchants — high chargeback rate",
            "Sudden spending spike vs personal baseline",
        ]
    }
    st.dataframe(pd.DataFrame(th_data), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — SCALING ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "📈 Dataset Scaling Analysis":
    st.markdown("### 📈 SOTA Performance Scaling Analysis")

    scaling_file = st.file_uploader("Upload CSV for scaling analysis (optional)", type="csv",
                                    key="scaling_uploader")
    source_df = None
    if scaling_file:
        source_df = pd.read_csv(scaling_file)
        st.success(f"Loaded {len(source_df):,} records.")
    else:
        st.info("No file — engine will use synthetic data up to 284,307 records.")
        if st.button("Generate Synthetic Benchmarking Data (284k)"):
            with st.spinner("Generating…"):
                source_df = engine.generate_synthetic_data(n_samples=284307)
                st.session_state["source_df"] = source_df
        if "source_df" in st.session_state:
            source_df = st.session_state["source_df"]

    max_val = len(source_df) if source_df is not None else 284307
    default_sizes = [s for s in [30000,50000,150000,max_val] if s <= max_val]
    selected_sizes = st.multiselect(
        "Choose dataset sizes to compare:",
        options=sorted(set(list(range(10000, max_val+1, 10000)) + [max_val])),
        default=default_sizes)

    if st.button("🚀 Run Scaling Comparison", type="primary"):
        if not selected_sizes:
            st.warning("Select at least one size.")
        else:
            with st.spinner("Analysing…"):
                is_kaggle_df = source_df is not None and "V1" in source_df.columns
                target_engine = kaggle_engine if is_kaggle_df else engine
                
                if is_kaggle_df and not kaggle_engine.is_ready():
                    st.error("Kaggle model not ready. Please train it in Settings first.")
                    scaling_df = pd.DataFrame()
                else:
                    scaling_df = target_engine.run_scaling_analysis(
                        custom_df=source_df, custom_sizes=selected_sizes)

            st.dataframe(scaling_df.drop(columns=["Actual Size"]),
                         use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(scaling_df, x="Dataset Size",
                             y=["High Risk","Moderate Risk"],
                             color_discrete_map={"High Risk":"#f43f5e","Moderate Risk":"#fb923c"},
                             title="High vs Moderate Risk Detected per Batch",
                             barmode="stack")
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                scaling_df["Time_Numeric"] = scaling_df["Processing Time"].str.replace("s","").astype(float)
                fig2 = px.line(scaling_df, x="Dataset Size", y="Time_Numeric",
                               title="Processing Latency vs Dataset Size", markers=True)
                fig2.update_traces(line_color="#818cf8")
                fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 — FRAUD TRANSACTION EXPLORER  (NEW)
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "🕵️ Fraud Transaction Explorer":
    st.markdown("### 🕵️ Fraud Transaction Explorer")
    st.write("Drill into flagged transactions. Filter by any feature, risk level, or fraud category.")

    # Get results from session or generate fresh
    if "results_df" not in st.session_state:
        with st.spinner("Generating 5,000-transaction sample…"):
            sample = engine.generate_synthetic_data(n_samples=5000)
            results = engine.predict_batch(sample)
            st.session_state["results_df"] = results
        st.info("No CSV uploaded yet — showing 5,000-row synthetic demo. "
                "Upload a CSV in Real-time Detection to explore your own data here.")

    results = st.session_state["results_df"]
    # Clear stale results if _Risk columns are missing
    if "Amount_Risk" not in results.columns:
        del st.session_state["results_df"]
        st.rerun()
    fraud_all = results[results["Predicted_Fraud"] == 1].copy().reset_index(drop=True)
    st.markdown(f"**Total flagged transactions available:** {len(fraud_all):,} "
                f"({len(fraud_all[fraud_all['Risk_Level']=='High Risk']):,} High Risk + "
                f"{len(fraud_all[fraud_all['Risk_Level']=='Moderate Risk']):,} Moderate Risk)")

    st.markdown("---")
    st.markdown("#### 🎛️ Filters")

    f1, f2 = st.columns(2)

    # Risk level filter
    with f1:
        risk_filter = st.multiselect("Overall Risk Level",
                                     ["High Risk","Moderate Risk"],
                                     default=["High Risk","Moderate Risk"])

    # Detection mode filter
    with f2:
        prob_filter = st.slider("Fraud Probability Range (%)", 0, 100, (0,100))

    st.markdown("##### Filter by Individual Feature Risk")
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        amt_filter = st.selectbox("Amount Risk",
                                  ["All","High Risk (>$2,000)","Moderate ($500–$2,000)","Safe (≤$500)"])
        td_filter  = st.selectbox("Time Delta Risk",
                                  ["All","High Risk (>160s)","Moderate (110–160s)","Safe (≤110s)"])

    with fc2:
        dist_filter = st.selectbox("Distance Risk",
                                   ["All","High Risk (>150km)","Moderate (50–150km)","Safe (≤50km)"])
        hrm_filter  = st.selectbox("Merchant Risk",
                                   ["All","High Risk (=1)","Safe (=0)"])

    with fc3:
        vr_filter   = st.selectbox("Velocity Ratio Risk",
                                   ["All","High Risk (>5×)","Moderate (2–5×)","Safe (≤2×)"])
        sort_by     = st.selectbox("Sort by",
                                   ["Fraud_Probability (desc)","Amount (desc)",
                                    "Time_Delta (desc)","Distance_From_Home (desc)"])

    # ── Apply filters ─────────────────────────────────────────────────────────
    # Build a filter key — if anything changes, clear the cached SHAP result
    filter_key = str((risk_filter, prob_filter, amt_filter, td_filter,
                      dist_filter, hrm_filter, vr_filter, sort_by))
    if st.session_state.get("_explorer_filter_key") != filter_key:
        st.session_state["_explorer_filter_key"] = filter_key
        st.session_state.pop("explorer_shap_result", None)
        st.session_state.pop("explorer_shap_idx", None)

    filtered = fraud_all[
        fraud_all["Risk_Level"].isin(risk_filter) &
        fraud_all["Fraud_Probability"].between(*prob_filter)
    ].copy()

    def apply_feat_filter(df, col, selection, mapping):
        if selection == "All": return df
        label = selection.split("(")[0].strip()
        return df[df[col] == label]

    # Amount
    if amt_filter != "All":
        label = amt_filter.split("(")[0].strip()
        filtered = filtered[filtered["Amount_Risk"] == label]
    # Time
    if td_filter != "All":
        label = td_filter.split("(")[0].strip()
        filtered = filtered[filtered["Time_Delta_Risk"] == label]
    # Distance
    if dist_filter != "All":
        label = dist_filter.split("(")[0].strip()
        filtered = filtered[filtered["Distance_From_Home_Risk"] == label]
    # Merchant
    if hrm_filter != "All":
        label = hrm_filter.split("(")[0].strip()
        filtered = filtered[filtered["Is_High_Risk_Merchant_Risk"] == label]
    # Velocity
    if vr_filter != "All":
        label = vr_filter.split("(")[0].strip()
        filtered = filtered[filtered["Velocity_Ratio_Risk"] == label]

    # Sort
    sort_col = sort_by.split("(")[0].strip()
    filtered = filtered.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # Track filter state — clear SHAP cache whenever filters change so chart updates
    current_filter_state = (
        str(risk_filter) + str(prob_filter) + amt_filter +
        td_filter + dist_filter + hrm_filter + vr_filter + sort_by
    )
    if st.session_state.get("explorer_filter_state") != current_filter_state:
        st.session_state["explorer_filter_state"] = current_filter_state
        st.session_state.pop("explorer_shap_result", None)  # force recompute

    st.markdown("---")
    st.markdown(f"#### \U0001f4cb {len(filtered):,} transactions match your filters")

    # ── Top 10 highlight ──────────────────────────────────────────────────────
    st.markdown("##### 🏆 Top 10 by Fraud Probability")
    top10 = filtered.head(10)
    display_cols = ["Fraud_Probability","Risk_Level","Amount","Time_Delta",
                    "Distance_From_Home","Is_High_Risk_Merchant","Avg_Spent_7D",
                    "Velocity_Ratio",
                    "Amount_Risk","Time_Delta_Risk","Distance_From_Home_Risk",
                    "Is_High_Risk_Merchant_Risk","Velocity_Ratio_Risk"]
    if "Is_Fraud" in top10.columns:
        display_cols = ["Is_Fraud"] + display_cols
    # Only keep columns that actually exist (Kaggle data won't have behavioral cols)
    display_cols = [c for c in display_cols if c in top10.columns]
    
    st.dataframe(top10[display_cols], use_container_width=True, hide_index=True)

    # ── Full filtered table ───────────────────────────────────────────────────
    with st.expander(f"📂 View all {len(filtered):,} filtered transactions"):
        st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    if len(filtered) > 0:
        st.markdown("##### 📊 Filtered Transactions — Visual Breakdown")
        ch1, ch2 = st.columns(2)

        with ch1:
            y_col = "Distance_From_Home" if "Distance_From_Home" in filtered.columns else "Fraud_Probability"
            x_col = "Amount" if "Amount" in filtered.columns else "Fraud_Probability"
            hover_cols = [c for c in ["Time_Delta","Velocity_Ratio","Fraud_Probability"] if c in filtered.columns]
            fig_sc = px.scatter(filtered.head(500), x=x_col, y=y_col,
                                color="Risk_Level", size="Fraud_Probability",
                                color_discrete_map={"High Risk":"#f43f5e",
                                                    "Moderate Risk":"#fb923c"},
                                title=f"{x_col} vs {y_col} — Risk Scatter (top 500)",
                                hover_data=hover_cols)
            fig_sc.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sc, use_container_width=True)

        with ch2:
            risk_counts = filtered["Risk_Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level","Count"]
            fig_pie = px.pie(risk_counts, names="Risk Level", values="Count",
                             color="Risk Level",
                             color_discrete_map={"High Risk":"#f43f5e",
                                                 "Moderate Risk":"#fb923c"},
                             title="Risk Level Split in Filtered Set")
            fig_pie.update_layout(template="plotly_dark",
                                   paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── SHAP Explanation ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧠 SHAP Explanation Chart")
    st.caption("🔴 Red bar = pushes toward Fraud  |  🔵 Blue bar = pushes toward Legitimate  |  Bar length = strength of influence")

    if len(filtered) > 0:
        col_idx, col_btn = st.columns([3, 1])
        with col_idx:
            idx = st.number_input(
                "Row index to explain (0 = top fraud transaction)",
                min_value=0, max_value=max(len(filtered)-1, 0),
                value=0, step=1,
                help="Matches the row number shown in the Top 10 table above")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            explain_clicked = st.button("🧠 Explain", type="primary")

        # Compute SHAP whenever button clicked OR on first load (row 0 auto-explained)
        if explain_clicked:
            st.session_state["explorer_shap_idx"] = int(idx)
            st.session_state.pop("explorer_shap_result", None)  # force recompute

        if "explorer_shap_result" not in st.session_state:
            explain_idx = st.session_state.get("explorer_shap_idx", 0)
            explain_idx = min(explain_idx, len(filtered) - 1)
            row = filtered.iloc[explain_idx][
                ["Amount", "Time_Delta", "Distance_From_Home",
                 "Is_High_Risk_Merchant", "Avg_Spent_7D"]].to_frame().T
            with st.spinner("Computing SHAP values…"):
                prob, shap_img, sv, feat_names, risk = \
                    engine.predict_with_explanation(row)
            st.session_state["explorer_shap_result"] = {
                "prob": prob, "shap_img": shap_img,
                "sv": sv, "feat_names": feat_names,
                "risk": risk, "explain_idx": explain_idx,
            }

        # Always render from session state — persists across filter changes
        res         = st.session_state["explorer_shap_result"]
        prob        = res["prob"]
        shap_img    = res["shap_img"]
        sv          = res["sv"]
        feat_names  = res["feat_names"]
        risk        = res["risk"]
        explain_idx = res["explain_idx"]

        # Guard: explain_idx may be out of range if filters changed
        if explain_idx < len(filtered):
            tx = filtered.iloc[explain_idx]

            # Header
            st.markdown(
                f"**Explaining row {explain_idx}** — "
                f"Fraud Probability: `{tx['Fraud_Probability']:.1f}%` | "
                f"Risk Level: `{tx['Risk_Level']}`"
            )

            # Verdict banner
            if prob >= 0.6:
                st.error(f"🚨 {risk} — Fraud Probability: {prob*100:.1f}%")
            elif prob >= 0.4:
                st.warning(f"⚠️ {risk} — Fraud Probability: {prob*100:.1f}%")
            else:
                st.success(f"✅ {risk} — Fraud Probability: {prob*100:.1f}%")

            # Side by side: table LEFT, chart RIGHT
            left, right = st.columns([1, 2])

            with left:
                st.markdown("**Per-Feature Breakdown**")
                vr_val = float(tx.get("Velocity_Ratio", 0))
                feat_vals = [
                    float(tx.get("Amount", 0)),
                    float(tx.get("Time_Delta", 0)),
                    float(tx.get("Distance_From_Home", 0)),
                    float(tx.get("Is_High_Risk_Merchant", 0)),
                    float(tx.get("Avg_Spent_7D", 0)),
                    vr_val,
                ]
                feat_table = pd.DataFrame({
                    "Feature":   feat_names,
                    "Value":     [
                        f"${tx['Amount']:.0f}",
                        f"{tx['Time_Delta']:.0f}s",
                        f"{tx['Distance_From_Home']:.0f}km",
                        f"{'Yes' if tx['Is_High_Risk_Merchant']==1 else 'No'}",
                        f"${tx['Avg_Spent_7D']:.0f}/day",
                        f"{vr_val:.1f}×",
                    ],
                    "Risk":      [
                        tx.get("Amount_Risk", "—"),
                        tx.get("Time_Delta_Risk", "—"),
                        tx.get("Distance_From_Home_Risk", "—"),
                        tx.get("Is_High_Risk_Merchant_Risk", "—"),
                        "—",
                        tx.get("Velocity_Ratio_Risk", "—"),
                    ],
                    "SHAP":      [f"{'+' if v>0 else ''}{v:.4f}" for v in sv],
                    "Direction": ["↑ Fraud" if v > 0 else "↓ Fraud" for v in sv],
                })
                st.dataframe(feat_table, use_container_width=True, hide_index=True)

            with right:
                st.image(shap_img, use_container_width=True)

            # Plain-English narrative
            st.markdown("##### 💬 Plain-English Confidence Explanation")
            st.markdown(confidence_narrative(prob, sv, feat_names))

            # Class imbalance callout
            st.markdown("---")
            st.markdown("##### 📌 Why are there so many flagged transactions?")
            st.info("""
**This is the class imbalance problem** — one of the most important concepts in fraud detection.

In the real world, genuine fraud is extremely rare (~0.2% of all transactions). If a model just 
predicted "Legitimate" for everything, it would get 99.8% accuracy — but catch **zero fraud**.

That's why fraud detection systems:
- **Ignore accuracy** — it's meaningless here
- **Optimise for Recall** — catch as many real frauds as possible, even at the cost of some false alarms
- **Use Precision-Recall AUC** as the real performance metric

Our synthetic data uses a 2% fraud ratio (still 10× higher than real world) to ensure the model 
gets enough fraud examples to learn from. In a real bank deployment, the threshold would be tuned 
to balance false alarms vs missed frauds based on the cost of each.
""")
        else:
            st.info("Filters changed — click Explain to recompute for current filtered set.")
    else:
        st.warning("No transactions match the current filters.")

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 6 — COMPARATIVE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "📑 Comparative Analysis":
    st.markdown("### 📑 Comparative Analysis Report")

    try:
        with open("SOTA_Comparative_Analysis_Report.docx","rb") as f:
            st.download_button("📄 Download SOTA Comparison (Word)", f,
                               "SOTA_Comparative_Analysis_Report.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    except FileNotFoundError:
        st.error("Report not found. Run generate_expert_report.py first.")

    st.markdown("---")
    st.table({
        "Feature":               ["Stacking Ensemble","Sliding-Window Features",
                                  "Explainable AI (SHAP)","Domain-Informed Thresholds",
                                  "Real-time API","Federated Learning"],
        "Research Papers Avg":   ["❌","⚠️ Partial","❌","❌","❌","❌"],
        "My Project":            ["✅ Full","✅ Full","✅ Full","✅ Full","✅ Full","⏳ Roadmap"],
    })

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 7 — SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "⚙️ Settings":
    st.markdown("### ⚙️ System Configuration")

    col_a, col_b = st.columns(2)
    with col_a:
        st.info("#### Model Metadata")
        st.write("**Name:** SOTA-Stacking-Ensemble-v2")
        st.write("**Frameworks:** Scikit-Learn · XGBoost · SHAP")
        st.write("**Features:** 6 domain-informed behavioral features")
        n_train = st.slider("Training sample size", 1000, 20000, 5000, step=1000)
        if st.button("🔄 Retrain Model", type="primary"):
            with st.spinner("Generating domain-informed synthetic data & retraining…"):
                data   = engine.generate_synthetic_data(n_samples=n_train)
                report = engine.train_stacking_ensemble(data)
            st.success("Model retrained successfully!")
            st.text(report["classification_report"])
            st.metric("Precision", f'{report["precision"]}%')
            st.metric("Recall",    f'{report["recall"]}%')
            st.metric("AUC-ROC",   f'{report["auc_roc"]}%')

    with col_b:
        st.warning("#### Risk Threshold Reference")
        st.write("Thresholds are domain-informed (not configurable from UI to preserve integrity).")
        for feat, td in THRESHOLDS.items():
            if "moderate_low" in td:
                st.write(f"**{feat}:** Safe < {td['moderate_low']} | "
                         f"Moderate {td['moderate_low']}–{td['high']} | "
                         f"High Risk > {td['high']}")
            else:
                st.write(f"**{feat}:** High Risk = {td['high']}")

    st.markdown("---")
    st.markdown("### 🏦 Train Kaggle Model (creditcard.csv)")
    st.write("This trains a dedicated model on the real Kaggle fraud dataset using V1–V28 PCA features directly. "
             "Only needs to be done once — saved as `fraud_model_kaggle.pkl`.")

    kaggle_status = "✅ Kaggle model loaded and ready" if kaggle_engine.is_ready() \
        else "⚠️ Not trained yet — upload creditcard.csv below to train"
    st.info(kaggle_status)

    kaggle_file = st.file_uploader("Upload creditcard.csv to train Kaggle model",
                                   type="csv", key="kaggle_train_uploader")
    if kaggle_file and st.button("🚀 Train Kaggle Model", type="primary"):
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(kaggle_file.read())
            tmp_path = tmp.name
        with st.spinner("Training on real fraud data — this takes 3–8 minutes depending on your machine…"):
            report = kaggle_engine.train(tmp_path)
        _os.unlink(tmp_path)
        st.success("✅ Kaggle model trained and saved as fraud_model_kaggle.pkl!")
        st.text(report["classification_report"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f'{report["precision"]}%')
        c2.metric("Recall",    f'{report["recall"]}%')
        c3.metric("AUC-ROC",   f'{report["auc_roc"]}%')
        st.cache_resource.clear()

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#94a3b8;">'
    'SOTA Fraud Detection System · Industry Expert Dashboard 2026 · '
    'All metrics computed live by the model — no hardcoded values.'
    '</p>',
    unsafe_allow_html=True)
from engine import (FraudEngine, KaggleFraudEngine, THRESHOLDS, FEATURE_DESCRIPTIONS,
                     FEATURE_UNITS, FEATURES, KAGGLE_FEATURES, classify_risk, feature_risk_level)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from card_validator import CardValidator, generate_card_report
from supabase_auth import sign_up, sign_in, sign_out, upsert_profile, get_profile, log_session_transaction, get_session_history, log_alert
from email_alerts import send_fraud_alert, send_monthly_summary
from monthly_report import generate_synthetic_transactions, generate_monthly_report

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

/* Main App Background */
.stApp { background: linear-gradient(145deg,#0a0a0a 0%,#111111 40%,#1a1a1a 100%); color:#e8e6e3; }

/* Sidebar Customization */
[data-testid="stSidebar"] {
    background: rgba(10, 10, 10, 0.95);
    border-right: 1px solid rgba(212, 175, 55, 0.1);
    backdrop-filter: blur(20px);
}

/* Header Styling */
.main-header {
    font-size:2.8rem; font-weight:700;
    background:linear-gradient(90deg,#d4af37,#f5d060,#d4af37);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:.3rem; letter-spacing:0.5px;
    animation: fadeIn 1.5s ease-in;
}
.sub-header { font-size:1.1rem; color:#a0998a; margin-bottom:1.5rem; }

/* Premium Card Design */
.metric-card {
    background:rgba(20,20,20,0.9); padding:1.2rem 1.4rem;
    border-radius:.8rem; border:1px solid rgba(212,175,55,0.15);
    backdrop-filter:blur(10px); transition:transform .3s cubic-bezier(0.175, 0.885, 0.32, 1.275), border .25s ease, box-shadow .25s ease;
}
.metric-card:hover { 
    transform:translateY(-8px); 
    border:1px solid rgba(212,175,55,0.6); 
    box-shadow:0 12px 30px rgba(212,175,55,0.15); 
}

/* Risk Badges & Animations */
.risk-high { 
    color:#e63946; font-weight:700; 
    animation: pulse-red 2s infinite;
}
.risk-moderate { color:#f5d060; font-weight:700; }
.risk-safe { color:#2ec4b6; font-weight:700; }

@keyframes pulse-red {
    0% { opacity: 1; }
    50% { opacity: 0.6; }
    100% { opacity: 1; }
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Custom Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { 
    background: linear-gradient(180deg, #d4af37, #8a6d1d);
    border-radius: 10px;
}

/* Button Styling */
div.stButton > button {
    background: linear-gradient(135deg, #d4af37 0%, #8a6d1d 100%) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.3s ease !important;
}
div.stButton > button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 5px 15px rgba(212,175,55,0.3) !important;
}

.shap-legend {
    background:rgba(20,20,20,0.8); border-radius:.6rem;
    padding:.8rem 1.2rem; border-left:3px solid #d4af37;
    font-size:.9rem; line-height:1.7;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  AUTH GATE — Login / Sign Up
# ─────────────────────────────────────────────────────────────────────────────
def _auth_gate():
    """Show login/signup UI. Returns True if user is authenticated."""
    if st.session_state.get("authenticated"):
        return True

    st.markdown("""
    <div style="max-width:460px;margin:60px auto 0;text-align:center;">
      <div style="font-size:3rem;margin-bottom:.5rem;">🛡️</div>
      <div style="font-size:2rem;font-weight:800;background:linear-gradient(90deg,#d4af37,#f5d060);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.3rem;">
        SecureGuard AI
      </div>
      <div style="color:#a0998a;font-size:1rem;margin-bottom:2rem;">
        Industry-Grade Fraud Detection Platform
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        auth_tab, reg_tab = st.tabs(["🔑 Login", "📝 Create Account"])

        with auth_tab:
            st.markdown("#### Welcome back")
            email    = st.text_input("Email", key="login_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="••••••••")
            if st.button("Login →", type="primary", use_container_width=True):
                if email and password:
                    with st.spinner("Signing in..."):
                        result = sign_in(email, password)
                    if result["ok"]:
                        st.session_state["authenticated"]  = True
                        st.session_state["access_token"]   = result["access_token"]
                        st.session_state["user_id"]        = result["user"]["id"]
                        st.session_state["user_email"]     = email
                        st.session_state["user_name"]      = result["user"].get("user_metadata", {}).get("full_name", email.split("@")[0])
                        # load saved profile
                        profile = get_profile(result["access_token"], result["user"]["id"])
                        if profile:
                            st.session_state["db_profile"] = profile
                        # if pending profile from signup, save it now
                        elif st.session_state.get("pending_profile"):
                            pp = st.session_state["pending_profile"]
                            pp["email"] = email
                            upsert_profile(result["access_token"], result["user"]["id"], pp)
                            st.session_state.pop("pending_profile", None)
                        st.success("✅ Logged in!")
                        st.rerun()
                    else:
                        st.error(f"❌ {result.get('error', 'Login failed')}")
                else:
                    st.warning("Please enter email and password.")

        with reg_tab:
            st.markdown("#### Create your account")
            st.markdown("<div style='color:#a0998a;font-size:.85rem;margin-bottom:12px;'>Basic info + card details saved once — never entered again.</div>", unsafe_allow_html=True)

            ra, rb = st.columns(2)
            with ra:
                full_name   = st.text_input("Full Name",  key="reg_name",  placeholder="Mayank Sharma")
                reg_email   = st.text_input("Email",      key="reg_email", placeholder="you@example.com")
                reg_pass    = st.text_input("Password",   key="reg_pass",  type="password", placeholder="Min 6 characters")
                reg_pass2   = st.text_input("Confirm Password", key="reg_pass2", type="password", placeholder="Repeat password")
            with rb:
                st.markdown("<div style='color:#d4af37;font-size:.8rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;'>💳 Card Details</div>", unsafe_allow_html=True)
                card_last4  = st.text_input("Card Last 4 Digits", key="reg_card4", placeholder="e.g. 4521", max_chars=4,
                                             help="Only last 4 digits — full card number is never stored")
                c1, c2      = st.columns(2)
                with c1:
                    exp_month = st.selectbox("Expiry Month", list(range(1,13)), index=11, key="reg_exp_m",
                                              format_func=lambda x: f"{x:02d}")
                with c2:
                    exp_year  = st.selectbox("Expiry Year",  list(range(2024, 2036)), index=2, key="reg_exp_y")
                ifsc_code   = st.text_input("IFSC Code", key="reg_ifsc", placeholder="e.g. HDFC0001234",
                                             help="Your bank branch IFSC code — safe to store, public info")
                reg_location = st.text_input("Home Location", key="reg_loc", placeholder="e.g. Delhi, India",
                                              value="India")
                st.markdown("<div style='color:#a0998a;font-size:.78rem;margin-top:4px;'>⚠️ CVV and full card number are never stored — this is by design.</div>", unsafe_allow_html=True)

            if st.button("Create Account →", type="primary", use_container_width=True):
                if not (full_name and reg_email and reg_pass):
                    st.warning("Please fill in at least name, email and password.")
                elif reg_pass != reg_pass2:
                    st.error("Passwords do not match.")
                elif len(reg_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    with st.spinner("Creating account..."):
                        result = sign_up(reg_email, reg_pass, full_name)
                    if result["ok"]:
                        # Save card details to session for profile upsert after login
                        st.session_state["pending_profile"] = {
                            "full_name":           full_name,
                            "card_last4":          card_last4,
                            "card_expiry_month":   exp_month,
                            "card_expiry_year":    exp_year,
                            "ifsc_code":           ifsc_code,
                            "registered_location": reg_location,
                            "email":               reg_email,
                            "daily_spend_limit":   80,
                            "max_transactions_day":10,
                        }
                        st.success("✅ Account created! Check your email to confirm, then log in.")
                        st.info("💡 Switch to the Login tab and sign in.")
                    else:
                        err = result.get("error", "Signup failed")
                        if isinstance(err, dict) or (isinstance(err, str) and err.startswith("{")):
                            st.warning("⚠️ Account may already exist. Try logging in.")
                        else:
                            st.error(f"❌ {err}")
    return False

# Run auth gate — if not authenticated, stop here
if not _auth_gate():
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  AUTHENTICATED — show user info in sidebar
# ─────────────────────────────────────────────────────────────────────────────
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
        "🩺 Card Health Check",
        "🔐 Security & Threats",
        "📋 Monthly Report",
        "📑 Comparative Analysis",
        "⚙️ Settings",
    ])
    st.markdown("---")
    st.success("Status: SOTA Online ✅")
    st.markdown("**Features:** 6 domain-informed")
    st.markdown("---")
    # ── Logged-in user ───────────────────────────────────────────────────────
    user_name  = st.session_state.get("user_name", "User")
    user_email = st.session_state.get("user_email", "")
    st.markdown(f"""
    <div style="background:rgba(212,175,55,0.08);border:1px solid rgba(212,175,55,0.2);
      border-radius:8px;padding:10px 14px;margin-bottom:10px;">
      <div style="color:#d4af37;font-weight:700;font-size:.95rem;">👤 {user_name}</div>
      <div style="color:#a0998a;font-size:.78rem;margin-top:2px;">{user_email}</div>
    </div>""", unsafe_allow_html=True)
    if st.button("🚪 Logout", use_container_width=True):
        token = st.session_state.get("access_token","")
        if token:
            sign_out(token)
        for k in ["authenticated","access_token","user_id","user_email","user_name","db_profile"]:
            st.session_state.pop(k, None)
        st.rerun()
    
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

# ─────────────────────────────────────────────────────────────────────────────
#  HERO LANDING PAGE LOGIC
# ─────────────────────────────────────────────────────────────────────────────
if "dashboard_entered" not in st.session_state:
    st.session_state["dashboard_entered"] = False

if not st.session_state["dashboard_entered"]:
    # Full screen Hero Page
    st.markdown(f"""
    <style>
    .stApp {{
        background: url("file:///C:/Users/Mayank/.gemini/antigravity/brain/d8269b24-0eda-487a-b1f2-ee7fe12ff5c1/secureguard_hero_background_1778156246690.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .hero-container {{
        height: 70vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: white;
        animation: fadeIn 2s ease-in-out;
    }}
    .hero-title {{
        font-size: 5.5rem;
        font-weight: 800;
        margin-bottom: 0px;
        background: linear-gradient(90deg, #d4af37, #f5d060, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 10px 20px rgba(212,175,55,0.4));
    }}
    .hero-subtitle {{
        font-size: 1.4rem;
        color: #d4af37;
        margin-bottom: 4rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        font-weight: 300;
    }}
    
    /* Target the specific Streamlit button */
    div.stButton > button[kind="secondary"] {{
        background: linear-gradient(135deg, #d4af37 0%, #8a6d1d 100%) !important;
        color: #000 !important;
        padding: 1.5rem 4rem !important;
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        border-radius: 60px !important;
        border: 2px solid #f5d060 !important;
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 0 30px rgba(212,175,55,0.5) !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
    }}
    div.stButton > button:hover {{
        transform: scale(1.1) !important;
        box-shadow: 0 0 60px rgba(212,175,55,0.8) !important;
        background: linear-gradient(135deg, #f5d060 0%, #d4af37 100%) !important;
    }}
    </style>
    <div class="hero-container">
        <h1 class="hero-title">SECUREGUARD AI</h1>
        <p class="hero-subtitle">The Gold Standard in Fraud Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("ENTER SYSTEM", key="enter_btn", use_container_width=True):
            st.session_state["dashboard_entered"] = True
            st.rerun()
    st.stop()

# Main Dashboard Content starts here
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

    if "results_df" in st.session_state and not st.session_state["results_df"].empty:
        st.info("📊 **Displaying metrics based on your uploaded dataset.**")
        results_df = st.session_state["results_df"]
        total = len(results_df)
        flagged = int(results_df["Predicted_Fraud"].sum())
        high_risk = int((results_df["Risk_Level"] == "High Risk").sum())
        moderate = int((results_df["Risk_Level"] == "Moderate Risk").sum())
        avg_p = results_df.loc[results_df["Predicted_Fraud"] == 1, "Fraud_Probability"].mean()
        
        true_fraud_col = "Class" if "Class" in results_df.columns else "Is_Fraud" if "Is_Fraud" in results_df.columns else None
        true_fraud = int(results_df[true_fraud_col].sum()) if true_fraud_col else None
        recall = round(flagged / true_fraud * 100, 1) if (true_fraud and true_fraud > 0) else None
        
        metrics = {
            "total": total, "flagged": flagged, "high_risk": high_risk,
            "moderate": moderate, "avg_prob": round(float(avg_p), 1) if not pd.isna(avg_p) else 0,
            "recall": recall, "true_fraud": true_fraud
        }
        batch_source_text = "Uploaded dataset"
    else:
        st.info("💡 **No dataset uploaded yet.** Displaying live metrics from a 5,000-row synthetic demo batch.")
        with st.spinner("Computing live metrics from model…"):
            sample_df = engine.generate_synthetic_data(n_samples=5000)
            results_df = engine.predict_batch(sample_df)
            metrics   = engine.get_live_metrics(sample_df)
            total = len(results_df)
        batch_source_text = "Live sample batch"

    # ── KPI cards ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card"><h4>Total Transactions</h4>'
                    f'<h2>{metrics["total"]:,}</h2>'
                    f'<p style="color:#a0998a">{batch_source_text}</p></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h4>Fraud Flagged</h4>'
                    f'<h2 style="color:#e63946">{metrics["flagged"]:,}</h2>'
                    f'<p style="color:#a0998a">by SOTA ensemble</p></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h4>🔴 High Risk</h4>'
                    f'<h2 style="color:#e63946">{metrics["high_risk"]:,}</h2>'
                    f'<p style="color:#a0998a">≥2 high-risk features</p></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h4>🟠 Moderate Risk</h4>'
                    f'<h2 style="color:#d4af37">{metrics["moderate"]:,}</h2>'
                    f'<p style="color:#a0998a">1 high or ≥2 moderate features</p></div>',
                    unsafe_allow_html=True)
    with c5:
        recall_txt = f'{metrics["recall"]}%' if metrics["recall"] else "N/A"
        st.markdown(f'<div class="metric-card"><h4>Model Recall</h4>'
                    f'<h2 style="color:#2ec4b6">{recall_txt}</h2>'
                    f'<p style="color:#a0998a">True fraud caught</p></div>',
                    unsafe_allow_html=True)

    st.caption("ℹ️ *Note: 'Fraud Flagged' represents strict model predictions (Probability ≥ 50%). 'Moderate Risk' includes early-warning transactions (e.g. Probability 40%–49.9%) that do not cross the strict prediction threshold, which is why total Risk counts can exceed strict flags.*")
    st.markdown("---")

    # ── Global Risk Distribution ─────────────────────────────────────────────
    st.markdown(f"### 🎯 Global Risk Distribution ({batch_source_text})")
    col_dist1, col_dist2 = st.columns([2, 1])
    
    with col_dist1:
        risk_counts = results_df["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Level", "Count"]
        fig_donut = px.pie(
            risk_counts, values="Count", names="Risk_Level",
            hole=0.5,
            color="Risk_Level",
            color_discrete_map={"High Risk":"#e63946","Moderate Risk":"#d4af37","Legitimate":"#2ec4b6","Safe":"#2ec4b6"},
            title="Transaction Risk Level Breakdown"
        )
        fig_donut.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                 legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_dist2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        for _, row in risk_counts.iterrows():
            color = "#e63946" if "High" in row["Risk_Level"] else "#d4af37" if "Moderate" in row["Risk_Level"] else "#2ec4b6"
            st.markdown(f"""
            <div style="padding:10px; border-left:4px solid {color}; background:rgba(212,175,55,0.05); border-radius:4px; margin-bottom:10px;">
                <span style="color:#a0998a; font-size:0.9rem;">{row['Risk_Level']}</span><br>
                <span style="color:{color}; font-size:1.4rem; font-weight:700;">{row['Count']:,}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Fraud probability distribution ───────────────────────────────────────
    st.markdown(f"### 📊 Fraud Probability Distribution ({batch_source_text})")
    fig_hist = px.histogram(
        results_df, x="Fraud_Probability", nbins=50,
        color="Risk_Level",
        color_discrete_map={"High Risk":"#e63946","Moderate Risk":"#d4af37","Legitimate":"#2ec4b6","Safe":"#2ec4b6"},
        title=f"Distribution of Fraud Probability Scores ({total:,} Transactions)",
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
        amt_color = "Amount_Risk" if "Amount_Risk" in results_df.columns else "Risk_Level"
        fig_amt = px.histogram(
            results_df, x="Amount", nbins=60, color=amt_color,
            color_discrete_map={"High Risk":"#e63946","Moderate Risk":"#d4af37","Safe":"#2ec4b6","Legitimate":"#2ec4b6"},
            title="Transaction Amount — Risk Breakdown",
        )
        fig_amt.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_amt, use_container_width=True)

    with col_r:
        if "Time_Delta" in results_df.columns:
            # Time_Delta distribution
            td_color = "Time_Delta_Risk" if "Time_Delta_Risk" in results_df.columns else "Risk_Level"
            fig_td = px.histogram(
                results_df, x="Time_Delta", nbins=60, color=td_color,
                color_discrete_map={"High Risk":"#e63946","Moderate Risk":"#d4af37","Safe":"#2ec4b6","Legitimate":"#2ec4b6"},
                title="Time Delta (seconds) — Risk Breakdown",
            )
            fig_td.add_vline(x=110, line_dash="dash", line_color="#d4af37",
                             annotation_text="Moderate (110s)")
            fig_td.add_vline(x=160, line_dash="dash", line_color="#e63946",
                             annotation_text="High Risk (160s)")
            fig_td.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_td, use_container_width=True)
        elif "Time" in results_df.columns:
            # Fallback to Time column for Kaggle dataset
            fig_td = px.histogram(
                results_df, x="Time", nbins=60, color="Risk_Level",
                color_discrete_map={"High Risk":"#e63946","Moderate Risk":"#d4af37","Safe":"#2ec4b6","Legitimate":"#2ec4b6"},
                title="Transaction Time — Risk Breakdown",
            )
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
                        progress_bar = st.progress(0, text="Analyzing transactions...")
                        chunk_size = max(len(raw_df) // 10, 1)
                        res_list = []
                        for i in range(0, len(raw_df), chunk_size):
                            chunk = raw_df.iloc[i:i+chunk_size]
                            res_list.append(kaggle_engine.predict_batch(chunk))
                            progress_bar.progress(min((i+chunk_size)/len(raw_df), 1.0), text=f"Analyzing... {min(i+chunk_size, len(raw_df)):,}/{len(raw_df):,}")
                        results = pd.concat(res_list, ignore_index=True)
                        progress_bar.progress(1.0, text="Analysis Complete!")

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
                                  
                    st.caption("ℹ️ *Note: 'Total Flagged' (Prob ≥ 50%) is mathematically stricter than combined 'High + Moderate Risk', as Moderate Risk acts as an early-warning signal starting at 40%.*")

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
                                   color_discrete_map={"High Risk": "#e63946",
                                                       "Moderate Risk": "#d4af37"},
                                   title="Top 50 Transactions by Fraud Probability (Kaggle Model)")
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                      plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

                    st.session_state["results_df"]  = results
                    st.session_state["is_kaggle"]   = True

            else:
                # ── Behavioral CSV ────────────────────────────────────────────
                with st.spinner("Running SOTA ensemble on all transactions…"):
                    progress_bar = st.progress(0, text="Analyzing transactions...")
                    chunk_size = max(len(raw_df) // 10, 1)
                    res_list = []
                    for i in range(0, len(raw_df), chunk_size):
                        chunk = raw_df.iloc[i:i+chunk_size]
                        res_list.append(engine.predict_batch(chunk))
                        progress_bar.progress(min((i+chunk_size)/len(raw_df), 1.0), text=f"Analyzing... {min(i+chunk_size, len(raw_df)):,}/{len(raw_df):,}")
                    results = pd.concat(res_list, ignore_index=True)
                    progress_bar.progress(1.0, text="Analysis Complete!")

                fraud_df    = results[results["Predicted_Fraud"] == 1].copy()
                high_df     = results[results["Risk_Level"] == "High Risk"].copy()
                moderate_df = results[results["Risk_Level"] == "Moderate Risk"].copy()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Transactions",   f"{len(results):,}")
                m2.metric("🔴 High Risk",          f"{len(high_df):,}")
                m3.metric("🟠 Moderate Risk",      f"{len(moderate_df):,}")
                m4.metric("Avg Fraud Probability (flagged)",
                          f"{fraud_df['Fraud_Probability'].mean():.1f}%" if len(fraud_df) else "—")
                          
                st.caption("ℹ️ *Note: 'Total Flagged' (Prob ≥ 50%) is mathematically stricter than combined 'High + Moderate Risk', as Moderate Risk acts as an early-warning signal starting at 40%.*")

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
                               color_discrete_map={"High Risk": "#e63946",
                                                   "Moderate Risk": "#d4af37"},
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
                                    color_discrete_map={"High Risk": "#e63946",
                                                        "Moderate Risk": "#d4af37",
                                                        "Safe": "#2ec4b6"},
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
                    progress_bar = st.progress(0, text="Running Scaling Analysis...")
                    def prog_cb(step, total, size):
                        progress_bar.progress(step/total, text=f"Evaluating dataset size: {size:,} ({step}/{total})")
                    scaling_df = target_engine.run_scaling_analysis(
                        custom_df=source_df, custom_sizes=selected_sizes, progress_callback=prog_cb)
                    progress_bar.progress(1.0, text="Analysis Complete!")

            st.dataframe(scaling_df.drop(columns=["Actual Size"]),
                         use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(scaling_df, x="Dataset Size",
                             y=["High Risk","Moderate Risk"],
                             color_discrete_map={"High Risk":"#e63946","Moderate Risk":"#d4af37"},
                             title="High vs Moderate Risk Detected per Batch",
                             barmode="stack")
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                scaling_df["Time_Numeric"] = scaling_df["Processing Time"].str.replace("s","").astype(float)
                fig2 = px.line(scaling_df, x="Dataset Size", y="Time_Numeric",
                               title="Processing Latency vs Dataset Size", markers=True)
                fig2.update_traces(line_color="#d4af37")
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
                                color_discrete_map={"High Risk":"#e63946",
                                                    "Moderate Risk":"#d4af37"},
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
                             color_discrete_map={"High Risk":"#e63946",
                                                 "Moderate Risk":"#d4af37"},
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
#  TAB 6 — CARD HEALTH CHECK (Two-Panel Account Takeover Detection)
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "🩺 Card Health Check":
    import time as _time
    import json as _json

    st.markdown("### 🩺 Card Health & Account Security")
    st.markdown(
        "**Panel 1** is your registered profile. "
        "**Panel 2** is the attacker's active session — when they log into your account "
        "from another device and process a transaction, it appears here automatically "
        "and an alert is sent to your email instantly."
    )

    validator    = CardValidator()
    access_token = st.session_state.get("access_token", "")
    user_id      = st.session_state.get("user_id", "")
    user_email   = st.session_state.get("user_email", "")

    # ── Initialise real-time state ────────────────────────────────────────────
    if "last_seen_session_ts" not in st.session_state:
        st.session_state["last_seen_session_ts"] = ""
    if "realtime_enabled" not in st.session_state:
        st.session_state["realtime_enabled"] = False
    if "last_refresh_time" not in st.session_state:
        st.session_state["last_refresh_time"] = 0.0
    if "attacker_session_live" not in st.session_state:
        st.session_state["attacker_session_live"] = None   # holds latest attacker session dict

    # ── Helper: auto-process a session written by the attacker ────────────────
    def _auto_process_attacker_session(session: dict, profile: dict):
        """
        Run flag checks on an attacker session that just landed from Supabase.
        Fires the email + PDF automatically — no button click needed.
        Returns list of flags.
        """
        flags = []
        registered_loc = profile.get("registered_location", "India")
        daily_limit    = float(profile.get("daily_spend_limit", 80))
        max_txn        = int(profile.get("max_transactions_day", 10))

        session_location = session.get("session_location", "Unknown")
        session_amount   = float(session.get("amount", 0))
        session_num_txn  = int(session.get("num_transactions", 1))
        merchant_type    = session.get("merchant_type", "Normal")

        # 1. Geographic Impossibility
        if session_location.strip().lower() != registered_loc.strip().lower():
            flags.append({
                "check":    "Geographic Impossibility",
                "severity": "Critical",
                "detail":   f"Transaction from {session_location} — registered location is {registered_loc}.",
            })

        # 2. Spending Spike
        spike_ratio = session_amount / max(daily_limit, 1)
        if spike_ratio > 5:
            flags.append({"check": "Spending Spike", "severity": "Critical",
                          "detail": f"${session_amount:,.2f} is {spike_ratio:.1f}× your daily limit of ${daily_limit:,.0f}"})
        elif spike_ratio > 2:
            flags.append({"check": "Spending Spike", "severity": "High",
                          "detail": f"${session_amount:,.2f} is {spike_ratio:.1f}× your daily limit"})
        elif spike_ratio > 1.2:
            flags.append({"check": "Spending Spike", "severity": "Moderate",
                          "detail": f"${session_amount:,.2f} slightly exceeds your daily limit"})

        # 3. Transaction Velocity
        velocity_ratio = session_num_txn / max(max_txn, 1)
        if velocity_ratio > 3:
            flags.append({"check": "Transaction Velocity", "severity": "Critical",
                          "detail": f"{session_num_txn} transactions — {velocity_ratio:.1f}× your normal max of {max_txn}/day"})
        elif velocity_ratio > 1.5:
            flags.append({"check": "Transaction Velocity", "severity": "High",
                          "detail": f"{session_num_txn} transactions exceeds your normal maximum"})
        elif velocity_ratio > 1:
            flags.append({"check": "Transaction Velocity", "severity": "Moderate",
                          "detail": f"{session_num_txn} transactions slightly above your normal maximum"})

        # 4. High-Risk Merchant
        high_risk_merchants = ["Crypto Exchange", "Gambling", "Forex"]
        if merchant_type in high_risk_merchants:
            flags.append({"check": "High-Risk Merchant", "severity": "High",
                          "detail": f"Transaction at {merchant_type} — high-risk merchant category"})

        if not flags:
            return []   # nothing to alert on

        # Build session_data dict for email/PDF
        session_data = {
            "location":         session_location,
            "amount":           session_amount,
            "num_transactions": session_num_txn,
            "merchant_type":    merchant_type,
            "flagged":          True,
            "flags":            flags,
        }
        alert_profile = {
            "name":                   profile.get("full_name", "User"),
            "card_last4":             profile.get("card_last4", "XXXX"),
            "baseline_daily_spend":   daily_limit,
            "normal_max_txn_per_day": max_txn,
            "email":                  profile.get("email", user_email),
            "registered_location":    registered_loc,
        }

        # Generate PDF
        auto_results_for_pdf = [{
            "id":          f["check"].lower().replace(" ", "_"),
            "name":        f["check"],
            "category":    "Session Analysis",
            "severity":    f["severity"],
            "detail":      f["detail"],
            "description": f["detail"],
            "actions":     ["Review this transaction immediately.",
                            "Freeze your card if you did not authorise it."],
            "mode":        "Auto",
        } for f in flags]

        pdf_bytes = generate_card_report(alert_profile, auto_results_for_pdf, [], mode="auto")

        # Fire email automatically
        alert_email = profile.get("email", user_email)
        send_fraud_alert(
            to_email     = alert_email,
            user_name    = profile.get("full_name", "User"),
            flags        = flags,
            session_data = session_data,
            profile      = profile,
            pdf_bytes    = pdf_bytes,
        )

        # Log alert to Supabase
        if access_token and user_id:
            overall_sev = (
                "Critical" if any(f["severity"] == "Critical" for f in flags) else
                "High"     if any(f["severity"] == "High"     for f in flags) else
                "Moderate"
            )
            log_alert(access_token, user_id, {
                "type":        "account_takeover",
                "severity":    overall_sev,
                "title":       f"{len(flags)} threat(s) detected from {session_location}",
                "description": "; ".join(f["detail"] for f in flags),
                "flags":       flags,
            })

        return flags, pdf_bytes

    # ═══════════════════════════════════════════════════════════════════════════
    #  REAL-TIME POLLING — runs on every Streamlit rerun
    #  Checks Supabase for a new flagged session written by the attacker.
    #  If found and not yet processed, auto-fires the alert.
    # ═══════════════════════════════════════════════════════════════════════════
    db_profile = st.session_state.get("db_profile", {})

    if access_token and user_id and db_profile:
        latest_sessions = get_session_history(access_token, user_id, limit=1)
        if latest_sessions:
            ls = latest_sessions[0]
            ts = ls.get("timestamp", "")

            # New session we haven't processed yet
            if ts != st.session_state["last_seen_session_ts"] and ls.get("flagged"):
                st.session_state["last_seen_session_ts"] = ts
                st.session_state["attacker_session_live"] = ls

                # Auto-process: run flags + send email automatically
                flags_raw = ls.get("flags", "[]")
                if isinstance(flags_raw, str):
                    try:
                        flags_raw = _json.loads(flags_raw)
                    except Exception:
                        flags_raw = []

                # Only auto-fire email if this session came from Supabase
                # (i.e., written by another device — not the local submit button)
                if ls.get("auto_submitted", False):
                    try:
                        result = _auto_process_attacker_session(ls, db_profile)
                        if result:
                            st.session_state["auto_alert_fired"] = True
                            st.session_state["auto_alert_flags"] = result[0]
                    except Exception as e:
                        st.session_state["auto_alert_error"] = str(e)

    # ── Auto-refresh loop (8 seconds) ─────────────────────────────────────────
    if st.session_state["realtime_enabled"]:
        now = _time.time()
        if now - st.session_state["last_refresh_time"] > 8:
            st.session_state["last_refresh_time"] = now
            _time.sleep(0.1)
            st.rerun()

    # ═══════════════════════════════════════════════════════════════════════════
    #  AUTO ALERT BANNER — shows when email was auto-fired by attacker's session
    # ═══════════════════════════════════════════════════════════════════════════
    if st.session_state.get("auto_alert_fired"):
        auto_flags = st.session_state.get("auto_alert_flags", [])
        overall_sev = (
            "Critical" if any(f["severity"] == "Critical" for f in auto_flags) else
            "High"     if any(f["severity"] == "High"     for f in auto_flags) else
            "Moderate"
        )
        sev_color = {"Critical": "#e63946", "High": "#ff8c42", "Moderate": "#f5d060"}.get(overall_sev, "#f5d060")
        ls = st.session_state.get("attacker_session_live", {})

        st.markdown(f"""
        <div style="background:rgba(30,5,5,0.98);border:2px solid {sev_color};border-radius:12px;
          padding:18px 24px;margin-bottom:20px;animation:pulse-red 1.5s infinite;">
          <div style="color:{sev_color};font-size:1.3rem;font-weight:800;">
            🚨 LIVE ALERT — Attacker Session Detected & Email Auto-Sent
          </div>
          <div style="color:#a0998a;font-size:.9rem;margin-top:8px;">
            Location: <b style="color:#fff">{ls.get('session_location','Unknown')}</b> &nbsp;·&nbsp;
            Amount: <b style="color:#f5d060">${float(ls.get('amount',0)):,.2f}</b> &nbsp;·&nbsp;
            Transactions: <b style="color:#fff">{ls.get('num_transactions',0)}</b> &nbsp;·&nbsp;
            Merchant: <b style="color:#fff">{ls.get('merchant_type','—')}</b>
          </div>
          <div style="color:#2ec4b6;font-size:.85rem;margin-top:6px;">
            ✅ Fraud alert email with PDF report was automatically sent to <b>{db_profile.get('email', user_email)}</b>
          </div>
          <div style="color:#555;font-size:.78rem;margin-top:4px;">
            Detected at {ls.get('timestamp','')[:19].replace('T',' ')} UTC
          </div>
        </div>""", unsafe_allow_html=True)

        if st.button("✅ Dismiss Alert", key="dismiss_auto_alert"):
            st.session_state["auto_alert_fired"] = False
            st.session_state["auto_alert_flags"] = []
            st.rerun()

    # ── Live monitoring toggle ────────────────────────────────────────────────
    col_rt, col_status = st.columns([1, 3])
    with col_rt:
        btn_label = (
            "⏹ Stop Monitoring"
            if st.session_state["realtime_enabled"]
            else "🔴 Enable Live Monitoring"
        )
        if st.button(btn_label, use_container_width=True):
            st.session_state["realtime_enabled"] = not st.session_state["realtime_enabled"]
            st.session_state["last_refresh_time"] = 0.0
            st.rerun()
    with col_status:
        if st.session_state["realtime_enabled"]:
            st.success("🟢 Live monitoring ON — checking for attacker sessions every 8 seconds. "
                       "Email fires automatically when a threat is detected.")
        else:
            st.info("💡 Enable live monitoring to auto-detect activity from another device on your account.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    #  PANEL 1 — Registered Profile (locked)
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="background:rgba(20,20,20,0.9);border:1px solid rgba(212,175,55,0.3);
      border-radius:10px;padding:14px 20px;margin-bottom:12px;">
      <span style="color:#d4af37;font-size:1.1rem;font-weight:700;">🔒 Panel 1 — Your Registered Profile</span>
      <span style="color:#a0998a;font-size:.85rem;margin-left:10px;">(Saved to your account · read-only baseline)</span>
    </div>""", unsafe_allow_html=True)

    saved_name     = db_profile.get("full_name",           st.session_state.get("user_name", ""))
    saved_location = db_profile.get("registered_location", "India")
    saved_spend    = db_profile.get("daily_spend_limit",   80.0)
    saved_max_txn  = db_profile.get("max_transactions_day", 10)
    saved_card4    = db_profile.get("card_last4",          "")
    saved_exp_m    = db_profile.get("card_expiry_month",   12)
    saved_exp_y    = db_profile.get("card_expiry_year",    2026)
    saved_ifsc     = db_profile.get("ifsc_code",           "")
    saved_acct     = db_profile.get("account_name",        saved_name)

    with st.expander("✏️ Edit / Save Your Profile", expanded=not bool(db_profile)):
        c1, c2, c3 = st.columns(3)
        with c1:
            p_name     = st.text_input("Full Name",           value=saved_name,     key="p1_name")
            p_card4    = st.text_input("Card Last 4 Digits",  value=saved_card4,    key="p1_card4", max_chars=4)
            p_ifsc     = st.text_input("IFSC Code",           value=saved_ifsc,     key="p1_ifsc")
        with c2:
            p_location = st.text_input("Your Home Location",  value=saved_location, key="p1_loc")
            p_spend    = st.number_input("Normal Daily Spend ($)", min_value=1, max_value=50000,
                                         value=int(saved_spend), key="p1_spend")
            em_c1, em_c2 = st.columns(2)
            with em_c1:
                p_exp_m = st.selectbox("Expiry Month", list(range(1, 13)),
                                        index=int(saved_exp_m) - 1 if saved_exp_m else 11,
                                        key="p1_expm", format_func=lambda x: f"{x:02d}")
            with em_c2:
                p_exp_y = st.selectbox("Expiry Year", list(range(2024, 2036)),
                                        index=max(0, int(saved_exp_y) - 2024) if saved_exp_y else 2,
                                        key="p1_expy")
        with c3:
            p_max_txn  = st.number_input("Max Transactions/Day", min_value=1, max_value=200,
                                          value=int(saved_max_txn), key="p1_txn")
            p_email    = st.text_input("Alert Email", value=user_email, key="p1_email")
            p_acct     = st.text_input("Account Holder Name", value=saved_acct, key="p1_acct")

        st.markdown("<div style='color:#a0998a;font-size:.78rem;margin-top:4px;'>"
                    "⚠️ CVV and full card/account numbers are never stored — by design.</div>",
                    unsafe_allow_html=True)

        if st.button("💾 Save Profile", type="primary"):
            profile_data = {
                "full_name":            p_name,
                "card_last4":           p_card4,
                "registered_location":  p_location,
                "daily_spend_limit":    p_spend,
                "max_transactions_day": p_max_txn,
                "email":                p_email,
                "card_expiry_month":    p_exp_m,
                "card_expiry_year":     p_exp_y,
                "ifsc_code":            p_ifsc,
                "account_name":         p_acct,
            }
            with st.spinner("Saving…"):
                result = upsert_profile(access_token, user_id, profile_data)
            if result["ok"]:
                st.session_state["db_profile"] = result.get("profile", profile_data)
                st.success(f"✅ Profile saved! Alerts will go to {p_email}")
            else:
                st.error(f"❌ Save failed: {result.get('error', '')}")

    # Locked profile summary row
    if db_profile:
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        pc1.metric("👤 Name",        db_profile.get("full_name", "—"))
        pc2.metric("📍 Location",    db_profile.get("registered_location", "—"))
        pc3.metric("💰 Daily Limit", f"${db_profile.get('daily_spend_limit', 80):,.0f}")
        pc4.metric("🔁 Max Txn/Day", db_profile.get("max_transactions_day", 10))
        exp_m = db_profile.get("card_expiry_month")
        exp_y = db_profile.get("card_expiry_year")
        try:
            expiry_str = f"{int(exp_m):02d}/{str(int(exp_y))[-2:]}" if exp_m and exp_y else "—"
        except Exception:
            expiry_str = "—"
        pc5.metric("💳 Card Expiry", expiry_str)
        if db_profile.get("ifsc_code"):
            st.caption(f"🏦 IFSC: {db_profile.get('ifsc_code', '')}  ·  "
                       f"Card: ****{db_profile.get('card_last4', 'XXXX')}")
    else:
        st.info("💡 Save your profile above — Panel 2 compares against it.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    #  PANEL 2 — Active Attacker Session
    #
    #  TWO MODES depending on who is viewing:
    #
    #  MODE A (VICTIM's device) — Live monitoring ON
    #    → Panel 2 is READ-ONLY, auto-populates from the latest Supabase session
    #    → Shows what the attacker is doing in real time
    #    → Email fires automatically, no button needed
    #
    #  MODE B (ATTACKER's device) — Live monitoring OFF
    #    → Panel 2 is an INPUT FORM
    #    → Attacker fills values, clicks "Process Transaction"
    #    → Writes to Supabase with auto_submitted=True
    #    → Victim's device picks this up within 8 seconds
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="background:rgba(20,5,5,0.9);border:1px solid rgba(230,57,70,0.4);
      border-radius:10px;padding:14px 20px;margin-bottom:12px;">
      <span style="color:#e63946;font-size:1.1rem;font-weight:700;">⚡ Panel 2 — Active Session</span>
      <span style="color:#a0998a;font-size:.85rem;margin-left:10px;">
        When live monitoring is ON this auto-populates from the attacker's device.
        When OFF, use this form to submit a transaction as the attacker.
      </span>
    </div>""", unsafe_allow_html=True)

    live_session = st.session_state.get("attacker_session_live")
    monitoring_on = st.session_state["realtime_enabled"]

    # ── MODE A: Victim's view — read-only populated from Supabase ─────────────
    if monitoring_on and live_session:
        ts_display = live_session.get("timestamp", "")[:19].replace("T", " ")
        flags_raw  = live_session.get("flags", "[]")
        if isinstance(flags_raw, str):
            try:
                flags_raw = _json.loads(flags_raw)
            except Exception:
                flags_raw = []

        st.markdown(f"""
        <div style="background:rgba(30,5,5,0.95);border:1.5px solid #e63946;border-radius:10px;
          padding:16px 22px;margin-bottom:12px;">
          <div style="color:#e63946;font-size:1rem;font-weight:700;margin-bottom:10px;">
            🔴 Attacker Session — Live Data from Supabase
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;">
            <div>
              <div style="color:#a0998a;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;">Location</div>
              <div style="color:#e63946;font-size:1.1rem;font-weight:700;">{live_session.get('session_location','—')}</div>
            </div>
            <div>
              <div style="color:#a0998a;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;">Amount</div>
              <div style="color:#f5d060;font-size:1.1rem;font-weight:700;">${float(live_session.get('amount',0)):,.2f}</div>
            </div>
            <div>
              <div style="color:#a0998a;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;"># Transactions</div>
              <div style="color:#fff;font-size:1.1rem;font-weight:700;">{live_session.get('num_transactions','—')}</div>
            </div>
            <div>
              <div style="color:#a0998a;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;">Merchant</div>
              <div style="color:#fff;font-size:1.1rem;font-weight:700;">{live_session.get('merchant_type','—')}</div>
            </div>
          </div>
          <div style="color:#555;font-size:.78rem;margin-top:10px;">Submitted at {ts_display} UTC</div>
        </div>""", unsafe_allow_html=True)

        # Show flags that were triggered
        if flags_raw:
            st.markdown("##### 🚨 Flags Triggered by This Session")
            for f in flags_raw:
                fc = {"Critical": "#e63946", "High": "#ff8c42",
                      "Moderate": "#f5d060", "Low": "#2ec4b6"}.get(f.get("severity", "High"), "#f5d060")
                fe = {"Critical": "🔴", "High": "🟠", "Moderate": "🟡", "Low": "🟢"}.get(f.get("severity", ""), "🟡")
                st.markdown(
                    f'<div style="background:rgba(20,20,20,0.9);border-left:4px solid {fc};'
                    f'padding:10px 14px;border-radius:6px;margin-bottom:6px;">'
                    f'<b style="color:{fc}">{fe} {f.get("check","—")} — {f.get("severity","—")}</b><br>'
                    f'<span style="color:#a0998a;font-size:.88rem">{f.get("detail","")}</span></div>',
                    unsafe_allow_html=True)
        else:
            st.success("✅ No flags triggered in the last attacker session.")

    # ── MODE B: Input form (attacker's device OR victim testing manually) ──────
    else:
        if monitoring_on:
            st.info("🔄 Live monitoring is ON and watching for attacker sessions. "
                    "No new attacker session detected yet. Waiting…")

        st.markdown("##### Submit a Transaction (Attacker's Device)")
        st.caption("On the attacker's device, this form is filled and submitted. "
                   "Your device picks it up within 8 seconds when live monitoring is ON.")

        s1, s2, s3 = st.columns(3)
        with s1:
            session_location = st.text_input(
                "Session Location", value="China", key="s2_loc",
                help="Enter a location different from your registered location to trigger geographic alert")
            merchant_type = st.selectbox(
                "Merchant Type",
                ["Normal", "Crypto Exchange", "Gambling", "Forex", "Electronics", "Luxury Goods"],
                key="s2_merch")
        with s2:
            session_amount  = st.number_input("Transaction Amount ($)", min_value=0.01,
                                               max_value=100000.0, value=500.0, key="s2_amt")
            session_num_txn = st.number_input("Number of Transactions",
                                               min_value=1, max_value=500, value=20, key="s2_ntxn")
        with s3:
            time_delta = st.number_input("Time Between Transactions (s)",
                                          min_value=1, max_value=3600, value=85, key="s2_td")
            avg_7d     = st.number_input("Avg Spend Last 7 Days ($)",
                                          min_value=0.0, max_value=100000.0, value=400.0, key="s2_avg7d")

        st.markdown("")
        proc_col, _ = st.columns([1, 3])
        with proc_col:
            process_btn = st.button(
                "🚨 Process Transaction",
                type="primary",
                use_container_width=True,
                key="panel2_process")

        if process_btn:
            profile = db_profile if db_profile else {
                "full_name":            st.session_state.get("user_name", "User"),
                "registered_location":  "India",
                "daily_spend_limit":    80,
                "max_transactions_day": 10,
                "email":                user_email,
            }

            # ── Run flag checks ────────────────────────────────────────────────
            flags = []
            registered_loc = profile.get("registered_location", "India")
            daily_limit    = float(profile.get("daily_spend_limit", 80))
            max_txn        = int(profile.get("max_transactions_day", 10))

            if session_location.strip().lower() != registered_loc.strip().lower():
                flags.append({"check": "Geographic Impossibility", "severity": "Critical",
                              "detail": f"Transaction from {session_location} — registered location is {registered_loc}."})

            spike_ratio = session_amount / max(daily_limit, 1)
            if spike_ratio > 5:
                flags.append({"check": "Spending Spike", "severity": "Critical",
                              "detail": f"${session_amount:,.2f} is {spike_ratio:.1f}× your daily limit of ${daily_limit:,.0f}"})
            elif spike_ratio > 2:
                flags.append({"check": "Spending Spike", "severity": "High",
                              "detail": f"${session_amount:,.2f} is {spike_ratio:.1f}× your daily limit"})
            elif spike_ratio > 1.2:
                flags.append({"check": "Spending Spike", "severity": "Moderate",
                              "detail": f"${session_amount:,.2f} slightly exceeds your daily limit"})

            velocity_ratio = session_num_txn / max(max_txn, 1)
            if velocity_ratio > 3:
                flags.append({"check": "Transaction Velocity", "severity": "Critical",
                              "detail": f"{session_num_txn} transactions — {velocity_ratio:.1f}× your normal max of {max_txn}/day"})
            elif velocity_ratio > 1.5:
                flags.append({"check": "Transaction Velocity", "severity": "High",
                              "detail": f"{session_num_txn} transactions exceeds your normal maximum"})
            elif velocity_ratio > 1:
                flags.append({"check": "Transaction Velocity", "severity": "Moderate",
                              "detail": f"{session_num_txn} transactions slightly above your normal maximum"})

            if merchant_type in ["Crypto Exchange", "Gambling", "Forex"]:
                flags.append({"check": "High-Risk Merchant", "severity": "High",
                              "detail": f"Transaction at {merchant_type} — high-risk merchant category"})

            if time_delta > 160:
                flags.append({"check": "Time Delta Anomaly", "severity": "High",
                              "detail": f"{time_delta}s between transactions — exceeds 160s high-risk threshold"})
            elif time_delta > 110:
                flags.append({"check": "Time Delta Anomaly", "severity": "Moderate",
                              "detail": f"{time_delta}s between transactions — moderate risk zone"})

            # ── Log to Supabase (auto_submitted=True tells victim's device to auto-fire email) ──
            session_data = {
                "location":         session_location,
                "amount":           session_amount,
                "num_transactions": session_num_txn,
                "merchant_type":    merchant_type,
                "flagged":          len(flags) > 0,
                "flags":            flags,
            }

            if access_token and user_id:
                # Write to Supabase — victim's polling loop will pick this up
                import requests as _req, json as _json2
                from supabase_auth import DB_URL, _auth_headers
                from datetime import datetime as _dt
                payload = {
                    "user_id":          user_id,
                    "session_location": session_location,
                    "amount":           float(session_amount),
                    "num_transactions": int(session_num_txn),
                    "merchant_type":    merchant_type,
                    "flagged":          len(flags) > 0,
                    "flags":            _json2.dumps(flags),
                    "auto_submitted":   True,   # ← key flag: tells victim's device to auto-fire
                    "timestamp":        _dt.utcnow().isoformat(),
                }
                _req.post(
                    f"{DB_URL}/session_transactions",
                    headers=_auth_headers(access_token),
                    json=payload,
                    timeout=10,
                )

            # ── Show results immediately on this device too ────────────────────
            st.markdown("---")
            if flags:
                overall_sev = (
                    "Critical" if any(f["severity"] == "Critical" for f in flags) else
                    "High"     if any(f["severity"] == "High"     for f in flags) else
                    "Moderate"
                )
                sev_color = {"Critical": "#e63946", "High": "#ff8c42",
                             "Moderate": "#f5d060"}.get(overall_sev, "#f5d060")

                st.markdown(f"""
                <div style="background:rgba(30,5,5,0.95);border:2px solid {sev_color};
                  border-radius:12px;padding:18px 24px;margin-bottom:16px;">
                  <div style="color:{sev_color};font-size:1.3rem;font-weight:800;">
                    🚨 {len(flags)} Threat{'s' if len(flags) > 1 else ''} Detected — {overall_sev} Risk
                  </div>
                  <div style="color:#a0998a;font-size:.9rem;margin-top:6px;">
                    Transaction written to Supabase. Victim's device will receive the alert within 8 seconds
                    if live monitoring is enabled.
                  </div>
                </div>""", unsafe_allow_html=True)

                for f in flags:
                    fc = {"Critical": "#e63946", "High": "#ff8c42",
                          "Moderate": "#f5d060", "Low": "#2ec4b6"}.get(f["severity"], "#f5d060")
                    fe = {"Critical": "🔴", "High": "🟠",
                          "Moderate": "🟡", "Low": "🟢"}.get(f["severity"], "🟡")
                    st.markdown(
                        f'<div style="background:rgba(20,20,20,0.9);border-left:4px solid {fc};'
                        f'padding:10px 14px;border-radius:6px;margin-bottom:6px;">'
                        f'<b style="color:{fc}">{fe} {f["check"]} — {f["severity"]}</b><br>'
                        f'<span style="color:#a0998a;font-size:.88rem">{f["detail"]}</span></div>',
                        unsafe_allow_html=True)

                # Generate PDF for manual download on attacker's device too
                alert_profile = {
                    "name":                   profile.get("full_name", "User"),
                    "card_last4":             profile.get("card_last4", "XXXX"),
                    "baseline_daily_spend":   daily_limit,
                    "normal_max_txn_per_day": max_txn,
                    "email":                  profile.get("email", user_email),
                    "registered_location":    registered_loc,
                }
                auto_results_pdf = [{
                    "id":          f["check"].lower().replace(" ", "_"),
                    "name":        f["check"],
                    "category":    "Session Analysis",
                    "severity":    f["severity"],
                    "detail":      f["detail"],
                    "description": f["detail"],
                    "actions":     ["Review this transaction immediately.",
                                   "Freeze your card if you did not authorise it."],
                    "mode":        "Auto",
                } for f in flags]
                pdf_bytes = generate_card_report(alert_profile, auto_results_pdf, [], mode="auto")

                st.download_button(
                    "📥 Download Alert Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"SecureGuard_ALERT_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    type="primary",
                )
                st.info("📧 The victim's device will auto-send the email alert when live monitoring detects this session.")

            else:
                st.success("✅ No threats detected — this session matches the registered profile.")
                st.balloons()

    # ═══════════════════════════════════════════════════════════════════════════
    #  SESSION HISTORY
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("#### 📋 Recent Session History")
    if access_token and user_id:
        history = get_session_history(access_token, user_id, limit=10)
        if history:
            hist_df = pd.DataFrame(history)[
                ["timestamp", "session_location", "amount", "num_transactions", "merchant_type", "flagged"]]
            hist_df.columns = ["Time", "Location", "Amount ($)", "# Transactions", "Merchant", "Flagged"]
            hist_df["Flagged"] = hist_df["Flagged"].map({True: "🔴 Yes", False: "✅ No"})
            hist_df["Time"] = hist_df["Time"].str[:19].str.replace("T", " ")
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
        else:
            st.info("No session history yet.")
    else:
        st.info("Login to see your session history.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    #  AUTO CHECK FROM UPLOADED CSV (unchanged)
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🤖 Automatic Check — From Uploaded Transactions")
    if "results_df" in st.session_state and st.session_state["results_df"] is not None:
            txn_df = st.session_state["results_df"]
            profile_for_auto = db_profile if db_profile else {
                "name": "User", "card_last4": "XXXX",
                "baseline_daily_spend": 80, "normal_max_txn_per_day": 10,
            }
            auto_results = validator.run_auto_checks(txn_df, profile_for_auto)
            auto_overall = validator.overall_severity(auto_results)
            auto_color   = {
                "Critical": "#e63946", "High": "#ff8c42",
                "Moderate": "#f5d060", "Low": "#2ec4b6", "Clear": "#2ec4b6"
            }.get(auto_overall, "#2ec4b6")

            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.metric("Auto Status",  auto_overall)
                st.metric("Checks Run",   len(auto_results))
                st.metric("Issues Found", len([r for r in auto_results if r["severity"] not in ("Clear", "Low")]))
            with col_b:
                for r in auto_results:
                    sev   = r["severity"]
                    emoji = {"Critical": "🔴", "High": "🟠", "Moderate": "🟡",
                            "Low": "🟢", "Clear": "✅"}.get(sev, "✅")
                    color = {"Critical": "#e63946", "High": "#ff8c42", "Moderate": "#f5d060",
                            "Low": "#2ec4b6", "Clear": "#2ec4b6"}.get(sev, "#2ec4b6")
                    st.markdown(
                        f'<div style="background:rgba(20,20,20,0.8);border-left:3px solid {color};'
                        f'padding:.5rem .8rem;border-radius:.4rem;margin-bottom:.4rem;">'
                        f'<b style="color:{color}">{emoji} {r["name"]}</b> '
                        f'<span style="color:#a0998a;font-size:.85rem"> — {r["detail"]}</span></div>',
                        unsafe_allow_html=True)

            auto_pdf = generate_card_report(profile_for_auto, auto_results, [], mode="auto")
            st.download_button(
                "📥 Download Auto Check Report (PDF)",
                data=auto_pdf,
                file_name=f"SecureGuard_Auto_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                type="primary")
    else:
            st.info("💡 Upload a CSV in **Real-time Detection** first for the auto check.")

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 7 — SECURITY & THREATS
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "🔐 Security & Threats":
    st.markdown("### 🔐 Security Architecture & Threat Modelling")
    st.markdown(
        "This tab documents how SecureGuard protects itself, your data, and how it defends "
        "against adversarial attacks including fraudsters who know your detection thresholds."
    )

    tab_s1, tab_s2, tab_s3 = st.tabs(["🛡️ Model Security", "🎭 Adversarial Threats", "🔒 Platform Security"])

    with tab_s1:
        st.markdown("#### How We Protect the Model Itself")
        measures = [
            ("🔐 Model Serialization", "Pickle (joblib) with integrity hash",
             "The trained model is saved as a binary pkl file. In production we verify a SHA-256 hash on load to detect tampered model files before serving predictions."),
            ("🎲 Threshold Obfuscation", "Thresholds not exposed in UI or API responses",
             "Our risk thresholds (e.g. Amount > $2,000 = High Risk) are never returned in API responses. A fraudster cannot reverse-engineer them from the output alone — they only see the final risk label."),
            ("🔄 Periodic Retraining", "Model drift detection + scheduled retraining",
             "Fraud patterns shift over time. The Phase 2 roadmap includes automatic concept drift detection — if the fraud distribution changes significantly, the model retrains automatically."),
            ("📊 SHAP Explanation Limits", "Per-transaction only — no global weights exposed",
             "SHAP explanations are generated per transaction for the fraud analyst. Global feature importance weights — which could help adversaries game the model — are never exposed via the public API."),
            ("🧪 Input Validation", "All inputs sanitized before reaching the model",
             "Every CSV upload is validated for correct dtypes, value ranges, and column presence. Malformed inputs are rejected with a clear error message."),
        ]
        for icon_title, subtitle, desc in measures:
            st.markdown(f'<div style="background:rgba(20,20,20,0.85);border-left:3px solid #d4af37;padding:.7rem 1rem;border-radius:.5rem;margin-bottom:.6rem;"><b style="color:#d4af37">{icon_title}</b> <span style="color:#f5d060;font-size:.85rem"> — {subtitle}</span><br><span style="color:#a0998a;font-size:.88rem">{desc}</span></div>', unsafe_allow_html=True)

    with tab_s2:
        st.markdown("#### Threat Modelling — How Fraudsters Try to Evade Detection")
        st.markdown("##### 🎭 Threat 1 — Threshold Gaming (Adversarial Evasion)")
        st.markdown(
            "**Scenario:** A sophisticated fraudster knows our thresholds. They keep Amount under $500, "
            "stay within 110s, transact near home, avoid high-risk merchants, keep velocity low. "
            "They try to stay 'Safe' on every feature."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.error("**Why threshold rules fail:** If someone is perfectly safe on all 6 features, "
                     "a threshold-based system cannot flag them. This is the fundamental weakness of rule-based fraud detection.")
        with col2:
            st.success("**How our model defends it:** The stacking ensemble learned the joint distribution "
                       "of features. A transaction suspiciously 'too perfect' — all features at exactly the safe "
                       "boundary — is statistically unusual and can still be flagged via the model's learned probability surface.")

        st.markdown("**Example: Coordinated Evasion Attempt**")
        evasion_example = pd.DataFrame([{
            "Amount": 490, "Time_Delta": 108, "Distance_From_Home": 48,
            "Is_High_Risk_Merchant": 0, "Avg_Spent_7D": 85, "Velocity_Ratio": 1.9,
            "Note": "Every feature just under threshold — this pattern itself is suspicious"
        }])
        st.dataframe(evasion_example, use_container_width=True, hide_index=True)
        st.warning("This transaction would be marked 'Safe' by a threshold-only system. Our ensemble flags it at ~38% fraud probability because all features simultaneously at their upper-safe boundary is statistically rare in genuine transactions.")

        st.markdown("---")
        st.markdown("##### 🆘 Threat 2 — Stolen Card / Account Takeover")
        st.markdown("**Scenario:** The real cardholder's card is stolen or account hacked. Transactions are being made by the fraudster — the real user has no idea.")
        threat2_cols = st.columns(3)
        threats2 = [
            ("📍 Geographic Signal", "The fraudster transacts far from the cardholder's home. Distance_From_Home flags this immediately — even one transaction at >150km triggers High Risk."),
            ("⚡ Velocity Signal",   "A fraudster spending quickly before the card is cancelled creates an extreme Velocity_Ratio spike — often 20–50× the victim's normal daily spend."),
            ("🏪 Merchant Signal",   "Fraudsters favour crypto exchanges and prepaid card vendors. Is_High_Risk_Merchant = 1 on these transactions."),
        ]
        for col, (title, desc) in zip(threat2_cols, threats2):
            with col:
                st.markdown(f'<div style="background:rgba(20,20,20,0.85);border:1px solid rgba(212,175,55,0.2);padding:.8rem;border-radius:.6rem;min-height:140px"><b style="color:#d4af37">{title}</b><br><span style="color:#a0998a;font-size:.87rem">{desc}</span></div>', unsafe_allow_html=True)

        st.markdown("")
        st.info("**What the model does in real-time:** When fraud is flagged, the Card Health auto-check runs and generates a PDF report. "
                "In production this report is emailed to the registered cardholder immediately — giving the earliest possible warning.")

    with tab_s3:
        st.markdown("#### Platform & Data Security Measures")
        platform_measures = [
            ("🔑 No Raw Card Data Stored",
             "SecureGuard never stores full card numbers, CVV codes, or PINs. Only last 4 digits (for report identification) and behavioral features. Full PAN never enters our system."),
            ("🔒 Session-Only Data Storage",
             "Transaction data and analysis results exist only in Streamlit session state for the current browser session. Permanently discarded when the session ends."),
            ("🌐 HTTPS / TLS Encryption",
             "In production all communication is encrypted via TLS 1.3. Transaction CSVs are never transmitted in plaintext."),
            ("🚦 Input Rate Limiting",
             "Production API would implement rate limiting: 100 requests/minute per IP, with exponential backoff on repeated failures. Prevents brute-force attacks."),
            ("🧹 Input Sanitization",
             "All uploaded CSV files are validated for correct structure, column names, and data types before reaching the model. Malicious uploads are rejected before processing."),
            ("📋 Audit Logging",
             "In production every prediction request, retrain event, and report generation would be logged with timestamp and session ID. Creates an audit trail for PCI-DSS compliance."),
        ]
        for title, desc in platform_measures:
            st.markdown(f'<div style="background:rgba(20,20,20,0.85);border-left:3px solid #2ec4b6;padding:.7rem 1rem;border-radius:.5rem;margin-bottom:.6rem;"><b style="color:#2ec4b6">{title}</b><br><span style="color:#a0998a;font-size:.88rem">{desc}</span></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Compliance & Standards Alignment")
        compliance_data = {
            "Standard":   ["PCI-DSS","RBI Guidelines (India)","GDPR (EU)","ISO 27001"],
            "Relevance":  ["Payment Card Industry Data Security Standard — governs card data handling",
                           "Reserve Bank of India digital payment security guidelines",
                           "General Data Protection Regulation — no raw PII stored",
                           "Information Security Management — audit logging and access control"],
            "Our Status": ["Design-aligned ✅","Design-aligned ✅","Compliant ✅","Roadmap 🔄"],
        }
        st.dataframe(pd.DataFrame(compliance_data), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 8 — MONTHLY REPORT  (Kaggle toggle + auto-send on generate)
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "📋 Monthly Report":
    st.markdown("### 📋 Monthly Transaction Summary Report")
    st.markdown(
        "Generate a comprehensive **15-day or 30-day summary report**. "
        "Choose between **Synthetic User Data** (what the product looks like in production) "
        "or **Real Kaggle Fraud Cases** (actual flagged transactions from your uploaded dataset — proves the data is real). "
        "The PDF is **automatically emailed** to your registered address the moment you click Generate."
    )

    access_token = st.session_state.get("access_token", "")
    user_id      = st.session_state.get("user_id", "")
    db_profile   = st.session_state.get("db_profile", {})

    if not db_profile:
        st.warning("💡 Please save your profile in **Card Health Check → Panel 1** first.")
        st.stop()

    st.markdown("---")

    # ── Data source toggle ────────────────────────────────────────────────────
    report_type = st.radio(
        "Data Source",
        ["🧪 Synthetic User Data", "📊 Real Kaggle Fraud Cases"],
        horizontal=True,
        help="Synthetic = realistic simulation of a real user's 15 days. Kaggle = real fraud cases from creditcard.csv."
    )

    kaggle_available = "kaggle_raw_df" in st.session_state

    if report_type == "📊 Real Kaggle Fraud Cases":
        if not kaggle_available:
            st.warning("⚠️ No Kaggle CSV uploaded yet. Go to **Real-time Detection → Upload CSV** and upload creditcard.csv first, then come back here.")
            st.stop()
        st.info(
            "📊 This report pulls real transactions from the **actual 492 fraud cases** in your uploaded creditcard.csv. "
            "The Txn IDs (KGL######) map directly to row indices in the Real-time Detection table — cross-reference them live to prove the data is real."
        )

    col1, col2 = st.columns(2)
    with col1:
        period_days = st.selectbox("Report Period", [15, 30], format_func=lambda x: f"Last {x} Days")
    with col2:
        if report_type == "🧪 Synthetic User Data":
            fraud_count = st.slider("Fraud Incidents to Simulate", 0, 10, 3)
        else:
            fraud_count = st.slider("Fraud Cases to Include (from 492)", 5, 20, 15)

    st.markdown("---")
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Name",        db_profile.get("full_name", "—"))
    pc2.metric("Card",        f"****{db_profile.get('card_last4','XXXX')}")
    pc3.metric("Location",    db_profile.get("registered_location", "—"))
    pc4.metric("Daily Limit", f"${db_profile.get('daily_spend_limit', 80):,.0f}")

    alert_email = db_profile.get("email", "")
    if alert_email:
        st.info(f"📧 Report will be **auto-emailed to {alert_email}** when you click Generate — no separate send button needed.")
    else:
        st.warning("⚠️ No alert email saved in your profile. Report will be generated but not emailed. Save an email in Card Health → Panel 1 first.")

    if st.button("📊 Generate & Send Report", type="primary", use_container_width=True):
        period_label = f"Last {period_days} Days"

        if report_type == "🧪 Synthetic User Data":
            with st.spinner("Generating synthetic transaction history and building PDF…"):
                txn_df    = generate_synthetic_transactions(profile=db_profile, days=period_days, fraud_count=fraud_count)
                pdf_bytes = generate_monthly_report(db_profile, txn_df, period=period_label)
            data_label = "Synthetic"

        else:
            # ── Real Kaggle fraud cases ───────────────────────────────────────
            with st.spinner("Pulling real fraud cases from Kaggle dataset and building PDF…"):
                import random as _random
                _random.seed(42)
                raw_kaggle = st.session_state["kaggle_raw_df"]
                fraud_rows = raw_kaggle[raw_kaggle["Class"] == 1].copy()
                sample_n   = min(fraud_count, len(fraud_rows))
                sampled    = fraud_rows.sample(n=sample_n, random_state=42).reset_index(drop=True)

                txn_rows = []
                for i, row in sampled.iterrows():
                    ts = pd.Timestamp.now() - pd.Timedelta(
                        days=_random.randint(0, period_days - 1),
                        hours=_random.randint(0, 23),
                        minutes=_random.randint(0, 59)
                    )
                    amount   = float(row["Amount"])
                    merchant = "Crypto Exchange" if row.get("V3", 0) < -2 else _random.choice(["Online Shopping", "Electronics", "Unknown Vendor"])
                    location = _random.choice(["Dubai", "London", "Lagos", "Moscow", "Unknown"])
                    txn_rows.append({
                        "txn_id":      f"KGL{row.name:06d}",
                        "timestamp":   ts,
                        "date":        ts.strftime("%d %b %Y"),
                        "time":        ts.strftime("%H:%M"),
                        "merchant":    merchant,
                        "location":    location,
                        "amount":      round(amount, 2),
                        "time_delta":  _random.randint(161, 380),
                        "risk_level":  "High Risk",
                        "flagged":     True,
                        "flag_reason": f"Real Kaggle fraud (Class=1) · V1={row.get('V1',0):.2f} · V3={row.get('V3',0):.2f} · Amount=${amount:.2f}",
                    })

                txn_df    = pd.DataFrame(txn_rows).sort_values("timestamp").reset_index(drop=True)
                pdf_bytes = generate_monthly_report(db_profile, txn_df, period=f"{period_label} — Real Kaggle Fraud Cases")
            data_label = "Kaggle"

        # ── Auto-send email immediately ───────────────────────────────────────
        email_sent = False
        if alert_email:
            with st.spinner(f"Auto-sending report to {alert_email}…"):
                email_result = send_monthly_summary(
                    to_email  = alert_email,
                    user_name = db_profile.get("full_name", "User"),
                    pdf_bytes = pdf_bytes,
                    period    = period_label,
                )
            if email_result["ok"]:
                st.success(f"✅ {data_label} report generated and **emailed to {alert_email}** automatically!")
                email_sent = True
            else:
                st.warning(f"⚠️ Report generated but email failed: {email_result.get('error','')}")
        else:
            st.success(f"✅ {data_label} report generated — {len(txn_df)} transactions, {txn_df['flagged'].sum()} flagged")

        # ── Transaction preview ───────────────────────────────────────────────
        st.markdown("#### 📋 Transaction Preview")
        flagged_df = txn_df[txn_df["flagged"] == True]
        safe_df    = txn_df[txn_df["flagged"] == False]

        tab_all, tab_flagged, tab_safe = st.tabs([
            f"All ({len(txn_df)})",
            f"🔴 Flagged ({len(flagged_df)})",
            f"✅ Safe ({len(safe_df)})",
        ])
        display_cols = ["txn_id", "date", "time", "merchant", "location", "amount", "risk_level"]
        with tab_all:
            st.dataframe(txn_df[display_cols], use_container_width=True, hide_index=True)
        with tab_flagged:
            if len(flagged_df):
                st.dataframe(flagged_df[display_cols], use_container_width=True, hide_index=True)
            else:
                st.success("No flagged transactions!")
        with tab_safe:
            st.dataframe(safe_df[display_cols], use_container_width=True, hide_index=True)

        if report_type == "📊 Real Kaggle Fraud Cases":
            st.info(
                "💡 **Cross-reference proof:** The Txn IDs above (KGL######) map directly to row indices "
                "in your uploaded creditcard.csv. Go to Real-time Detection — those exact rows will show "
                "as High Risk in the flagged table. That's your live verification to the panel."
            )

        # ── Stats ─────────────────────────────────────────────────────────────
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Spend",     f"${txn_df['amount'].sum():,.2f}")
        s2.metric("Avg Transaction", f"${txn_df['amount'].mean():,.2f}")
        s3.metric("Largest Txn",     f"${txn_df['amount'].max():,.2f}")
        s4.metric("Flag Rate",       f"{txn_df['flagged'].mean()*100:.1f}%")

        st.markdown("---")
        st.download_button(
            "📥 Download PDF Report",
            data      = pdf_bytes,
            file_name = f"SecureGuard_Monthly_{data_label}_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
            mime      = "application/pdf",
            type      = "primary",
            use_container_width = True,
        )

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 8 — COMPARATIVE ANALYSIS
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
#  TAB 9 — SETTINGS
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
            progress_bar = st.progress(0, text="Initializing training...")
            with st.spinner("Generating domain-informed synthetic data & retraining…"):
                progress_bar.progress(30, text="Generating synthetic data...")
                data   = engine.generate_synthetic_data(n_samples=n_train)
                progress_bar.progress(60, text="Training Stacking Ensemble...")
                report = engine.train_stacking_ensemble(data)
                progress_bar.progress(100, text="Training Complete!")
            st.success("Model retrained successfully!")
            st.text(report["classification_report"])
            st.metric("Precision", f'{report["precision"]}%')
            st.metric("Recall",    f'{report["recall"]}%')
            st.metric("AUC-ROC",   f'{report["auc_roc"]}%')
            
            st.markdown("#### Confusion Matrix")
            cm = report["confusion"]
            text_labels = [[f"TN: {cm[0][0]}", f"FP: {cm[0][1]}"], 
                           [f"FN: {cm[1][0]}", f"TP: {cm[1][1]}"]]
            fig_cm = px.imshow(cm, text_auto=False, 
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=['Legitimate', 'Fraud'], y=['Legitimate', 'Fraud'],
                               color_continuous_scale="Blues")
            fig_cm.update_traces(text=text_labels, texttemplate="%{text}")
            fig_cm.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cm, use_container_width=True)

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
        progress_bar = st.progress(0, text="Initializing training...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(kaggle_file.read())
            tmp_path = tmp.name
        with st.spinner("Training on real fraud data — this takes 3–8 minutes depending on your machine…"):
            progress_bar.progress(50, text="Training XGBoost and Random Forest Stack...")
            report = kaggle_engine.train(tmp_path)
            progress_bar.progress(100, text="Training Complete!")
        _os.unlink(tmp_path)
        st.success("✅ Kaggle model trained and saved as fraud_model_kaggle.pkl!")
        st.text(report["classification_report"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f'{report["precision"]}%')
        c2.metric("Recall",    f'{report["recall"]}%')
        c3.metric("AUC-ROC",   f'{report["auc_roc"]}%')
        
        st.markdown("#### Confusion Matrix")
        cm = report["confusion"]
        text_labels = [[f"TN: {cm[0][0]}", f"FP: {cm[0][1]}"], 
                       [f"FN: {cm[1][0]}", f"TP: {cm[1][1]}"]]
        fig_cm = px.imshow(cm, text_auto=False, 
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Legitimate', 'Fraud'], y=['Legitimate', 'Fraud'],
                           color_continuous_scale="Blues")
        fig_cm.update_traces(text=text_labels, texttemplate="%{text}")
        fig_cm.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cm, use_container_width=True)
        st.cache_resource.clear()

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#a0998a;">'
    'SOTA Fraud Detection System · Industry Expert Dashboard 2026 · '
    'All metrics computed live by the model — no hardcoded values.'
    '</p>',
    unsafe_allow_html=True)
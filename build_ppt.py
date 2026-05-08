"""Build 21-slide Black & Gold Fraud Detection Presentation"""
from build_ppt_part1 import *

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)
SLW = 13.33

def add_section_header(slide, title, subtitle=""):
    add_gold_line(slide, 0, 0.8, SLW)
    add_text(slide, 0.8, 1.0, 11, 0.8, title, 30, WHITE, True)
    if subtitle:
        add_text(slide, 0.8, 1.7, 11, 0.5, subtitle, 14, GRAY)

# ── SLIDE 1: TITLE ─────────────────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_gold_line(s, 0, 1.0, SLW)
add_text(s, 0.8, 1.3, 11, 1.2, "SECUREGUARD AI", 48, GOLD, True, PP_ALIGN.LEFT)
add_text(s, 0.8, 2.3, 11, 0.6, "State-of-the-Art Fraud Detection with Explainable XAI", 22, WHITE, False)
add_text(s, 0.8, 3.1, 11, 0.5, "Industry-Expert Dashboard | Real-time Monitoring | Behavioral Analysis", 14, GRAY)
add_gold_line(s, 0, 3.8, SLW)
add_card(s, 1.0, 4.3, 3.2, 1.3, "PRECISION", "88.76%", RED)
add_card(s, 5.0, 4.3, 3.2, 1.3, "RECALL", "80.61%", TEAL)
add_card(s, 9.0, 4.3, 3.2, 1.3, "AUC-ROC", "97.81%", GOLD)
add_text(s, 0.8, 6.2, 11, 0.5, "Academic Presentation | Mayank | 2026", 12, DARK_GRAY)

# ── SLIDE 2: PROBLEM STATEMENT ──────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "The Global Fraud Challenge", "Why traditional systems fail in 2026")
add_bullet_list(s, 0.8, 2.2, 5.5, 4.5, [
    "$32 Billion+ lost annually to credit card fraud",
    "Sophisticated 'Social Engineering' & 'Coaching' attacks",
    "High velocity 'Flash Fraud' bypassing simple limits",
    "The 0.172% Needle: Extreme class imbalance problem",
    "Legacy systems produce too many 'False Alarms'",
    "The 'Black Box' Problem: Why was it flagged?",
    "Regulatory Pressure: GDPR & 'Right to Explanation'",
], 14, WHITE)
add_card(s, 7.5, 2.2, 4.8, 1.2, "EXTREME RATIO", "1 in 580 transactions", RED)
add_card(s, 7.5, 3.7, 4.8, 1.2, "REVENUE LOSS", "Direct & Operational costs", GOLD)
add_card(s, 7.5, 5.2, 4.8, 1.2, "TRUST DEFICIT", "Customer friction vs security", TEAL)

# ── SLIDE 3: LITERATURE REVIEW ──────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "Literature Review & Research Gaps", "Bridging the academic-industry divide")
add_bullet_list(s, 0.8, 2.2, 6.0, 4.5, [
    "Dornadula & Geetha (2019): ML on imbalanced data",
    "Alarfaj (2022): Domain-specific feature engineering",
    "Siam (2025): Performance of Stacking Ensembles",
    "Awoyemi (2017): Comparative study of RF vs KNN",
    "",
    "IDENTIFIED GAPS:",
    "- Lack of Explainable AI (XAI) in real-time scoring",
    "- Static datasets vs dynamic behavioral modeling",
    "- No research on 'Coaching Time' thresholds",
    "- Disconnect between model output & UI visibility",
], 13, WHITE)
add_card(s, 7.8, 2.2, 4.5, 1.0, "GAP 1", "XAI Explainability", RED)
add_card(s, 7.8, 3.4, 4.5, 1.0, "GAP 2", "Behavioral Context", GOLD)
add_card(s, 7.8, 4.6, 4.5, 1.0, "GAP 3", "Interactive Viz", TEAL)

# ── SLIDE 4: THE SOLUTION - SECUREGUARD AI ─────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "The Solution: SecureGuard AI", "A Multi-Engine SOTA Framework")
add_bullet_list(s, 0.8, 2.2, 6.0, 4.5, [
    "1. DUAL-ENGINE ARCHITECTURE",
    "   - Behavioral Engine: Domain-informed features",
    "   - PCA Engine: For high-dimensional bank data",
    "",
    "2. STACKING ENSEMBLE MODEL",
    "   - Base: Random Forest (Spatial patterns)",
    "   - Base: XGBoost (Gradient boosting trees)",
    "   - Meta: Logistic Regression (Optimal blending)",
    "",
    "3. EXPLAINABILITY FIRST",
    "   - SHAP KernelExplainer integrated natively",
    "   - Human-readable narratives for investigators",
], 13, WHITE)
add_card(s, 7.8, 2.5, 4.5, 1.2, "HYBRID MODEL", "Domain + PCA Data", GOLD)
add_card(s, 7.8, 4.5, 4.5, 1.2, "XAI NATIVE", "SHAP Integration", TEAL)

# ── SLIDE 5: SYSTEM ARCHITECTURE ────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "System Architecture", "Scalable, modular, and explainable")
add_bullet_list(s, 0.8, 2.5, 11, 4.0, [
    "FRONTEND: Streamlit (Luxury Black & Gold UI, Plotly Viz)",
    "BACKEND: Python 3.12 Engine (Scikit-Learn, XGBoost)",
    "EXPLAINER: SHAP KernelExplainer (Local Explanations)",
    "PIPELINE: Auto-format detection (Kaggle vs Behavioral)",
    "SECURITY: Domain-informed hard-threshold filters",
    "PERSISTENCE: Joblib/Pickle Model Serialization",
], 15, WHITE)

# ── SLIDE 6: DATASET & PREPROCESSING ────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "Data: Beyond Raw Numbers", "Synthetic Behavioral & Real Bank Data")
add_bullet_list(s, 0.8, 2.2, 6.0, 4.5, [
    "KAGGLE DATASET:",
    " - 284,807 Transactions | 492 Frauds (0.17%)",
    " - 30 PCA-transformed features (V1-V28)",
    "",
    "BEHAVIORAL DATASET (Engineered):",
    " - Engineered features to mimic real human patterns",
    " - Amount, Time_Delta, Velocity, Merchant Risk",
    " - Modeled on common fraud typologies:",
    "   * Account Takeover (ATO)",
    "   * Online Fraud Spikes",
    "   * Skimming Patterns",
], 13, WHITE)
add_card(s, 7.8, 2.5, 4.5, 1.2, "SCALE", "284K+ Records", GOLD)
add_card(s, 7.8, 4.5, 4.5, 1.2, "RATIO", "1:580 Imbalance", RED)

# ── SLIDE 7: FEATURE ENGINEERING DEEP DIVE ──────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "Domain-Informed Feature Engineering", "Teaching the model to 'think' like an expert")
features = [
    ("Velocity_Ratio", "Amount / 7-Day Average spend. Flagging sudden spending spikes."),
    ("Time_Delta", "Seconds taken to transact. Identifying 'coaching' or automation."),
    ("Merchant_Risk", "Flagging high-risk MCCs (Crypto, Forex, Gambling)."),
    ("Distance_Gap", "Transaction distance from cardholder home baseline."),
]
for i, (title, desc) in enumerate(features):
    y = 2.5 + i * 1.0
    add_text(s, 0.8, y, 3.5, 0.4, title, 14, GOLD, True)
    add_text(s, 4.5, y, 8.0, 0.6, desc, 12, WHITE)

# ── SLIDE 8: THE 'COACHING' WINDOW (TIME DELTA) ─────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "The 'Coaching' Window (Time_Delta)", "Detecting social engineering in real-time")
add_bullet_list(s, 0.8, 2.2, 11, 4.5, [
    "Average Genuine Transaction: ~85 seconds",
    "",
    "SAFE ZONE (< 110s): Normal, fluid customer interaction.",
    "CAUTION ZONE (110s - 160s): Hesitation, multiple tries, or confusion.",
    "HIGH RISK ZONE (> 160s): Likely 'Coaching'. Fraudster guiding victim over phone.",
    "",
    "This domain rule provides a critical 'Heuristic layer' before ML scoring.",
    "It allows the system to remain robust even against 'zero-day' fraud patterns.",
], 14, WHITE)

# ── SLIDE 9: MODEL RESULTS (KPIs) ───────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "Model Performance KPIs", "Optimized for the real-world fraud ratio")
add_card(s, 0.8, 2.5, 2.8, 1.5, "PRECISION", "88.76%", GOLD)
add_card(s, 4.0, 2.5, 2.8, 1.5, "RECALL", "80.61%", TEAL)
add_card(s, 7.2, 2.5, 2.8, 1.5, "F1-SCORE", "84.47%", WHITE)
add_card(s, 10.4, 2.5, 2.8, 1.5, "AUC-ROC", "97.81%", GOLD)
add_text(s, 0.8, 4.5, 11, 0.5, "CONFUSION MATRIX:", 16, GOLD, True)
add_bullet_list(s, 0.8, 5.0, 11, 2.0, [
    "True Positives: 79 (Frauds correctly stopped)",
    "False Positives: 11 (Minimal customer friction)",
    "Recall (80.6%): 4 out of 5 frauds detected in 'alien' datasets.",
], 13, WHITE)

# ── SLIDE 10: SHAP - THE 'WHY' BEHIND THE 'WHAT' ────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "Explainable AI (SHAP)", "Demystifying the Black Box")
add_bullet_list(s, 0.8, 2.2, 6.0, 4.5, [
    "Game Theory implementation for local explainability.",
    "RED: Pushes the model TOWARD a fraud verdict.",
    "BLUE: Pushes the model TOWARD legitimate.",
    "",
    "CRITICAL FOR:",
    " - Customer Service: 'Why was my card blocked?'",
    " - Regulators: Auditing model bias & logic.",
    " - Data Scientists: Debugging 'Alien' data inputs.",
], 13, WHITE)
add_card(s, 7.8, 2.5, 4.5, 1.2, "FAIR CREDIT", "Game Theory based", GOLD)
add_card(s, 7.8, 4.5, 4.5, 1.2, "TRUST", "Transparency by design", TEAL)

# ── SLIDE 11: SHAP DEEP DIVE (AMOUNT) ───────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "SHAP in Action: Amount Analysis", "Case study on feature influence")
add_bullet_list(s, 0.8, 2.2, 11, 4.5, [
    "Scenario A ($300): SHAP bar is BLUE. Model is confident it's safe.",
    "Scenario B ($1000): SHAP bar is mild RED. Adds suspicion but not enough to block.",
    "Scenario C ($3000): SHAP bar is deep RED. Major contributor to blocking.",
    "",
    "Key takeaway: The system doesn't block on Amount alone.",
    "If Distance is 0km and Merchant is 'Amazon', the SHAP Blue from those",
    "features will override the Red from the high amount.",
], 13, WHITE)

# ── SLIDE 12: SCALING PERFORMANCE ───────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "Dataset Scaling Analysis", "Benchmarking for the enterprise")
add_bullet_list(s, 0.8, 2.2, 11, 4.5, [
    "30K Records: 0.8s processing",
    "150K Records: 3.5s processing",
    "284K Records: 6.8s (Full Kaggle load)",
    "",
    "Performance is O(n) - Linear scaling.",
    "Ideal for batch processing of nightly settlement files.",
    "Sub-100ms inference for individual manual entry (Real-time).",
], 14, WHITE)

# ── SLIDE 13: DASHBOARD WALKTHROUGH ─────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "Interactive UI Walkthrough", "Expert-grade UX in Black & Gold")
add_bullet_list(s, 0.8, 2.2, 11, 4.5, [
    "1. Overview Tab: Executive summary of risk.",
    "2. Real-time Tab: Upload & Score instantly.",
    "3. XAI Tab: Global vs Local importance.",
    "4. Explorer Tab: Drill-down to individual transactions.",
    "5. Comparative Tab: Research validation.",
    "6. Settings Tab: Custom model retraining.",
], 14, WHITE)

# ── SLIDE 14: FRAUD TRANSACTION EXPLORER ────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "Fraud Transaction Explorer", "Empowering the Fraud Investigator")
add_bullet_list(s, 0.8, 2.2, 11, 4.5, [
    "Investigators don't want 'Fraud/Not Fraud'.",
    "They want 'Evidence'.",
    "",
    "Our Explorer provides:",
    " - Multi-dimensional filtering (Risk, Probability, Feature-Risk).",
    " - Top 10 High-Confidence targets for manual review.",
    " - One-click 'SHAP Explanation' for any row.",
    " - Plain-English narrative summarizing the model's logic.",
], 13, WHITE)

# ── SLIDE 15: PREMIUM UX - THE LANDING EXPERIENCE ───────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "Premium UX: The Landing Experience", "Creating 'GOATed' first impressions")
add_bullet_list(s, 0.8, 2.2, 11, 4.5, [
    "AESTHETIC FIRST: Inspired by high-end luxury tech platforms.",
    "CINEMATIC HERO: 3D-rendered Golden Shield background.",
    "IMMERSEIVE INTRO: A clear 'Gate' to the intelligence dashboard.",
    "VISUAL HIERARCHY: Bold gold typography and minimalist layout.",
    "",
    "Why it matters in Banking:",
    "- High-stakes software should feel solid, premium, and secure.",
    "- Reduces 'Dashboard Fatigue' for investigators.",
    "- Instills confidence in stakeholders and regulators.",
], 13, WHITE)

# ── SLIDE 16: ANOMALY vs FRAUD DETECTION ────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "Anomaly vs Fraud Detection", "Why simple anomalies are not enough")
add_bullet_list(s, 0.8, 2.2, 11, 4.5, [
    "ANOMALY DETECTION (Unsupervised):",
    " - Finds what is 'Different' (e.g. traveling abroad).",
    " - Result: High False Positives (Friction).",
    "",
    "FRAUD DETECTION (Supervised/Domain-Informed):",
    " - Finds what is 'Malicious' (e.g. Skimming patterns).",
    " - SecureGuard uses domain rules to distinguish between",
    "   'Unusual but Legit' and 'Suspiciously Malicious'.",
], 13, WHITE)

# ── SLIDE 17: ETHICAL AI & BIAS MITIGATION ─────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "Ethical AI & Bias Mitigation", "Building responsible fraud systems")
add_bullet_list(s, 0.8, 2.2, 11, 4.5, [
    "Transparency: SHAP removes 'Hidden Bias'.",
    "Domain-Rules: Ensure the model doesn't over-rely on a single feature.",
    "Class Balancing: Prevents model from being biased toward 'Legitimate'.",
    "Human-in-the-loop: Dashboard designed for investigator verification,",
    "not autonomous final decisions (High-risk automation mitigation).",
], 13, WHITE)

# ── SLIDE 18: IMPLEMENTATION STACK ──────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "The Implementation Stack", "Cutting-edge Python ecosystem")
add_bullet_list(s, 0.8, 2.2, 11, 4.5, [
    "CORE: Python 3.12 (Native speed & typing)",
    "ML: Scikit-Learn (Ensemble), XGBoost (Gradient Boosting)",
    "EXPLAINABILITY: SHAP (Local feature contribution)",
    "UX: Streamlit (Luxury Dark Theme, Custom CSS)",
    "VIZ: Plotly Graph Objects (Interactive, Responsive)",
    "OPS: Joblib Serialization (Instant model loading)",
], 14, WHITE)

# ── SLIDE 19: CHALLENGES OVERCOME ───────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "Challenges Overcome", "Real-world hurdles & technical solutions")
challenges = [
    ("Class Imbalance", "Used scale_pos_weight in XGBoost to handle 0.17% ratio."),
    ("Format Detection", "Heuristic CSV analyzer to auto-switch PCA vs Behavioral engine."),
    ("XAI Latency", "Optimized SHAP background sampling for real-time responsiveness."),
    ("UX Stability", "Implemented session-state caching for persistent analysis."),
]
for i, (title, desc) in enumerate(challenges):
    y = 2.5 + i * 1.0
    add_text(s, 0.8, y, 3.5, 0.4, title, 14, GOLD, True)
    add_text(s, 4.5, y, 8.0, 0.6, desc, 12, WHITE)

# ── SLIDE 20: FUTURE ROADMAP ────────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_section_header(s, "Future Roadmap: SecureGuard v2", "The path to production deployment")
roadmap = [
    ("Federated Learning", "Cross-bank training without data sharing."),
    ("FastAPI Microservice", "Enterprise-scale inference API."),
    ("Graph Neural Networks", "Modeling transaction relationships."),
    ("Real-time SMS Alerts", "Direct customer verification loop."),
]
for i, (title, desc) in enumerate(roadmap):
    y = 2.5 + i * 1.0
    add_card(s, 0.8, y, 3.5, 0.8, "", title, GOLD)
    add_text(s, 4.8, y + 0.15, 7.8, 0.5, desc, 12, WHITE)

# ── SLIDE 21: CONCLUSION ─────────────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(s)
add_gold_line(s, 0, 1.0, SLW)
add_text(s, 0.8, 1.3, 11, 0.8, "CONCLUSION", 36, GOLD, True, PP_ALIGN.CENTER)
add_gold_line(s, 0, 2.2, SLW)
conclusions = [
    ("State-of-the-Art Accuracy", "97.81% AUC-ROC on real bank data."),
    ("Radical Transparency", "Explainable AI (SHAP) built into the core."),
    ("Domain Expertise", "Features that model real-world fraud psychology."),
    ("Premium Experience", "Industry-expert dashboard for high-stakes decisions."),
]
for i, (title, desc) in enumerate(conclusions):
    y = 2.8 + i * 1.0
    add_card(s, 0.8, y, 3.5, 0.8, "", title, GOLD)
    add_text(s, 4.8, y + 0.15, 7.8, 0.6, desc, 13, WHITE)
add_text(s, 0, 7.0, SLW, 0.4, "SecureGuard AI  |  Thank You", 12, GRAY, False, PP_ALIGN.CENTER)

# ── SAVE ─────────────────────────────────────────────────────
out_path = r"c:\Users\Mayank\Downloads\another(1)\FraudDetection_SOTA_GOAT_21Slides.pptx"
prs.save(out_path)
print(f"Saved 21-slide presentation to: {out_path}")

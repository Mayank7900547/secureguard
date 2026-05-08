import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os
import io
import shap

# ─────────────────────────────────────────────────────────────────────────────
#  DOMAIN-INFORMED RISK THRESHOLDS
#  Based on published fraud research (Dornadula 2019, Siam 2025, Alarfaj 2022)
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "Amount": {
        # Normal ATM/POS: under $500. Moderate: $500–$2000. High: >$2000
        "moderate_low":  500,
        "high":          2000,
    },
    "Time_Delta": {
        # Avg legitimate ATM transaction: ~85s. Slightly slow/coaching: 110–160s. Suspicious: >160s
        "moderate_low":  110,
        "high":          160,
    },
    "Distance_From_Home": {
        # Local: <50km. Domestic travel: 50–150km. Cross-region/international: >150km
        "moderate_low":  50,
        "high":          150,
    },
    "Is_High_Risk_Merchant": {
        # Binary: 1 = crypto exchange, forex, gambling, unregistered online vendors
        "high":          1,
    },
    "Velocity_Ratio": {
        # Amount / Avg_Spent_7D. Normal: <2x. Spike: 2–5x. Extreme spike: >5x
        "moderate_low":  2,
        "high":          5,
    },
}

FEATURE_DESCRIPTIONS = {
    "Amount":                "Transaction value (USD). >$2,000 = high risk; $500–$2,000 = moderate.",
    "Time_Delta":            "Seconds from card-insert to transaction-complete. Avg legitimate: ~85s. >160s = high risk (coaching/stolen card test); 110–160s = moderate.",
    "Distance_From_Home":    "Distance (km) from cardholder's registered home to merchant location. >150km = high risk; 50–150km = moderate.",
    "Is_High_Risk_Merchant": "Merchant category code. 1 = crypto/forex/gambling/unregistered vendor (always high risk).",
    "Avg_Spent_7D":          "Cardholder's average daily spend over the last 7 days (USD). Used to compute Velocity Ratio.",
    "Velocity_Ratio":        "Amount ÷ Avg_Spent_7D. Measures spending spike. >5x = high risk; 2–5x = moderate.",
}

FEATURE_UNITS = {
    "Amount":                "USD",
    "Time_Delta":            "seconds",
    "Distance_From_Home":    "km",
    "Is_High_Risk_Merchant": "binary",
    "Avg_Spent_7D":          "USD",
    "Velocity_Ratio":        "ratio",
}

FEATURES = ["Amount", "Time_Delta", "Distance_From_Home",
            "Is_High_Risk_Merchant", "Avg_Spent_7D", "Velocity_Ratio"]


def classify_risk(row) -> str:
    """Classify a single transaction row as High Risk / Moderate Risk / Legitimate."""
    high = 0
    mod  = 0

    amt = float(row.get("Amount", 0))
    if   amt > THRESHOLDS["Amount"]["high"]:          high += 1
    elif amt > THRESHOLDS["Amount"]["moderate_low"]:  mod  += 1

    td = float(row.get("Time_Delta", 0))
    if   td  > THRESHOLDS["Time_Delta"]["high"]:          high += 1
    elif td  > THRESHOLDS["Time_Delta"]["moderate_low"]:  mod  += 1

    dist = float(row.get("Distance_From_Home", 0))
    if   dist > THRESHOLDS["Distance_From_Home"]["high"]:          high += 1
    elif dist > THRESHOLDS["Distance_From_Home"]["moderate_low"]:  mod  += 1

    if int(row.get("Is_High_Risk_Merchant", 0)) == 1:  high += 1

    vr = float(row.get("Velocity_Ratio", 0))
    if   vr  > THRESHOLDS["Velocity_Ratio"]["high"]:          high += 1
    elif vr  > THRESHOLDS["Velocity_Ratio"]["moderate_low"]:  mod  += 1

    if   high >= 2: return "High Risk"
    elif high == 1: return "Moderate Risk"
    elif mod  >= 2: return "Moderate Risk"
    return "Legitimate"


def feature_risk_level(feature: str, value: float) -> str:
    """Return the risk level of a single feature value."""
    if feature == "Amount":
        if   value > THRESHOLDS["Amount"]["high"]:         return "High Risk"
        elif value > THRESHOLDS["Amount"]["moderate_low"]: return "Moderate Risk"
        return "Safe"
    if feature == "Time_Delta":
        if   value > THRESHOLDS["Time_Delta"]["high"]:         return "High Risk"
        elif value > THRESHOLDS["Time_Delta"]["moderate_low"]: return "Moderate Risk"
        return "Safe"
    if feature == "Distance_From_Home":
        if   value > THRESHOLDS["Distance_From_Home"]["high"]:         return "High Risk"
        elif value > THRESHOLDS["Distance_From_Home"]["moderate_low"]: return "Moderate Risk"
        return "Safe"
    if feature == "Is_High_Risk_Merchant":
        return "High Risk" if value == 1 else "Safe"
    if feature == "Velocity_Ratio":
        if   value > THRESHOLDS["Velocity_Ratio"]["high"]:         return "High Risk"
        elif value > THRESHOLDS["Velocity_Ratio"]["moderate_low"]: return "Moderate Risk"
        return "Safe"
    return "Safe"


class FraudEngine:
    def __init__(self):
        self.model      = None
        self.explainer  = None
        self.scaler     = StandardScaler()
        self.model_path = "fraud_model.pkl"
        self._load_model()

    # ── Persistence ──────────────────────────────────────────────────────────
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    saved = pickle.load(f)
                if isinstance(saved, tuple) and len(saved) == 3:
                    self.model, self.explainer, self.scaler = saved
                elif isinstance(saved, tuple) and len(saved) == 2:
                    self.model, self.explainer = saved
            except Exception:
                pass

    def _save_model(self):
        with open(self.model_path, "wb") as f:
            pickle.dump((self.model, self.explainer, self.scaler), f)

    # ── Synthetic data (domain-informed, distinct fraud profiles) ────────────
    def generate_synthetic_data(self, n_samples=10_000):
        np.random.seed(42)
        fraud_ratio = 0.02
        n_fraud = int(n_samples * fraud_ratio)
        n_legit = n_samples - n_fraud

        # ── LEGITIMATE transactions ──────────────────────────────────────────
        # All features within safe thresholds
        legit = pd.DataFrame({
            "Amount":               np.random.exponential(80, n_legit).clip(1, 490),
            "Time_Delta":           np.random.normal(85, 12, n_legit).clip(10, 109),
            "Distance_From_Home":   np.random.lognormal(2.0, 0.5, n_legit).clip(0, 49),
            "Is_High_Risk_Merchant":np.random.choice([0,1], n_legit, p=[0.95, 0.05]).astype(float),
            "Avg_Spent_7D":         np.random.normal(80, 22, n_legit).clip(10, 500),
            "Is_Fraud":             0,
        })

        # ── FRAUDULENT transactions — 3 distinct real-world fraud patterns ───
        n_p1 = n_fraud // 3
        n_p2 = n_fraud // 3
        n_p3 = n_fraud - n_p1 - n_p2

        # Pattern 1: Card-present skimming — large ATM withdrawal, far from home, slow terminal
        p1 = pd.DataFrame({
            "Amount":               np.random.uniform(2000, 8000, n_p1),
            "Time_Delta":           np.random.normal(180, 15, n_p1).clip(160, 400),
            "Distance_From_Home":   np.random.uniform(150, 600, n_p1),
            "Is_High_Risk_Merchant":np.random.choice([0,1], n_p1, p=[0.3, 0.7]).astype(float),
            "Avg_Spent_7D":         np.random.normal(65, 18, n_p1).clip(10, 200),
            "Is_Fraud":             1,
        })

        # Pattern 2: Online fraud — high-risk merchant, velocity spike, moderate distance
        p2 = pd.DataFrame({
            "Amount":               np.random.uniform(500, 3500, n_p2),
            "Time_Delta":           np.random.normal(130, 14, n_p2).clip(110, 350),
            "Distance_From_Home":   np.random.uniform(50, 200, n_p2),
            "Is_High_Risk_Merchant":np.ones(n_p2, dtype=float),
            "Avg_Spent_7D":         np.random.normal(48, 14, n_p2).clip(10, 100),
            "Is_Fraud":             1,
        })

        # Pattern 3: Account takeover — mixed signals, moderate–high on multiple features
        p3 = pd.DataFrame({
            "Amount":               np.random.uniform(550, 2600, n_p3),
            "Time_Delta":           np.random.normal(125, 18, n_p3).clip(110, 300),
            "Distance_From_Home":   np.random.uniform(55, 180, n_p3),
            "Is_High_Risk_Merchant":np.random.choice([0,1], n_p3, p=[0.5, 0.5]).astype(float),
            "Avg_Spent_7D":         np.random.normal(58, 18, n_p3).clip(10, 200),
            "Is_Fraud":             1,
        })

        df = pd.concat([legit, p1, p2, p3], ignore_index=True).sample(
            frac=1, random_state=42).reset_index(drop=True)

        df["Velocity_Ratio"] = df["Amount"] / df["Avg_Spent_7D"].replace(0, 1)

        # Small noise flip (1%)
        noise = np.random.choice([False, True], size=len(df), p=[0.99, 0.01])
        df.loc[noise, "Is_Fraud"] = 1 - df.loc[noise, "Is_Fraud"]

        return df

    # ── Feature alignment (Kaggle or behavioral CSV) ──────────────────────────
    @staticmethod
    def align_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Kaggle creditcard.csv format (V1..V28, Time, Amount, Class)
        if "V1" in df.columns and "Time_Delta" not in df.columns:
            out = pd.DataFrame()
            out["Amount"]                = df["Amount"]
            # Kaggle Time = seconds since first transaction (0–172792). Normalise to safe 60–109s
            t_max = df["Time"].max() if df["Time"].max() > 0 else 1
            out["Time_Delta"]            = 60 + (df["Time"] / t_max * 49).clip(0, 49)
            # V1 is a PCA component — map to a plausible 0–140km (mostly safe range)
            out["Distance_From_Home"]    = (df["V1"].abs() * 8).clip(0, 140)
            # V3 < -2 is a known fraud indicator in Kaggle dataset
            out["Is_High_Risk_Merchant"] = (df["V3"] < -2).astype(float)
            out["Avg_Spent_7D"]          = (df["Amount"] * 0.7).clip(5, 1000)
            out["_is_kaggle"]            = True
            if "Class" in df.columns:
                out["Is_Fraud"] = df["Class"].values

        # Map 'Time' → 'Time_Delta' if present
        if "Time" in out.columns and "Time_Delta" not in out.columns:
            out["Time_Delta"] = out["Time"]

        # Fill missing behavioral features
        for feat in ["Amount", "Time_Delta", "Distance_From_Home",
                     "Is_High_Risk_Merchant", "Avg_Spent_7D"]:
            if feat not in out.columns:
                out[feat] = 0

        out["Velocity_Ratio"] = out["Amount"] / out["Avg_Spent_7D"].replace(0, 1)
        return out

    # ── Batch prediction ─────────────────────────────────────────────────────
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        aligned   = self.align_features(df)
        is_kaggle = "_is_kaggle" in aligned.columns
        X         = aligned[FEATURES]
        X_sc      = self.scaler.transform(X)

        probs = self.model.predict_proba(X_sc)[:, 1]
        preds = (probs >= 0.5).astype(int)

        out = aligned.copy()
        out["Fraud_Probability"] = (probs * 100).round(2)
        out["Predicted_Fraud"]   = preds

        # Kaggle PCA features can't be reliably mapped to behavioral thresholds
        # so use model probability alone to determine risk level
        if is_kaggle:
            out["Risk_Level"] = "Legitimate"
            out.loc[probs >= 0.30, "Risk_Level"] = "Moderate Risk"
            out.loc[probs >= 0.50, "Risk_Level"] = "High Risk"
            out.drop(columns=["_is_kaggle"], inplace=True)
        else:
            out["Risk_Level"] = out.apply(
                lambda r: classify_risk(r.to_dict()), axis=1)

        # Per-feature risk labels
        for feat in ["Amount", "Time_Delta", "Distance_From_Home",
                     "Is_High_Risk_Merchant", "Velocity_Ratio"]:
            out[f"{feat}_Risk"] = out[feat].apply(
                lambda v, f=feat: feature_risk_level(f, v))  

        return out

    # ── Single transaction + SHAP ─────────────────────────────────────────────
    def predict_with_explanation(self, input_df: pd.DataFrame):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        aligned = self.align_features(input_df)
        X       = aligned[FEATURES]
        X_sc    = self.scaler.transform(X)

        prob        = self.model.predict_proba(X_sc)[0][1]
        shap_values = self.explainer.shap_values(
            pd.DataFrame(X_sc, columns=FEATURES))

        # KernelExplainer on predict_proba returns [class_0_shap, class_1_shap]
        # We always want class 1 (fraud) perspective so red = pushes toward fraud
        if isinstance(shap_values, list) and len(shap_values) == 2:
            sv = np.array(shap_values[1][0]).flatten()
        elif isinstance(shap_values, list) and len(shap_values) == 1:
            sv = -np.array(shap_values[0][0]).flatten()
        elif len(np.array(shap_values).shape) == 3:
            sv = np.array(shap_values)[0, :, 1].flatten()
        else:
            sv = -np.array(shap_values)[0].flatten()

        sv = sv[:len(FEATURES)]

        fig = Figure(figsize=(10, 5))
        ax  = fig.subplots()
        colors = ["#e63946" if v > 0 else "#2ec4b6" for v in sv]
        y_pos  = np.arange(len(FEATURES))

        ax.barh(y_pos, sv, color=colors, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(FEATURES, fontsize=11)
        ax.invert_yaxis()
        ax.axvline(0, color="white", lw=0.8, linestyle="--", alpha=0.4)
        ax.set_xlabel("← Reduces fraud risk  |  Increases fraud risk →", fontsize=10)
        ax.set_title("Explainable AI: Feature Impact on Fraud Decision", fontsize=13)
        ax.set_facecolor("#0d0d0d")
        fig.patch.set_facecolor("#0d0d0d")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor="#0d0d0d")
        buf.seek(0)

        risk = classify_risk(aligned.iloc[0].to_dict())
        return prob, buf, sv, FEATURES, risk

    # ── Global SHAP ───────────────────────────────────────────────────────────
    def get_global_explanation(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        df   = self.generate_synthetic_data(n_samples=300)
        X    = df[FEATURES]
        X_sc = self.scaler.transform(X)

        shap_values = self.explainer.shap_values(
            pd.DataFrame(X_sc, columns=FEATURES))

        if isinstance(shap_values, list):
            mean_shap = np.abs(shap_values[1]).mean(axis=0)
        elif len(np.array(shap_values).shape) == 3:
            mean_shap = np.abs(np.array(shap_values)[:, :, 1]).mean(axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)

        mean_shap  = np.array(mean_shap).flatten()[:len(FEATURES)]
        sorted_idx = np.argsort(mean_shap)
        top2       = np.argsort(mean_shap)[-2:]

        fig = Figure(figsize=(10, 6))
        ax  = fig.subplots()
        bars = ax.barh(np.arange(len(FEATURES)), mean_shap[sorted_idx],
                       color="#d4af37", alpha=0.85)
        for i, bar in enumerate(bars):
            if sorted_idx[i] in top2:
                bar.set_color("#e63946")

        ax.set_yticks(np.arange(len(FEATURES)))
        ax.set_yticklabels(np.array(FEATURES)[sorted_idx], fontsize=11)
        ax.set_xlabel("Mean |SHAP Value| — average impact on fraud probability", fontsize=10)
        ax.set_title("Global Feature Importance — What Drives Fraud Flags", fontsize=13)
        ax.set_facecolor("#0d0d0d")
        fig.patch.set_facecolor("#0d0d0d")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor="#0d0d0d")
        buf.seek(0)
        return buf

    # ── Live overview metrics ─────────────────────────────────────────────────
    def get_live_metrics(self, df: pd.DataFrame) -> dict:
        results   = self.predict_batch(df)
        total     = len(results)
        flagged   = int(results["Predicted_Fraud"].sum())
        high_risk = int((results["Risk_Level"] == "High Risk").sum())
        moderate  = int((results["Risk_Level"] == "Moderate Risk").sum())
        avg_p     = results.loc[results["Predicted_Fraud"] == 1,
                                "Fraud_Probability"].mean()

        true_fraud = int(results["Is_Fraud"].sum()) \
            if "Is_Fraud" in results.columns else None
        recall = round(flagged / true_fraud * 100, 1) \
            if true_fraud else None

        return {
            "total":      total,
            "flagged":    flagged,
            "high_risk":  high_risk,
            "moderate":   moderate,
            "avg_prob":   round(float(avg_p), 1) if not np.isnan(avg_p) else 0,
            "recall":     recall,
            "true_fraud": true_fraud,
        }

    # ── Training ──────────────────────────────────────────────────────────────
    def train_stacking_ensemble(self, df):
        from xgboost import XGBClassifier

        if "Velocity_Ratio" not in df.columns:
            df = df.copy()
            df["Velocity_Ratio"] = df["Amount"] / df["Avg_Spent_7D"].replace(0, 1)

        X = df[FEATURES]
        y = df["Is_Fraud"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc  = self.scaler.transform(X_test)

        base = [
            ("rf",  RandomForestClassifier(n_estimators=150, max_depth=10,
                                           class_weight="balanced", random_state=42)),
            ("xgb", XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05,
                                  objective="binary:logistic", scale_pos_weight=5,
                                  random_state=42, verbosity=0)),
        ]
        self.model = StackingClassifier(
            estimators=base,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5)

        self.model.fit(X_train_sc, y_train)

        bg = shap.sample(pd.DataFrame(X_train_sc, columns=FEATURES), 50)
        self.explainer = shap.KernelExplainer(self.model.predict_proba, bg)
        self._save_model()

        y_pred = self.model.predict(X_test_sc)
        y_prob = self.model.predict_proba(X_test_sc)[:, 1]

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred).tolist()

        return {
            "classification_report": classification_report(y_test, y_pred),
            "precision": round(precision_score(y_test, y_pred, zero_division=0)*100, 2),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0)*100, 2),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0)*100, 2),
            "auc_roc":   round(roc_auc_score(y_test, y_prob)*100, 2),
            "confusion": cm,
            "total_test": int(len(y_test)),
            "fraud_test": int(y_test.sum()),
            "caught":     int(((y_pred==1) & (y_test==1)).sum()),
        }

    # ── Scaling analysis ──────────────────────────────────────────────────────
    def run_scaling_analysis(self, custom_df=None, custom_sizes=None, progress_callback=None):
        import time as _time
        sizes   = sorted([int(s) for s in custom_sizes]) \
            if custom_sizes else [30000, 50000, 150000, 284307]
        full_df = custom_df if custom_df is not None \
            else self.generate_synthetic_data(n_samples=max(sizes))
        full_df = self.align_features(full_df)
        results = []

        for step, size in enumerate(sizes, start=1):
            size    = min(size, len(full_df))
            subset  = full_df.iloc[:size].copy()
            X_sc    = self.scaler.transform(subset[FEATURES])
            t0      = _time.time()
            preds   = self.model.predict(X_sc)
            elapsed = _time.time() - t0
            # Use model probability for risk counts — correct for both Kaggle and synthetic
            probs_sub = self.model.predict_proba(X_sc)[:, 1]
            high_r    = int((probs_sub >= 0.50).sum())
            mod_r     = int(((probs_sub >= 0.30) & (probs_sub < 0.50)).sum())

            results.append({
                "Dataset Size":    f"{size:,}",
                "Actual Size":     size,
                "Fraud Detected":  int(preds.sum()),
                "High Risk":       high_r,
                "Moderate Risk":   mod_r,
                "Detection Rate":  f"{preds.sum()/size*100:.2f}%",
                "Processing Time": f"{elapsed:.2f}s",
            })

            if progress_callback:
                progress_callback(step, len(sizes), size)

        return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
#  KAGGLE FRAUD ENGINE
#  Trained directly on V1-V28 + Amount + Time — no feature mapping
# ─────────────────────────────────────────────────────────────────────────────
KAGGLE_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

class KaggleFraudEngine:
    def __init__(self):
        self.model      = None
        self.explainer  = None
        self.scaler     = StandardScaler()
        self.model_path = "fraud_model_kaggle.pkl"
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    saved = pickle.load(f)
                if isinstance(saved, tuple) and len(saved) == 3:
                    self.model, self.explainer, self.scaler = saved
            except Exception:
                pass

    def _save_model(self):
        with open(self.model_path, "wb") as f:
            pickle.dump((self.model, self.explainer, self.scaler), f)

    def is_ready(self):
        return self.model is not None

    def train(self, csv_path: str) -> dict:
        from xgboost import XGBClassifier

        df = pd.read_csv(csv_path)
        X  = df[KAGGLE_FEATURES]
        y  = df["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc  = self.scaler.transform(X_test)

        # Class weight to handle 0.17% fraud ratio
        scale = int((y == 0).sum() / (y == 1).sum())

        base = [
            ("rf",  RandomForestClassifier(n_estimators=100, max_depth=8,
                                           class_weight="balanced", random_state=42,
                                           n_jobs=-1)),
            ("xgb", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                  objective="binary:logistic",
                                  scale_pos_weight=scale,
                                  random_state=42, verbosity=0, n_jobs=-1)),
        ]
        self.model = StackingClassifier(
            estimators=base,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5)

        self.model.fit(X_train_sc, y_train)

        # SHAP background sample
        bg = shap.sample(pd.DataFrame(X_train_sc, columns=KAGGLE_FEATURES), 50)
        self.explainer = shap.KernelExplainer(self.model.predict_proba, bg)
        self._save_model()

        y_pred = self.model.predict(X_test_sc)
        y_prob = self.model.predict_proba(X_test_sc)[:, 1]

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred).tolist()

        return {
            "classification_report": classification_report(y_test, y_pred),
            "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
            "auc_roc":   round(roc_auc_score(y_test, y_prob) * 100, 2),
            "confusion": cm,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        X    = df[KAGGLE_FEATURES]
        X_sc = self.scaler.transform(X)

        probs = self.model.predict_proba(X_sc)[:, 1]
        preds = (probs >= 0.5).astype(int)

        out = df.copy()
        out["Fraud_Probability"] = (probs * 100).round(2)
        out["Predicted_Fraud"]   = preds

        # Probability-based risk — appropriate for real imbalanced data
        out["Risk_Level"] = "Legitimate"
        out.loc[probs >= 0.40, "Risk_Level"] = "Moderate Risk"
        out.loc[probs >= 0.65, "Risk_Level"] = "High Risk"

        return out

    def predict_with_explanation(self, input_df: pd.DataFrame):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        X    = input_df[KAGGLE_FEATURES]
        X_sc = self.scaler.transform(X)

        prob        = self.model.predict_proba(X_sc)[0][1]
        shap_values = self.explainer.shap_values(
            pd.DataFrame(X_sc, columns=KAGGLE_FEATURES))

        if isinstance(shap_values, list):
            sv = np.array(shap_values[1][0]).flatten()
        elif len(np.array(shap_values).shape) == 3:
            sv = np.array(shap_values)[0, :, 1].flatten()
        else:
            sv = np.array(shap_values)[0].flatten()

        sv = sv[:len(KAGGLE_FEATURES)]

        # Show top 10 features by impact for readability
        top_idx    = np.argsort(np.abs(sv))[-10:]
        top_feats  = [KAGGLE_FEATURES[i] for i in top_idx]
        top_sv     = sv[top_idx]

        fig = Figure(figsize=(10, 6))
        ax  = fig.subplots()
        colors = ["#e63946" if v > 0 else "#2ec4b6" for v in top_sv]
        y_pos  = np.arange(len(top_feats))

        ax.barh(y_pos, top_sv, color=colors, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_feats, fontsize=11)
        ax.invert_yaxis()
        ax.axvline(0, color="white", lw=0.8, linestyle="--", alpha=0.4)
        ax.set_xlabel("← Reduces fraud risk  |  Increases fraud risk →", fontsize=10)
        ax.set_title("Explainable AI: Top 10 Feature Impact on Fraud Decision", fontsize=13)
        ax.set_facecolor("#0d0d0d")
        fig.patch.set_facecolor("#0d0d0d")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor="#0d0d0d")
        buf.seek(0)

        risk = "High Risk" if prob >= 0.65 else "Moderate Risk" if prob >= 0.4 else "Legitimate"
        return prob, buf, sv, KAGGLE_FEATURES, risk

    def get_global_explanation(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        # Use a small sample for global SHAP
        dummy = pd.DataFrame(
            self.scaler.transform(
                pd.DataFrame(np.zeros((1, len(KAGGLE_FEATURES))),
                             columns=KAGGLE_FEATURES)),
            columns=KAGGLE_FEATURES)

        # Generate background from scaler mean
        bg_vals = np.tile(self.scaler.mean_, (100, 1))
        noise   = np.random.normal(0, 0.1, bg_vals.shape)
        bg_df   = pd.DataFrame(bg_vals + noise, columns=KAGGLE_FEATURES)

        shap_values = self.explainer.shap_values(bg_df.iloc[:50])

        if isinstance(shap_values, list):
            mean_shap = np.abs(shap_values[1]).mean(axis=0)
        elif len(np.array(shap_values).shape) == 3:
            mean_shap = np.abs(np.array(shap_values)[:, :, 1]).mean(axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)

        mean_shap  = np.array(mean_shap).flatten()[:len(KAGGLE_FEATURES)]
        sorted_idx = np.argsort(mean_shap)
        top2       = np.argsort(mean_shap)[-2:]

        fig = Figure(figsize=(10, 7))
        ax  = fig.subplots()
        bars = ax.barh(np.arange(len(KAGGLE_FEATURES)),
                       mean_shap[sorted_idx], color="#d4af37", alpha=0.85)
        for i, bar in enumerate(bars):
            if sorted_idx[i] in top2:
                bar.set_color("#e63946")

        ax.set_yticks(np.arange(len(KAGGLE_FEATURES)))
        ax.set_yticklabels(np.array(KAGGLE_FEATURES)[sorted_idx], fontsize=9)
        ax.set_xlabel("Mean |SHAP Value| — average impact on fraud probability", fontsize=10)
        ax.set_title("Global Feature Importance (Kaggle Model)", fontsize=13)
        ax.set_facecolor("#0d0d0d")
        fig.patch.set_facecolor("#0d0d0d")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor="#0d0d0d")
        buf.seek(0)
        return buf

    def run_scaling_analysis(self, custom_df=None, custom_sizes=None, progress_callback=None):
        import time as _time
        sizes   = sorted([int(s) for s in custom_sizes]) \
            if custom_sizes else [30000, 50000, 150000, 284307]
        
        if custom_df is None:
            # We don't have a synthetic generator for Kaggle PCA features
            return pd.DataFrame()
            
        full_df = custom_df
        results = []

        for step, size in enumerate(sizes, start=1):
            size    = min(size, len(full_df))
            subset  = full_df.iloc[:size].copy()
            X_sc    = self.scaler.transform(subset[KAGGLE_FEATURES])
            t0      = _time.time()
            probs_sub = self.model.predict_proba(X_sc)[:, 1]
            elapsed = _time.time() - t0
            
            preds     = (probs_sub >= 0.50).astype(int)
            high_r    = int((probs_sub >= 0.65).sum())
            mod_r     = int(((probs_sub >= 0.40) & (probs_sub < 0.65)).sum())

            results.append({
                "Dataset Size":    f"{size:,}",
                "Actual Size":     size,
                "Fraud Detected":  int(preds.sum()),
                "High Risk":       high_r,
                "Moderate Risk":   mod_r,
                "Detection Rate":  f"{preds.sum()/size*100:.2f}%",
                "Processing Time": f"{elapsed:.2f}s",
            })

            if progress_callback:
                progress_callback(step, len(sizes), size)

        return pd.DataFrame(results)


if __name__ == "__main__":
    engine = FraudEngine()
    data   = engine.generate_synthetic_data(n_samples=5000)
    report = engine.train_stacking_ensemble(data)
    print(report["classification_report"])
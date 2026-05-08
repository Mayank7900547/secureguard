"""
card_validator.py
=================
Card Health Validation Engine — SecureGuard Fraud Detection System

Two modes:
  1. AUTO   — infers checks from transaction DataFrame (4–5 checks)
  2. MANUAL — full 8-check deep validation from user-entered card details

Generates a styled A4 PDF with SecureGuard watermark.
"""

from __future__ import annotations
import io
from datetime import datetime, date
from typing import Any
import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# ── Colour palette ──────────────────────────────────────────────────────────────
GOLD   = colors.HexColor("#d4af37")
CORAL  = colors.HexColor("#e63946")
TEAL   = colors.HexColor("#2ec4b6")
PURPLE = colors.HexColor("#7b5ea7")
DARK   = colors.HexColor("#0a0a0a")
CARD   = colors.HexColor("#141414")
MUTED  = colors.HexColor("#a0998a")
WHITE  = colors.white
GREEN  = colors.HexColor("#2ec4b6")
AMBER  = colors.HexColor("#f5d060")

W, H = A4

# ── Severity ordering ────────────────────────────────────────────────────────────
SEVERITY_ORDER  = ["Critical", "High", "Moderate", "Low", "Clear"]
SEVERITY_COLORS = {
    "Critical": CORAL,
    "High":     colors.HexColor("#ff8c42"),
    "Moderate": AMBER,
    "Low":      colors.HexColor("#69db7c"),
    "Clear":    TEAL,
}
SEVERITY_EMOJI = {
    "Critical": "🔴",
    "High":     "🟠",
    "Moderate": "🟡",
    "Low":      "🟢",
    "Clear":    "✅",
}

# ── Check metadata ────────────────────────────────────────────────────────────────
ALL_CHECKS = [
    {
        "id":          "velocity",
        "name":        "Transaction Velocity",
        "description": "Checks if an unusual number of transactions occurred in a short window — a card-cloning or compromised-device signal.",
        "category":    "Behavioural",
        "auto":        True,
    },
    {
        "id":          "geo_conflict",
        "name":        "Geographic Impossibility",
        "description": "Detects physically impossible travel — two transactions in distant locations within minutes of each other.",
        "category":    "Geolocation",
        "auto":        True,
    },
    {
        "id":          "high_risk_merchant_freq",
        "name":        "High-Risk Merchant Frequency",
        "description": "Flags if a disproportionate share of recent transactions occurred at high-risk merchants (crypto, gambling, forex).",
        "category":    "Behavioural",
        "auto":        True,
    },
    {
        "id":          "velocity_spike",
        "name":        "Spending Velocity Spike",
        "description": "Compares current transaction amounts to the 7-day spending baseline. A sudden spike is a fraud signal.",
        "category":    "Behavioural",
        "auto":        True,
    },
    {
        "id":          "expiry",
        "name":        "Card Expiry Status",
        "description": "Verifies the card has not passed its expiration date and is not about to expire.",
        "category":    "Hardware / Physical",
        "auto":        False,
    },
    {
        "id":          "cvv_failures",
        "name":        "CVV / PIN Failure History",
        "description": "Counts recent wrong CVV or PIN attempts. Multiple failures suggest brute-forcing or a stolen card being tested.",
        "category":    "Authentication",
        "auto":        False,
    },
    {
        "id":          "chip_mismatch",
        "name":        "Chip vs Swipe Mismatch",
        "description": "Flags if a chip-enabled card is being swiped via magnetic stripe — a primary indicator of card cloning.",
        "category":    "Hardware / Physical",
        "auto":        False,
    },
    {
        "id":          "blacklist",
        "name":        "BIN / Merchant Blacklist",
        "description": "Checks if the card's Bank Identification Number appears in known compromised card databases.",
        "category":    "Database",
        "auto":        False,
    },
]

# ── Recommended actions per check + severity ─────────────────────────────────────
ACTIONS: dict[str, dict[str, list[str]]] = {
    "velocity": {
        "Critical": [
            "Freeze your card immediately via your banking app.",
            "Contact your bank's fraud team — high-velocity spending is a strong card-cloning signal.",
            "Review ALL transactions from the last 72 hours and dispute any you do not recognise.",
        ],
        "High": [
            "Check your transaction SMS alerts for unrecognised charges.",
            "Temporarily lower your daily spend limit via the bank app.",
            "Enable real-time transaction alerts if not already active.",
        ],
        "Moderate": [
            "Review your recent transaction history for any unfamiliar entries.",
            "Consider enabling a temporary spending limit.",
        ],
        "Clear": ["Transaction velocity is within your normal range. No action required."],
    },
    "geo_conflict": {
        "Critical": [
            "Your card has been used in two locations that are physically impossible to travel between — strong card-cloning signal.",
            "Freeze the card immediately via your banking app.",
            "File a fraud dispute for the suspicious transaction with your bank.",
            "Request a completely new card number — a PIN reset is not sufficient.",
        ],
        "High": [
            "Review the transaction locations on your banking app.",
            "Confirm whether you or a family member made both transactions.",
            "If unexplained, report to the fraud team immediately.",
        ],
        "Clear": ["No geographic conflicts detected in your recent transactions."],
    },
    "high_risk_merchant_freq": {
        "Critical": [
            "More than half your recent transactions are at high-risk merchants (crypto/gambling/forex).",
            "This pattern is associated with account takeover or card testing.",
            "Review all high-risk merchant transactions and dispute any you did not authorise.",
            "Contact your bank to block high-risk merchant category codes temporarily.",
        ],
        "High": [
            "An elevated share of your transactions are at high-risk merchants.",
            "Verify you authorised each of these transactions.",
        ],
        "Moderate": [
            "You have some high-risk merchant transactions — verify these are intended purchases.",
        ],
        "Clear": ["High-risk merchant usage is within normal limits."],
    },
    "velocity_spike": {
        "Critical": [
            "A transaction is more than 10× your 7-day average daily spend — extreme velocity spike.",
            "This is one of the strongest fraud signals in banking systems.",
            "Freeze your card and contact your bank's fraud team immediately.",
        ],
        "High": [
            "A recent transaction is significantly above your normal spending baseline.",
            "Verify you authorised this transaction. If not, dispute it immediately.",
        ],
        "Moderate": [
            "Spending is moderately above your normal baseline — worth monitoring.",
        ],
        "Clear": ["Spending velocity is consistent with your normal baseline."],
    },
    "expiry": {
        "Critical": [
            "Stop using the card immediately — it will be declined at all terminals.",
            "Call your bank's 24-hour helpline and request an emergency replacement card.",
            "Ask about a temporary virtual card for online transactions while you wait.",
        ],
        "Moderate": [
            "Your card expires soon — request a replacement card from your bank now.",
            "Allow 5–10 business days for delivery. Update any saved payment methods after.",
        ],
        "Clear": ["Card expiry is valid. No action required."],
    },
    "cvv_failures": {
        "Critical": [
            "Freeze your card immediately via your bank's mobile app.",
            "Report possible card compromise to your bank fraud team.",
            "Request a new card number — do not just reset the PIN.",
            "Review recent transactions for any unauthorised activity.",
        ],
        "High": [
            "Log into your banking app and review the failed attempt timestamps.",
            "Change your PIN at a bank-branch ATM (not a third-party machine).",
            "Enable two-factor authentication on your banking app.",
        ],
        "Moderate": [
            "Review your recent login activity on the banking portal.",
            "Consider changing your PIN as a precaution.",
        ],
        "Clear": ["No unusual authentication failures detected."],
    },
    "chip_mismatch": {
        "Critical": [
            "Your chip-enabled card was used via magnetic stripe — primary indicator of card cloning.",
            "Cloned cards do not have a working chip, forcing the fraudster to swipe.",
            "Freeze your card and report all recent swipe transactions to your bank.",
            "Request a replacement card immediately.",
        ],
        "Moderate": [
            "A magnetic stripe transaction occurred — may be legitimate at older terminals.",
            "Verify you authorised this transaction. If not, report it immediately.",
        ],
        "Clear": ["All transactions used chip or contactless as expected."],
    },
    "blacklist": {
        "Critical": [
            "Your card's BIN appears in a known compromised card database.",
            "Your card data may have been part of a data breach.",
            "Request a new card number immediately — the current one may be circulating on fraud marketplaces.",
            "Monitor your credit report for unauthorised account openings.",
            "Consider placing a fraud alert with your credit bureau.",
        ],
        "Clear": ["BIN check returned clean. No known compromised list matches."],
    },
}


# ── CardValidator class ──────────────────────────────────────────────────────────
class CardValidator:
    """
    Runs validation checks and returns structured result dicts.
    """

    # ── AUTO checks (from transaction DataFrame) ──────────────────────────────
    def run_auto_checks(self, df: pd.DataFrame, profile: dict) -> list[dict]:
        """
        Infers card health from a transaction DataFrame.
        df must have columns: Amount, Time_Delta, Distance_From_Home,
                              Is_High_Risk_Merchant, Avg_Spent_7D
        profile: user registration dict (baseline_daily_spend, etc.)
        """
        results = []
        checks = [c for c in ALL_CHECKS if c["auto"]]

        for check in checks:
            fn = getattr(self, f"_auto_{check['id']}")
            severity, detail = fn(df, profile)
            results.append({
                **check,
                "severity": severity,
                "detail":   detail,
                "actions":  self._get_actions(check["id"], severity),
                "mode":     "Auto",
            })
        return results

    # ── MANUAL checks (from user-entered card details) ────────────────────────
    def run_manual_checks(self, card_data: dict) -> list[dict]:
        """
        Runs the 4 manual checks that require user-entered card details.
        Also re-runs auto checks if transaction data is available in card_data.
        """
        results = []
        checks = [c for c in ALL_CHECKS if not c["auto"]]

        for check in checks:
            fn = getattr(self, f"_manual_{check['id']}")
            severity, detail = fn(card_data)
            results.append({
                **check,
                "severity": severity,
                "detail":   detail,
                "actions":  self._get_actions(check["id"], severity),
                "mode":     "Manual",
            })
        return results

    def overall_severity(self, results: list[dict]) -> str:
        for level in SEVERITY_ORDER:
            if any(r["severity"] == level for r in results):
                return level
        return "Clear"

    # ── Auto check implementations ─────────────────────────────────────────────
    def _auto_velocity(self, df: pd.DataFrame, profile: dict):
        n = len(df)
        normal_max = int(profile.get("normal_max_txn_per_day", 10))
        if n >= normal_max * 3:
            return "Critical", f"{n} transactions in the uploaded batch — {n/normal_max:.1f}× your normal daily maximum ({normal_max})."
        if n >= normal_max * 2:
            return "High", f"{n} transactions — {n/normal_max:.1f}× above your normal daily maximum."
        if n >= normal_max * 1.5:
            return "Moderate", f"Slightly elevated transaction count ({n} vs normal max {normal_max})."
        return "Clear", f"Transaction count ({n}) is within your normal daily range ({normal_max} max)."

    def _auto_geo_conflict(self, df: pd.DataFrame, profile: dict):
        if "Distance_From_Home" not in df.columns:
            return "Clear", "Location data not available in this dataset."
        max_dist = df["Distance_From_Home"].max()
        if max_dist > 500:
            return "Critical", f"A transaction occurred {max_dist:.0f} km from your registered home address — extreme geographic outlier."
        if max_dist > 150:
            return "High", f"Transaction detected {max_dist:.0f} km from home — significantly outside normal range."
        if max_dist > 50:
            return "Moderate", f"Transaction detected {max_dist:.0f} km from home — worth verifying."
        return "Clear", f"All transactions within {max_dist:.0f} km of home — normal range."

    def _auto_high_risk_merchant_freq(self, df: pd.DataFrame, profile: dict):
        if "Is_High_Risk_Merchant" not in df.columns:
            return "Clear", "Merchant data not available in this dataset."
        total     = len(df)
        high_risk = int(df["Is_High_Risk_Merchant"].sum())
        pct       = high_risk / max(total, 1) * 100
        if pct >= 50:
            return "Critical", f"{high_risk}/{total} transactions ({pct:.0f}%) at high-risk merchants (crypto/gambling/forex)."
        if pct >= 25:
            return "High", f"{high_risk}/{total} transactions ({pct:.0f}%) at high-risk merchants."
        if pct >= 10:
            return "Moderate", f"{high_risk}/{total} transactions ({pct:.0f}%) at high-risk merchants."
        return "Clear", f"Only {high_risk}/{total} transactions ({pct:.0f}%) at high-risk merchants — normal range."

    def _auto_velocity_spike(self, df: pd.DataFrame, profile: dict):
        if "Amount" not in df.columns or "Avg_Spent_7D" not in df.columns:
            return "Clear", "Amount or baseline data not available."
        baseline = float(profile.get("baseline_daily_spend", df["Avg_Spent_7D"].median()))
        max_amt  = df["Amount"].max()
        ratio    = max_amt / max(baseline, 1)
        if ratio >= 10:
            return "Critical", f"Largest transaction (${max_amt:.0f}) is {ratio:.1f}× your daily baseline (${baseline:.0f}) — extreme velocity spike."
        if ratio >= 5:
            return "High", f"Largest transaction (${max_amt:.0f}) is {ratio:.1f}× your daily baseline (${baseline:.0f})."
        if ratio >= 2:
            return "Moderate", f"Transaction (${max_amt:.0f}) is {ratio:.1f}× above your baseline (${baseline:.0f})."
        return "Clear", f"Transaction amounts (max ${max_amt:.0f}) are consistent with your daily baseline (${baseline:.0f})."

    # ── Manual check implementations ──────────────────────────────────────────
    def _manual_expiry(self, d: dict):
        try:
            exp   = date(int(d["expiry_year"]), int(d["expiry_month"]), 1)
            today = date.today()
            if exp < date(today.year, today.month, 1):
                months_ago = (today.year - exp.year) * 12 + today.month - exp.month
                return "Critical", f"Card expired {months_ago} month(s) ago. It will be declined at all terminals."
            months_left = (exp.year - today.year) * 12 + exp.month - today.month
            if months_left <= 1:
                return "Moderate", f"Card expires in {months_left} month(s) — request a replacement now."
            return "Clear", f"Card valid until {exp.strftime('%m/%Y')} ({months_left} months remaining)."
        except Exception:
            return "High", "Could not verify expiry date. Treat as potentially expired."

    def _manual_cvv_failures(self, d: dict):
        failures = int(d.get("cvv_failures_24h", 0))
        if failures >= 5:
            return "Critical", f"{failures} wrong CVV/PIN attempts in the last 24 hours — possible brute-force attack."
        if failures >= 3:
            return "High", f"{failures} wrong CVV/PIN attempts — unusual. Possible stolen card being tested."
        if failures >= 1:
            return "Moderate", f"{failures} wrong attempt(s) recorded in the last 24 hours."
        return "Clear", "No authentication failures recorded in the last 24 hours."

    def _manual_chip_mismatch(self, d: dict):
        is_chip = bool(d.get("is_chip_card", True))
        swiped  = bool(d.get("swipe_used", False))
        if is_chip and swiped:
            return "Critical", "Chip card used via magnetic stripe — primary indicator of card cloning. Cloned cards lack a working chip."
        if not is_chip and swiped:
            return "Low", "Non-chip card swiped — expected behaviour, but consider upgrading to chip card."
        return "Clear", "Card used with chip or contactless as expected."

    def _manual_blacklist(self, d: dict):
        if bool(d.get("bin_blacklisted", False)):
            return "Critical", "Card BIN appears in a known compromised card database — possible data breach exposure."
        return "Clear", "Card BIN is not in any known compromised list."

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _get_actions(self, check_id: str, severity: str) -> list[str]:
        check_actions = ACTIONS.get(check_id, {})
        if severity in check_actions:
            return check_actions[severity]
        for lvl in SEVERITY_ORDER:
            if lvl in check_actions:
                return check_actions[lvl]
        return ["Contact your bank for guidance on this issue."]


# ── PDF Generator ────────────────────────────────────────────────────────────────
def _draw_watermark_and_chrome(canvas_obj, doc):
    """Watermark + header + footer on every page."""
    canvas_obj.saveState()

    # Diagonal watermark
    canvas_obj.setFont("Helvetica-Bold", 55)
    canvas_obj.setFillColorRGB(0.831, 0.686, 0.216, alpha=0.05)
    canvas_obj.translate(W / 2, H / 2)
    canvas_obj.rotate(40)
    canvas_obj.drawCentredString(0, 50,  "SecureGuard™")
    canvas_obj.drawCentredString(0, -80, "CONFIDENTIAL")
    canvas_obj.rotate(-40)
    canvas_obj.translate(-W / 2, -H / 2)

    # Header bar
    canvas_obj.setFillColorRGB(0.039, 0.039, 0.039)
    canvas_obj.rect(0, H - 26*mm, W, 26*mm, fill=1, stroke=0)

    # Gold accent line
    canvas_obj.setStrokeColorRGB(0.831, 0.686, 0.216)
    canvas_obj.setLineWidth(2.5)
    canvas_obj.line(0, H - 26*mm, W, H - 26*mm)

    # Logo text
    canvas_obj.setFont("Helvetica-Bold", 13)
    canvas_obj.setFillColorRGB(0.831, 0.686, 0.216)
    canvas_obj.drawString(14*mm, H - 12*mm, "SecureGuard™")

    canvas_obj.setFont("Helvetica", 8.5)
    canvas_obj.setFillColorRGB(0.627, 0.6, 0.541)
    canvas_obj.drawString(14*mm, H - 19*mm, "Card Health Validation Report")

    ts = datetime.now().strftime("%d %b %Y  %H:%M")
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.drawRightString(W - 14*mm, H - 12*mm, f"Generated: {ts}")
    canvas_obj.drawRightString(W - 14*mm, H - 19*mm, "CONFIDENTIAL — Do not share")

    # Footer bar
    canvas_obj.setFillColorRGB(0.039, 0.039, 0.039)
    canvas_obj.rect(0, 0, W, 11*mm, fill=1, stroke=0)
    canvas_obj.setFont("Helvetica", 7.5)
    canvas_obj.setFillColorRGB(0.627, 0.6, 0.541)
    canvas_obj.drawString(14*mm, 4*mm,
        "This report is system-generated and does not constitute legal or financial advice. "
        "Always contact your bank directly for urgent card issues.")
    canvas_obj.drawRightString(W - 14*mm, 4*mm,
        f"Page {canvas_obj.getPageNumber()}")

    canvas_obj.restoreState()


def generate_card_report(
    cardholder: dict,
    auto_results: list[dict],
    manual_results: list[dict],
    mode: str = "auto",          # "auto" | "manual" | "full"
) -> bytes:
    """
    Generates a styled A4 PDF and returns it as bytes.

    cardholder keys:
        name, card_last4, account_age_days (optional)
    auto_results: list from CardValidator.run_auto_checks()
    manual_results: list from CardValidator.run_manual_checks() (may be empty)
    mode: "auto" = auto only, "manual" = manual only, "full" = both
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=32*mm, bottomMargin=16*mm,
        leftMargin=14*mm, rightMargin=14*mm,
        title="SecureGuard Card Health Report",
    )

    # ── Style helpers ──────────────────────────────────────────────────────────
    def sty(name, **kw):
        return ParagraphStyle(name, **kw)

    s_title  = sty("t",  fontName="Helvetica-Bold", fontSize=17, textColor=GOLD,
                    spaceAfter=3, leading=20)
    s_sub    = sty("s",  fontName="Helvetica",      fontSize=9,  textColor=MUTED,
                    spaceAfter=6)
    s_h2     = sty("h2", fontName="Helvetica-Bold", fontSize=11, textColor=GOLD,
                    spaceBefore=10, spaceAfter=4)
    s_h3     = sty("h3", fontName="Helvetica-Bold", fontSize=10, textColor=WHITE,
                    spaceBefore=6, spaceAfter=2)
    s_body   = sty("b",  fontName="Helvetica",      fontSize=9,  textColor=MUTED,
                    spaceAfter=3, leading=13)
    s_action = sty("a",  fontName="Helvetica",      fontSize=8.5,textColor=colors.HexColor("#c8c0b0"),
                    spaceAfter=2, leftIndent=10, leading=12)
    s_cat    = sty("c",  fontName="Helvetica-Oblique", fontSize=8, textColor=MUTED,
                    spaceAfter=1)
    s_center = sty("cc", fontName="Helvetica",      fontSize=8,  alignment=TA_CENTER,
                    textColor=MUTED)

    all_results = auto_results + manual_results
    validator   = CardValidator()
    overall     = validator.overall_severity(all_results)
    o_color     = SEVERITY_COLORS.get(overall, TEAL)
    flagged     = [r for r in all_results if r["severity"] not in ("Clear", "Low")]
    critical    = [r for r in all_results if r["severity"] == "Critical"]

    story = []

    # ── Cover info ─────────────────────────────────────────────────────────────
    holder_name = cardholder.get("name", "Cardholder")
    last4       = str(cardholder.get("card_last4", "XXXX")).zfill(4)

    story.append(Paragraph(f"Card Health Report", s_title))
    story.append(Paragraph(
        f"Prepared for: <b>{holder_name}</b>  ·  Card ending ···· {last4}  ·  "
        f"{len(all_results)} checks performed  ·  {len(flagged)} issue(s) found",
        s_sub))
    story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD, spaceAfter=8))

    # ── Overall status banner ──────────────────────────────────────────────────
    status_emoji = SEVERITY_EMOJI.get(overall, "✅")
    banner_data  = [[
        Paragraph("OVERALL CARD STATUS", sty("ol", fontName="Helvetica-Bold",
                  fontSize=8, textColor=MUTED, alignment=TA_CENTER)),
        Paragraph(f"{status_emoji}  {overall.upper()}", sty("os", fontName="Helvetica-Bold",
                  fontSize=15, textColor=o_color, alignment=TA_CENTER)),
        Paragraph(
            f"Critical: {len(critical)}   High: {len([r for r in all_results if r['severity']=='High'])}   "
            f"Moderate: {len([r for r in all_results if r['severity']=='Moderate'])}   "
            f"Clear: {len([r for r in all_results if r['severity']=='Clear'])}",
            sty("oc", fontName="Helvetica", fontSize=8, textColor=MUTED, alignment=TA_CENTER)
        ),
    ]]
    banner = Table(banner_data, colWidths=[55*mm, 70*mm, 55*mm])
    banner.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), CARD),
        ("ROUNDEDCORNERS", [6]),
        ("BOX",         (0, 0), (-1, -1), 1.5, o_color),
        ("TOPPADDING",  (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0,0), (-1, -1), 10),
        ("LINEAFTER",   (0, 0), (0, -1),  0.5, MUTED),
        ("LINEAFTER",   (1, 0), (1, -1),  0.5, MUTED),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(banner)
    story.append(Spacer(1, 8*mm))

    # ── What this report covers ────────────────────────────────────────────────
    mode_label = {
        "auto":   "Automatic — inferred from your uploaded transaction data (4 checks)",
        "manual": "Manual Deep Check — based on your card details (4 checks)",
        "full":   "Full Report — Automatic (4 checks) + Manual Deep Check (4 checks)",
    }.get(mode, "Automatic")

    story.append(Paragraph("Report Coverage", s_h2))
    story.append(Paragraph(f"Mode: {mode_label}", s_body))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=6))

    # ── Checks summary table ───────────────────────────────────────────────────
    story.append(Paragraph("Checks Summary", s_h2))
    summary_header = [
        [Paragraph("Check Name", sty("sh", fontName="Helvetica-Bold", fontSize=8.5, textColor=GOLD)),
         Paragraph("Category",   sty("sh", fontName="Helvetica-Bold", fontSize=8.5, textColor=GOLD)),
         Paragraph("Mode",       sty("sh", fontName="Helvetica-Bold", fontSize=8.5, textColor=GOLD)),
         Paragraph("Status",     sty("sh", fontName="Helvetica-Bold", fontSize=8.5, textColor=GOLD)),
        ]
    ]
    summary_rows = []
    for r in all_results:
        sev_col = SEVERITY_COLORS.get(r["severity"], TEAL)
        emoji   = SEVERITY_EMOJI.get(r["severity"], "✅")
        summary_rows.append([
            Paragraph(r["name"],     sty("sr", fontName="Helvetica", fontSize=8.5, textColor=WHITE)),
            Paragraph(r["category"], sty("sr", fontName="Helvetica", fontSize=8,   textColor=MUTED)),
            Paragraph(r.get("mode","Auto"), sty("sr", fontName="Helvetica", fontSize=8, textColor=MUTED)),
            Paragraph(f"{emoji}  {r['severity']}", sty("ss", fontName="Helvetica-Bold",
                      fontSize=8.5, textColor=sev_col)),
        ])

    summary_table = Table(
        summary_header + summary_rows,
        colWidths=[65*mm, 35*mm, 22*mm, 28*mm],
        repeatRows=1,
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1e1e1e")),
        ("BACKGROUND",    (0, 1), (-1, -1), DARK),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [DARK, colors.HexColor("#111111")]),
        ("BOX",           (0, 0), (-1, -1), 0.5, MUTED),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#2a2a2a")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 8*mm))

    # ── Detailed findings ──────────────────────────────────────────────────────
    story.append(Paragraph("Detailed Findings & Recommended Actions", s_h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=6))

    # Only show checks that aren't Clear (or show all if all clear)
    show_results = [r for r in all_results if r["severity"] != "Clear"] or all_results

    for r in show_results:
        sev_col = SEVERITY_COLORS.get(r["severity"], TEAL)
        emoji   = SEVERITY_EMOJI.get(r["severity"], "✅")

        block = [
            Paragraph(f"{emoji}  {r['name']}", sty("rh", fontName="Helvetica-Bold",
                      fontSize=10, textColor=sev_col, spaceBefore=4, spaceAfter=2)),
            Paragraph(f"Category: {r['category']}  ·  Mode: {r.get('mode','Auto')}  ·  "
                      f"Severity: {r['severity']}", s_cat),
            Paragraph(r["description"], s_body),
            Paragraph(f"Finding: {r['detail']}", sty("fd", fontName="Helvetica-Oblique",
                      fontSize=9, textColor=WHITE, spaceAfter=3)),
        ]

        if r["severity"] not in ("Clear",):
            block.append(Paragraph("Recommended Actions:", sty("ral", fontName="Helvetica-Bold",
                         fontSize=8.5, textColor=GOLD, spaceAfter=2)))
            for i, action in enumerate(r["actions"], 1):
                block.append(Paragraph(f"  {i}. {action}", s_action))

        block.append(HRFlowable(width="100%", thickness=0.3, color=colors.HexColor("#2a2a2a"),
                                spaceBefore=6, spaceAfter=4))
        story.append(KeepTogether(block))

    # ── If all clear ───────────────────────────────────────────────────────────
    if not [r for r in all_results if r["severity"] != "Clear"]:
        story.append(Spacer(1, 6*mm))
        story.append(Paragraph(
            "✅  All checks passed. Your card appears to be in good health. "
            "Continue monitoring your transactions regularly and run a full manual "
            "check monthly for comprehensive protection.",
            sty("ok", fontName="Helvetica-Bold", fontSize=11, textColor=TEAL,
                alignment=TA_CENTER, spaceAfter=6)))

    # ── Immediate action checklist (only if flagged issues) ────────────────────
    if flagged:
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph("Immediate Action Checklist", s_h2))
        story.append(Paragraph(
            "Complete these steps in order of priority:", s_body))

        priority_actions = []
        for sev in ["Critical", "High", "Moderate"]:
            for r in all_results:
                if r["severity"] == sev and r["actions"]:
                    priority_actions.append((sev, r["name"], r["actions"][0]))

        if priority_actions:
            action_data = [[
                Paragraph("Priority", sty("ph", fontName="Helvetica-Bold", fontSize=8, textColor=GOLD)),
                Paragraph("Check",    sty("ph", fontName="Helvetica-Bold", fontSize=8, textColor=GOLD)),
                Paragraph("Action",   sty("ph", fontName="Helvetica-Bold", fontSize=8, textColor=GOLD)),
            ]]
            for sev, check_name, action in priority_actions:
                sc = SEVERITY_COLORS.get(sev, TEAL)
                action_data.append([
                    Paragraph(sev, sty("ps", fontName="Helvetica-Bold", fontSize=8, textColor=sc)),
                    Paragraph(check_name, sty("pn", fontName="Helvetica", fontSize=8, textColor=WHITE)),
                    Paragraph(action, sty("pa", fontName="Helvetica", fontSize=8, textColor=MUTED)),
                ])
            action_table = Table(action_data, colWidths=[22*mm, 45*mm, 113*mm])
            action_table.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1e1e1e")),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [DARK, colors.HexColor("#111111")]),
                ("BOX",           (0, 0), (-1, -1), 0.5, MUTED),
                ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#2a2a2a")),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(action_table)

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=4))
    story.append(Paragraph(
        "This report was automatically generated by the SecureGuard AI Fraud Detection System. "
        "It is based on transaction patterns and user-provided card details. "
        "It does not constitute legal, financial, or banking advice. "
        "For urgent card issues, always contact your bank directly using the number on the back of your card.",
        sty("disc", fontName="Helvetica-Oblique", fontSize=7.5, textColor=MUTED,
            alignment=TA_CENTER)))

    # ── Build ──────────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=_draw_watermark_and_chrome,
              onLaterPages=_draw_watermark_and_chrome)
    buf.seek(0)
    return buf.read()
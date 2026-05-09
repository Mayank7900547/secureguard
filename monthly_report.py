"""
monthly_report.py
=================
SecureGuard — Synthetic Transaction Generator + Monthly Summary PDF Report
 
Generates 15-30 days of realistic synthetic transactions for a user
and produces a comprehensive watermarked PDF summary report.
"""
 
from __future__ import annotations
import io
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
 
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas as rl_canvas
 
# ── Colour palette ──────────────────────────────────────────────────────────
GOLD   = colors.HexColor("#d4af37")
CORAL  = colors.HexColor("#e63946")
TEAL   = colors.HexColor("#2ec4b6")
DARK   = colors.HexColor("#0a0a0a")
MUTED  = colors.HexColor("#a0998a")
WHITE  = colors.white
GREEN  = colors.HexColor("#2ec4b6")
AMBER  = colors.HexColor("#f5d060")
NAVY   = colors.HexColor("#0f1d38")
ORANGE = colors.HexColor("#ff8c42")
 
W, H = A4
 
MERCHANTS_SAFE     = ["Grocery Store","Pharmacy","Petrol Station","Restaurant","Online Shopping",
                       "Supermarket","Coffee Shop","Bookstore","Clothing Store","Electronics"]
MERCHANTS_MODERATE = ["Travel Agency","Hotel","Car Rental","Gaming Store","Furniture Store"]
MERCHANTS_HIGH     = ["Crypto Exchange","Gambling","Forex Broker","Luxury Goods","Casino"]
 
CITIES_INDIA  = ["Mumbai","Delhi","Bangalore","Hyderabad","Chennai","Pune","Kolkata","Jaipur"]
CITIES_ABROAD = ["Dubai","London","Singapore","New York","Hong Kong","Shanghai","Moscow","Lagos"]
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
 
def generate_synthetic_transactions(
    profile: dict,
    days: int = 15,
    fraud_count: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic transaction history for a user profile.
 
    profile keys used:
        full_name, registered_location, daily_spend_limit,
        max_transactions_day, card_last4
    """
    random.seed(seed)
    np.random.seed(seed)
 
    daily_limit  = float(profile.get("daily_spend_limit", 80))
    max_txn_day  = int(profile.get("max_transactions_day", 10))
    home_city    = profile.get("registered_location", "India").split(",")[0].strip()
    end_date     = datetime.now()
    start_date   = end_date - timedelta(days=days)
 
    rows = []
    txn_id = 1000
 
    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        n_txns = random.randint(1, max(2, max_txn_day - 2))
 
        for _ in range(n_txns):
            hour   = random.randint(8, 22)
            minute = random.randint(0, 59)
            ts     = current_date.replace(hour=hour, minute=minute, second=random.randint(0,59))
 
            # Normal transaction
            merchant_cat = random.choices(
                ["safe", "moderate", "high"],
                weights=[0.80, 0.15, 0.05]
            )[0]
 
            if merchant_cat == "safe":
                merchant = random.choice(MERCHANTS_SAFE)
                amount   = round(random.uniform(5, daily_limit * 0.8), 2)
                location = home_city
                risk     = "Safe"
            elif merchant_cat == "moderate":
                merchant = random.choice(MERCHANTS_MODERATE)
                amount   = round(random.uniform(daily_limit * 0.5, daily_limit * 1.5), 2)
                location = random.choice([home_city] + CITIES_INDIA[:3])
                risk     = "Moderate"
            else:
                merchant = random.choice(MERCHANTS_HIGH)
                amount   = round(random.uniform(daily_limit, daily_limit * 3), 2)
                location = random.choice(CITIES_INDIA)
                risk     = "High"
 
            time_delta = random.randint(30, 150)
 
            rows.append({
                "txn_id":      f"TXN{txn_id:06d}",
                "timestamp":   ts,
                "date":        ts.strftime("%d %b %Y"),
                "time":        ts.strftime("%H:%M"),
                "merchant":    merchant,
                "location":    location,
                "amount":      amount,
                "time_delta":  time_delta,
                "risk_level":  risk,
                "flagged":     False,
                "flag_reason": "",
            })
            txn_id += 1
 
    # Inject fraud transactions
    fraud_indices = random.sample(range(len(rows)), min(fraud_count, len(rows)))
    for idx in fraud_indices:
        rows[idx]["merchant"]    = random.choice(MERCHANTS_HIGH)
        rows[idx]["location"]    = random.choice(CITIES_ABROAD)
        rows[idx]["amount"]      = round(random.uniform(daily_limit * 3, daily_limit * 8), 2)
        rows[idx]["time_delta"]  = random.randint(161, 400)
        rows[idx]["risk_level"]  = "High Risk"
        rows[idx]["flagged"]     = True
        rows[idx]["flag_reason"] = "Geographic impossibility + spending spike + time delta anomaly"
 
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  WATERMARK
# ══════════════════════════════════════════════════════════════════════════════
 
def _watermark(c, doc):
    """Draw SecureGuard watermark on every page."""
    c.saveState()
    c.setFont("Helvetica-Bold", 52)
    c.setFillColorRGB(0.83, 0.69, 0.22, alpha=0.06)
    c.translate(W / 2, H / 2)
    c.rotate(35)
    c.drawCentredString(0, 0, "SecureGuard AI")
    c.rotate(-35)
    c.translate(-W / 2, -H / 2)
 
    # Footer
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(20*mm, 10*mm,
        f"SecureGuard AI · Confidential Report · Generated {datetime.now().strftime('%d %b %Y %H:%M')}")
    c.drawRightString(W - 20*mm, 10*mm, f"Page {doc.page}")
    c.restoreState()
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  MONTHLY SUMMARY PDF
# ══════════════════════════════════════════════════════════════════════════════
 
def generate_monthly_report(
    profile: dict,
    txn_df:  pd.DataFrame,
    period:  str = "Last 15 Days",
) -> bytes:
    """
    Generate a comprehensive monthly summary PDF report.
    Returns PDF as bytes.
    """
    buf = io.BytesIO()
 
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=18*mm, bottomMargin=18*mm,
        leftMargin=18*mm, rightMargin=18*mm,
    )
 
    # ── Styles ────────────────────────────────────────────────────────────────
    def sty(name, **kw):
        return ParagraphStyle(name, **kw)
 
    S = {
        "title":    sty("title",    fontName="Helvetica-Bold",  fontSize=22, textColor=GOLD,   spaceAfter=4),
        "subtitle": sty("subtitle", fontName="Helvetica",       fontSize=10, textColor=MUTED,  spaceAfter=12),
        "h2":       sty("h2",       fontName="Helvetica-Bold",  fontSize=13, textColor=GOLD,   spaceBefore=10, spaceAfter=6),
        "h3":       sty("h3",       fontName="Helvetica-Bold",  fontSize=11, textColor=WHITE,  spaceBefore=8,  spaceAfter=4),
        "body":     sty("body",     fontName="Helvetica",       fontSize=9,  textColor=MUTED,  spaceAfter=4,   leading=14),
        "flag":     sty("flag",     fontName="Helvetica-Bold",  fontSize=9,  textColor=CORAL,  spaceAfter=2),
        "safe":     sty("safe",     fontName="Helvetica-Bold",  fontSize=9,  textColor=GREEN,  spaceAfter=2),
        "warn":     sty("warn",     fontName="Helvetica-Bold",  fontSize=9,  textColor=AMBER,  spaceAfter=2),
        "center":   sty("center",   fontName="Helvetica",       fontSize=9,  textColor=MUTED,  alignment=TA_CENTER),
        "small":    sty("small",    fontName="Helvetica",       fontSize=8,  textColor=MUTED,  spaceAfter=2),
    }
 
    story = []
 
    # ── COVER HEADER ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 8*mm))
 
    # Header table with logo text + report info
    header_data = [[
        Paragraph("🛡️ <b>SecureGuard AI</b>", sty("hdr", fontName="Helvetica-Bold", fontSize=18, textColor=GOLD)),
        Paragraph(
            f"<b>Monthly Summary Report</b><br/>"
            f"<font color='#a0998a' size='9'>{period} · Generated {datetime.now().strftime('%d %b %Y')}</font>",
            sty("hdr2", fontName="Helvetica-Bold", fontSize=11, textColor=WHITE, alignment=TA_RIGHT)
        ),
    ]]
    header_tbl = Table(header_data, colWidths=[W*0.55 - 36*mm, W*0.45 - 36*mm])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), NAVY),
        ("ROUNDEDCORNERS", [6]),
        ("TOPPADDING",   (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,0), (-1,-1), 10),
        ("LEFTPADDING",  (0,0), (-1,-1), 14),
        ("RIGHTPADDING", (0,0), (-1,-1), 14),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 6*mm))
 
    # ── CARDHOLDER INFO ───────────────────────────────────────────────────────
    card4    = profile.get("card_last4", "XXXX")
    name     = profile.get("full_name", "—")
    location = profile.get("registered_location", "—")
    ifsc     = profile.get("ifsc_code", "—")
    exp_m    = profile.get("card_expiry_month")
    exp_y    = profile.get("card_expiry_year")
    expiry   = f"{int(exp_m):02d}/{str(int(exp_y))[-2:]}" if exp_m and exp_y else "—"
 
    info_data = [
        [Paragraph("<b>Cardholder</b>", S["small"]),   Paragraph(name, S["body"]),
         Paragraph("<b>Card Number</b>", S["small"]),   Paragraph(f"**** **** **** {card4}", S["body"])],
        [Paragraph("<b>Home Location</b>", S["small"]), Paragraph(location, S["body"]),
         Paragraph("<b>Card Expiry</b>", S["small"]),   Paragraph(expiry, S["body"])],
        [Paragraph("<b>IFSC Code</b>", S["small"]),     Paragraph(ifsc, S["body"]),
         Paragraph("<b>Report Period</b>", S["small"]), Paragraph(period, S["body"])],
    ]
    info_tbl = Table(info_data, colWidths=[(W-36*mm)/4]*4)
    info_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#0D1526")),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#1e2d4a")),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GOLD, spaceAfter=6*mm))
 
    # ── SUMMARY STATISTICS ────────────────────────────────────────────────────
    story.append(Paragraph("📊 Period Summary", S["h2"]))
 
    total_txns    = len(txn_df)
    total_spend   = txn_df["amount"].sum()
    flagged_txns  = txn_df[txn_df["flagged"] == True]
    safe_txns     = txn_df[txn_df["risk_level"] == "Safe"]
    avg_txn       = txn_df["amount"].mean()
    max_txn_amt   = txn_df["amount"].max()
    flag_count    = len(flagged_txns)
    flag_pct      = (flag_count / total_txns * 100) if total_txns > 0 else 0
 
    stat_color = CORAL if flag_count > 0 else GREEN
 
    stats_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Total Transactions",    str(total_txns),          "Total Spend",         f"${total_spend:,.2f}"],
        ["Flagged Transactions",  str(flag_count),          "Flag Rate",           f"{flag_pct:.1f}%"],
        ["Safe Transactions",     str(len(safe_txns)),      "Avg Transaction",     f"${avg_txn:,.2f}"],
        ["Largest Transaction",   f"${max_txn_amt:,.2f}",  "Days Covered",        str(txn_df["date"].nunique())],
    ]
 
    col_w = (W - 36*mm) / 4
    stats_tbl = Table(stats_data, colWidths=[col_w*1.3, col_w*0.7, col_w*1.3, col_w*0.7])
    stats_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#162447")),
        ("BACKGROUND",    (0,1), (-1,-1), colors.HexColor("#0D1526")),
        ("TEXTCOLOR",     (0,0), (-1,0),  GOLD),
        ("TEXTCOLOR",     (0,1), (-1,-1), WHITE),
        ("TEXTCOLOR",     (1,2), (1,2),   stat_color),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",      (1,1), (1,-1),  "Helvetica-Bold"),
        ("FONTNAME",      (3,1), (3,-1),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#1e2d4a")),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("ALIGN",         (1,0), (1,-1),  "CENTER"),
        ("ALIGN",         (3,0), (3,-1),  "CENTER"),
    ]))
    story.append(stats_tbl)
    story.append(Spacer(1, 6*mm))
 
    # ── RISK DISTRIBUTION ─────────────────────────────────────────────────────
    story.append(Paragraph("🎯 Risk Distribution", S["h2"]))
 
    risk_counts = txn_df["risk_level"].value_counts().to_dict()
    risk_rows   = []
    for level, color_val in [("Safe", GREEN), ("Moderate", AMBER), ("High", ORANGE), ("High Risk", CORAL)]:
        count = risk_counts.get(level, 0)
        if count > 0:
            pct   = count / total_txns * 100
            bar   = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            risk_rows.append([
                Paragraph(f"<b>{level}</b>", sty("rl", fontName="Helvetica-Bold", fontSize=9, textColor=color_val)),
                Paragraph(bar[:20], sty("bar", fontName="Helvetica", fontSize=8, textColor=color_val)),
                Paragraph(f"<b>{count}</b>", sty("cnt", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE, alignment=TA_CENTER)),
                Paragraph(f"{pct:.1f}%", sty("pct", fontName="Helvetica", fontSize=9, textColor=MUTED, alignment=TA_CENTER)),
            ])
 
    if risk_rows:
        risk_tbl = Table(risk_rows, colWidths=[(W-36*mm)*x for x in [0.2, 0.55, 0.12, 0.13]])
        risk_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#0D1526")),
            ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#1e2d4a")),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ]))
        story.append(risk_tbl)
 
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#1e2d4a"), spaceAfter=4*mm))
 
    # ── FLAGGED TRANSACTIONS ──────────────────────────────────────────────────
    if flag_count > 0:
        story.append(Paragraph(f"🚨 Flagged Transactions ({flag_count})", S["h2"]))
        story.append(Paragraph(
            "The following transactions were flagged by SecureGuard's detection engine "
            "as suspicious based on geographic, velocity, and spending anomalies.",
            S["body"]
        ))
        story.append(Spacer(1, 3*mm))
 
        flag_header = [
            Paragraph("<b>Txn ID</b>",    S["small"]),
            Paragraph("<b>Date</b>",      S["small"]),
            Paragraph("<b>Time</b>",      S["small"]),
            Paragraph("<b>Merchant</b>",  S["small"]),
            Paragraph("<b>Location</b>",  S["small"]),
            Paragraph("<b>Amount</b>",    S["small"]),
            Paragraph("<b>Risk</b>",      S["small"]),
        ]
        flag_rows_data = [flag_header]
 
        for _, row in flagged_txns.iterrows():
            flag_rows_data.append([
                Paragraph(str(row["txn_id"]), S["small"]),
                Paragraph(str(row["date"]),   S["small"]),
                Paragraph(str(row["time"]),   S["small"]),
                Paragraph(str(row["merchant"]), S["small"]),
                Paragraph(str(row["location"]), sty("loc_f", fontName="Helvetica-Bold", fontSize=8, textColor=CORAL)),
                Paragraph(f"${row['amount']:,.2f}", sty("amt_f", fontName="Helvetica-Bold", fontSize=8, textColor=CORAL)),
                Paragraph("🔴 HIGH", sty("risk_f", fontName="Helvetica-Bold", fontSize=8, textColor=CORAL)),
            ])
 
        col_ws = [(W-36*mm)*x for x in [0.12, 0.13, 0.09, 0.22, 0.18, 0.13, 0.13]]
        flag_tbl = Table(flag_rows_data, colWidths=col_ws, repeatRows=1)
        flag_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#1a0505")),
            ("BACKGROUND",    (0,1), (-1,-1), colors.HexColor("#0D1526")),
            ("TEXTCOLOR",     (0,0), (-1,0),  CORAL),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8),
            ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#2a1010")),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#0D1526"), colors.HexColor("#111d35")]),
        ]))
        story.append(flag_tbl)
        story.append(Spacer(1, 4*mm))
 
        # Flag reasons
        for _, row in flagged_txns.iterrows():
            if row.get("flag_reason"):
                story.append(Paragraph(
                    f"<b>{row['txn_id']}</b> — {row['flag_reason']}",
                    sty("fr", fontName="Helvetica", fontSize=8, textColor=CORAL, leftIndent=8, spaceAfter=3)
                ))
 
        story.append(Spacer(1, 4*mm))
        story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#1e2d4a"), spaceAfter=4*mm))
 
    # ── ALL TRANSACTIONS TABLE ────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("📋 Full Transaction Log", S["h2"]))
    story.append(Paragraph(
        f"Complete record of all {total_txns} transactions during the {period} period.",
        S["body"]
    ))
    story.append(Spacer(1, 3*mm))
 
    txn_header = [
        Paragraph("<b>Txn ID</b>",   S["small"]),
        Paragraph("<b>Date</b>",     S["small"]),
        Paragraph("<b>Time</b>",     S["small"]),
        Paragraph("<b>Merchant</b>", S["small"]),
        Paragraph("<b>Location</b>", S["small"]),
        Paragraph("<b>Amount</b>",   S["small"]),
        Paragraph("<b>Status</b>",   S["small"]),
    ]
    txn_rows = [txn_header]
 
    for _, row in txn_df.iterrows():
        is_flagged = row.get("flagged", False)
        status_p   = Paragraph(
            "🔴 FLAGGED" if is_flagged else "✅ Safe",
            sty("st", fontName="Helvetica-Bold", fontSize=7,
                textColor=CORAL if is_flagged else GREEN)
        )
        amt_color = CORAL if is_flagged else WHITE
        txn_rows.append([
            Paragraph(str(row["txn_id"]),   S["small"]),
            Paragraph(str(row["date"]),     S["small"]),
            Paragraph(str(row["time"]),     S["small"]),
            Paragraph(str(row["merchant"]), S["small"]),
            Paragraph(str(row["location"]), S["small"]),
            Paragraph(f"${row['amount']:,.2f}",
                      sty("a", fontName="Helvetica-Bold" if is_flagged else "Helvetica",
                          fontSize=8, textColor=amt_color)),
            status_p,
        ])
 
    col_ws2 = [(W-36*mm)*x for x in [0.12, 0.13, 0.09, 0.22, 0.18, 0.13, 0.13]]
    txn_tbl = Table(txn_rows, colWidths=col_ws2, repeatRows=1)
    txn_tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0),  colors.HexColor("#162447")),
        ("TEXTCOLOR",      (0,0), (-1,0),  GOLD),
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 8),
        ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#1e2d4a")),
        ("TOPPADDING",     (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
        ("LEFTPADDING",    (0,0), (-1,-1), 5),
        ("RIGHTPADDING",   (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#0D1526"), colors.HexColor("#111d35")]),
    ]))
    story.append(txn_tbl)
    story.append(Spacer(1, 6*mm))
 
    # ── RECOMMENDATIONS ───────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=GOLD, spaceAfter=4*mm))
    story.append(Paragraph("💡 Recommendations & Actions", S["h2"]))
 
    recs = []
    if flag_count > 0:
        recs += [
            ("🚨 Immediate", CORAL,
             f"Review the {flag_count} flagged transaction(s) above with your bank immediately. "
             "If you did not authorise them, freeze your card and file a dispute."),
            ("🔐 Security", AMBER,
             "Change your banking app password and enable two-factor authentication if not already active."),
        ]
    if txn_df[txn_df["risk_level"] == "Moderate"].shape[0] > 2:
        recs.append(("⚠️ Monitor", AMBER,
            "You have multiple moderate-risk transactions. Consider reviewing your spending at travel agencies and hotels."))
 
    recs += [
        ("✅ Good Practice", GREEN,
         "Run a Full Card Health Check in the SecureGuard dashboard every 30 days for a comprehensive card safety report."),
        ("📧 Alerts", GREEN,
         "Ensure your alert email is up to date so you receive instant notifications for any suspicious activity."),
    ]
 
    for label, col, text in recs:
        rec_data = [[
            Paragraph(f"<b>{label}</b>",
                      sty("rl2", fontName="Helvetica-Bold", fontSize=9, textColor=col)),
            Paragraph(text, S["body"]),
        ]]
        rec_tbl = Table(rec_data, colWidths=[(W-36*mm)*0.18, (W-36*mm)*0.82])
        rec_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#0D1526")),
            ("LEFTBORDER",    (0,0), (0,-1),  3, col),
            ("LINEBEFOREALL", (0,0), (0,-1),  3, col),
            ("TOPPADDING",    (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("RIGHTPADDING",  (0,0), (-1,-1), 10),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(rec_tbl)
        story.append(Spacer(1, 3*mm))
 
    # ── FOOTER DISCLAIMER ─────────────────────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="100%", thickness=0.3, color=MUTED, spaceAfter=3*mm))
    story.append(Paragraph(
        "This report is generated by SecureGuard AI and is for informational purposes only. "
        "Transaction data shown is synthetic and generated for demonstration. "
        "SecureGuard AI is not a licensed financial institution. "
        "Always consult your bank for official account statements.",
        sty("disc", fontName="Helvetica-Oblique", fontSize=7, textColor=MUTED, alignment=TA_CENTER)
    ))
 
    doc.build(story, onFirstPage=_watermark, onLaterPages=_watermark)
    return buf.getvalue()
 
# ══════════════════════════════════════════════════════════════════════════════
#  KAGGLE FRAUD REPORT — uses real uploaded CSV transactions
# ══════════════════════════════════════════════════════════════════════════════
 
def prepare_kaggle_fraud_transactions(df: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    """
    Convert a Kaggle creditcard.csv DataFrame into a display-friendly
    transaction DataFrame compatible with generate_kaggle_fraud_report().
    """
    out = df.copy().head(limit).reset_index(drop=True)
 
    if "txn_id" not in out.columns:
        out["txn_id"] = [f"TXN{i+1:06d}" for i in range(len(out))]
 
    if "date" not in out.columns or "time" not in out.columns:
        base = datetime.now()
        timestamps = [base - timedelta(seconds=int(t)) for t in out.get("Time", range(len(out)))]
        out["date"] = [t.strftime("%d %b %Y") for t in timestamps]
        out["time"] = [t.strftime("%H:%M") for t in timestamps]
 
    if "merchant" not in out.columns:
        merchants_safe = ["Online Shopping", "Grocery Store", "Petrol Station",
                          "Restaurant", "Pharmacy", "Electronics", "Supermarket"]
        merchants_high = ["Crypto Exchange", "Gambling", "Forex Broker", "Casino"]
        random.seed(42)
        out["merchant"] = [
            random.choice(merchants_high) if row.get("Is_High_Risk_Merchant", 0) == 1
            else random.choice(merchants_safe)
            for _, row in out.iterrows()
        ]
 
    if "location" not in out.columns:
        cities = ["Mumbai", "Delhi", "Bangalore", "London", "Dubai",
                  "Singapore", "New York", "Chennai", "Pune", "Hyderabad"]
        random.seed(42)
        out["location"] = [random.choice(cities) for _ in range(len(out))]
 
    if "flagged" not in out.columns:
        if "Predicted_Fraud" in out.columns:
            out["flagged"] = out["Predicted_Fraud"].astype(bool)
        elif "Class" in out.columns:
            out["flagged"] = out["Class"].astype(bool)
        else:
            out["flagged"] = False
 
    if "risk_level" not in out.columns:
        if "Risk_Level" in out.columns:
            out["risk_level"] = out["Risk_Level"]
        else:
            out["risk_level"] = out["flagged"].map({True: "High Risk", False: "Safe"})
 
    if "amount" not in out.columns:
        out["amount"] = out.get("Amount", pd.Series([0.0] * len(out)))
 
    if "flag_reason" not in out.columns:
        out["flag_reason"] = out["flagged"].map(
            {True: "ML model flagged — high fraud probability", False: ""})
 
    return out[["txn_id", "date", "time", "merchant", "location",
                "amount", "risk_level", "flagged", "flag_reason"]]
 
 
def generate_kaggle_fraud_report(
    profile: dict,
    txn_df:  pd.DataFrame,
    period:  str = "Kaggle Dataset Analysis",
) -> bytes:
    """
    Generate a Kaggle fraud analysis PDF report.
    Reuses generate_monthly_report() — same layout, different period label.
    """
    return generate_monthly_report(profile, txn_df, period=period)

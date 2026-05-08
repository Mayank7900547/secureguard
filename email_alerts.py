"""
email_alerts.py
===============
SecureGuard — Gmail SMTP Alert System
Sends real-time fraud alerts and monthly summary emails.
"""

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from typing import Optional

# ── Gmail Config — reads from st.secrets (Streamlit Cloud) or falls back to hardcoded (local) ──
try:
    import streamlit as st
    GMAIL_ADDRESS  = st.secrets["GMAIL_ADDRESS"]
    GMAIL_APP_PASS = st.secrets["GMAIL_APP_PASS"]
except Exception:
    GMAIL_ADDRESS  = "dahiyamayank059@gmail.com"
    GMAIL_APP_PASS = "rvuqiezytzyptyda"


def _build_alert_html(user_name: str, flags: list[dict], session_data: dict, profile: dict) -> str:
    """Build a rich HTML email for real-time fraud alerts."""
    flag_rows = ""
    for f in flags:
        colour = {"Critical": "#e63946", "High": "#ff8c42",
                  "Moderate": "#f5d060", "Low": "#69db7c"}.get(f.get("severity","High"), "#f5d060")
        flag_rows += f"""
        <tr>
          <td style="padding:8px 12px;border-bottom:1px solid #222;color:#fff;font-size:14px;">{f.get('check','')}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #222;color:{colour};font-weight:700;font-size:14px;">{f.get('severity','')}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #222;color:#a0998a;font-size:13px;">{f.get('detail','')}</td>
        </tr>"""

    registered_loc = profile.get("registered_location", "India")
    session_loc    = session_data.get("location", "Unknown")
    amount         = session_data.get("amount", 0)
    num_tx         = session_data.get("num_transactions", 1)
    ts             = datetime.now().strftime("%d %b %Y, %H:%M IST")

    geo_warning = ""
    if registered_loc.lower() != session_loc.lower():
        geo_warning = f"""
        <div style="background:#1a0a0a;border:1px solid #e63946;border-radius:8px;padding:14px 18px;margin:16px 0;">
          <span style="color:#e63946;font-weight:700;font-size:15px;">🌍 Geographic Impossibility Detected</span><br>
          <span style="color:#a0998a;font-size:13px;margin-top:6px;display:block;">
            Your registered location is <strong style="color:#fff;">{registered_loc}</strong> but this transaction 
            originated from <strong style="color:#e63946;">{session_loc}</strong>.
            If you did not make this transaction, freeze your card immediately.
          </span>
        </div>"""

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#0a0a0a;font-family:'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0a0a0a;padding:30px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background:#111;border-radius:12px;border:1px solid #d4af3730;overflow:hidden;">

        <!-- Header -->
        <tr>
          <td style="background:linear-gradient(135deg,#1a1400,#0a0a0a);padding:28px 32px;border-bottom:1px solid #d4af3740;">
            <span style="font-size:26px;font-weight:800;background:linear-gradient(90deg,#d4af37,#f5d060);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:0.5px;">
              🛡️ SecureGuard AI
            </span>
            <div style="color:#a0998a;font-size:13px;margin-top:4px;">Real-time Fraud Alert System</div>
          </td>
        </tr>

        <!-- Alert Banner -->
        <tr>
          <td style="background:#1a0505;padding:20px 32px;border-bottom:1px solid #e6394630;">
            <div style="color:#e63946;font-size:20px;font-weight:700;">🚨 Suspicious Activity Detected</div>
            <div style="color:#a0998a;font-size:13px;margin-top:6px;">
              Hi <strong style="color:#fff;">{user_name}</strong>, we detected unusual activity on your account at <strong style="color:#f5d060;">{ts}</strong>.
            </div>
          </td>
        </tr>

        <!-- Transaction Details -->
        <tr>
          <td style="padding:20px 32px;border-bottom:1px solid #222;">
            <div style="color:#d4af37;font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
              Transaction Details
            </div>
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td style="color:#a0998a;font-size:13px;padding:4px 0;">Session Location</td>
                <td style="color:#e63946;font-weight:700;font-size:14px;text-align:right;">{session_loc}</td>
              </tr>
              <tr>
                <td style="color:#a0998a;font-size:13px;padding:4px 0;">Transaction Amount</td>
                <td style="color:#fff;font-size:14px;text-align:right;">${amount:,.2f}</td>
              </tr>
              <tr>
                <td style="color:#a0998a;font-size:13px;padding:4px 0;">Number of Transactions</td>
                <td style="color:#fff;font-size:14px;text-align:right;">{num_tx}</td>
              </tr>
              <tr>
                <td style="color:#a0998a;font-size:13px;padding:4px 0;">Your Registered Location</td>
                <td style="color:#2ec4b6;font-size:14px;text-align:right;">{registered_loc}</td>
              </tr>
            </table>
          </td>
        </tr>

        {geo_warning if geo_warning else ""}

        <!-- Flags Table -->
        <tr>
          <td style="padding:20px 32px;border-bottom:1px solid #222;">
            <div style="color:#d4af37;font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
              Flags Triggered ({len(flags)})
            </div>
            <table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #222;border-radius:8px;overflow:hidden;">
              <tr style="background:#1e1e1e;">
                <td style="padding:8px 12px;color:#d4af37;font-size:12px;font-weight:700;">CHECK</td>
                <td style="padding:8px 12px;color:#d4af37;font-size:12px;font-weight:700;">SEVERITY</td>
                <td style="padding:8px 12px;color:#d4af37;font-size:12px;font-weight:700;">DETAIL</td>
              </tr>
              {flag_rows}
            </table>
          </td>
        </tr>

        <!-- Action Steps -->
        <tr>
          <td style="padding:20px 32px;border-bottom:1px solid #222;">
            <div style="color:#d4af37;font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
              Immediate Actions Required
            </div>
            <div style="background:#0f1a0f;border:1px solid #2ec4b630;border-radius:8px;padding:14px 18px;">
              <div style="color:#2ec4b6;font-size:13px;line-height:1.8;">
                1️⃣ &nbsp;Log into your banking app and review recent transactions<br>
                2️⃣ &nbsp;If you did not authorise this, freeze your card immediately<br>
                3️⃣ &nbsp;Contact your bank's fraud team using the number on the back of your card<br>
                4️⃣ &nbsp;Change your banking app password and enable 2FA<br>
                5️⃣ &nbsp;Run a Full Card Health Check on SecureGuard for a complete report
              </div>
            </div>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="padding:18px 32px;background:#0a0a0a;">
            <div style="color:#555;font-size:11px;text-align:center;line-height:1.6;">
              This alert was generated automatically by SecureGuard AI.<br>
              If you made this transaction yourself, you can ignore this message.<br>
              © 2026 SecureGuard AI · Fraud Detection System
            </div>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


def send_fraud_alert(
    to_email:     str,
    user_name:    str,
    flags:        list[dict],
    session_data: dict,
    profile:      dict,
    pdf_bytes:    Optional[bytes] = None,
) -> dict:
    """
    Send a real-time fraud alert email with optional PDF attachment.
    Returns {ok: bool, error: str|None}
    """
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🚨 SecureGuard Alert — Suspicious Activity on Your Account"
        msg["From"]    = f"SecureGuard AI <{GMAIL_ADDRESS}>"
        msg["To"]      = to_email

        # Plain text fallback
        plain = (
            f"SecureGuard AI — Fraud Alert\n\n"
            f"Hi {user_name},\n\n"
            f"Suspicious activity was detected on your account.\n"
            f"Session Location: {session_data.get('location','Unknown')}\n"
            f"Amount: ${session_data.get('amount',0):,.2f}\n"
            f"Flags: {len(flags)} triggered\n\n"
            f"Please log into your banking app immediately and freeze your card if you did not make this transaction.\n\n"
            f"— SecureGuard AI"
        )
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(_build_alert_html(user_name, flags, session_data, profile), "html"))

        # Attach PDF report if provided
        if pdf_bytes:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(pdf_bytes)
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                "attachment",
                filename=f"SecureGuard_Alert_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            )
            msg.attach(part)

        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
            server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())

        return {"ok": True}

    except Exception as e:
        return {"ok": False, "error": str(e)}


def send_monthly_summary(
    to_email:  str,
    user_name: str,
    pdf_bytes: bytes,
    period:    str = "Last 15 Days",
) -> dict:
    """Send the monthly/15-day summary PDF report."""
    try:
        msg = MIMEMultipart()
        msg["Subject"] = f"📊 SecureGuard Monthly Report — {period}"
        msg["From"]    = f"SecureGuard AI <{GMAIL_ADDRESS}>"
        msg["To"]      = to_email

        body = MIMEText(
            f"Hi {user_name},\n\n"
            f"Your SecureGuard card health & fraud summary for the {period} is attached.\n\n"
            f"Review it to stay on top of any flagged transactions or card health warnings.\n\n"
            f"— SecureGuard AI",
            "plain",
        )
        msg.attach(body)

        part = MIMEBase("application", "octet-stream")
        part.set_payload(pdf_bytes)
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            "attachment",
            filename=f"SecureGuard_Summary_{datetime.now().strftime('%Y%m%d')}.pdf",
        )
        msg.attach(part)

        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
            server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())

        return {"ok": True}

    except Exception as e:
        return {"ok": False, "error": str(e)}
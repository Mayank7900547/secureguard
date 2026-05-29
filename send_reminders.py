"""
send_reminders.py
=================
SecureGuard — Daily Reminder Script
Run by GitHub Actions every morning at 9am IST.
Checks all users in Supabase whose Card Health Check is overdue
and sends them a reminder email.
"""

import os
import sys
import requests
import smtplib
import ssl
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ── Config from GitHub Actions secrets ───────────────────────────────────────
SUPABASE_URL  = os.environ.get("SUPABASE_URL",  "https://jeesqtofaccdvztwgnhc.supabase.co")
SUPABASE_ANON = os.environ.get("SUPABASE_ANON", "")
SUPABASE_SERVICE = os.environ.get("SUPABASE_SERVICE_KEY", "")  # service role key — bypasses RLS
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASS = os.environ.get("GMAIL_APP_PASS", "")
APP_URL       = os.environ.get("APP_URL", "https://a6a28n9sf5txvyahfppggz.streamlit.app")

DB_URL = f"{SUPABASE_URL}/rest/v1"

# ── Fetch all user profiles using service role key (bypasses RLS) ─────────────
def fetch_all_profiles() -> list:
    headers = {
        "apikey":        SUPABASE_SERVICE,
        "Authorization": f"Bearer {SUPABASE_SERVICE}",
        "Content-Type":  "application/json",
    }
    r = requests.get(
        f"{DB_URL}/user_profiles",
        headers=headers,
        params={"select": "user_id,full_name,email,reminder_frequency,last_report_at,card_last4,registered_location,daily_spend_limit"},
        timeout=15,
    )
    if r.status_code == 200:
        return r.json()
    print(f"❌ Failed to fetch profiles: {r.status_code} {r.text}")
    return []


# ── Check if a user is overdue ────────────────────────────────────────────────
def is_overdue(profile: dict) -> tuple:
    """Returns (overdue: bool, days_since: int)"""
    freq      = profile.get("reminder_frequency", "15days")
    freq_days = {"weekly": 7, "15days": 15, "30days": 30}.get(freq, 15)
    last_raw  = profile.get("last_report_at")

    if not last_raw:
        return True, 999  # never generated — always overdue

    try:
        last_dt    = datetime.fromisoformat(last_raw.replace("Z", ""))
        days_since = (datetime.utcnow() - last_dt).days
        return days_since >= freq_days, days_since
    except Exception:
        return True, 999


# ── Build reminder email HTML ─────────────────────────────────────────────────
def build_html(profile: dict, days_since: int) -> str:
    user_name   = profile.get("full_name", "User")
    card_last4  = profile.get("card_last4", "****")
    location    = profile.get("registered_location", "India")
    daily_limit = profile.get("daily_spend_limit", 80)
    freq        = profile.get("reminder_frequency", "15days")
    freq_label  = {"weekly": "Weekly", "15days": "Every 15 Days", "30days": "Every 30 Days"}.get(freq, "Every 15 Days")
    ts          = datetime.now().strftime("%d %b %Y")

    days_str = f"{days_since} days" if days_since < 900 else "a long time"

    checks_html = ""
    checks = [
        ("⚡ Transaction Velocity",        "Unusual number of transactions in a short window — card cloning signal."),
        ("🌍 Geographic Impossibility",    "Two transactions in physically impossible locations within minutes."),
        ("🏪 High-Risk Merchant Frequency","Too many transactions at casinos, crypto exchanges, or gambling sites."),
        ("📈 Spending Velocity Spike",     "A transaction more than 10× your 7-day average spend."),
        ("💳 Card Expiry Status",          "Check if your card is close to expiry or already expired."),
        ("🔐 CVV / PIN Failure History",   "Recent wrong CVV or PIN attempts — brute-forcing signal."),
        ("🔄 Chip vs Swipe Mismatch",      "Chip card used via magnetic stripe — primary card-cloning indicator."),
        ("🚫 BIN / Merchant Blacklist",    "Transaction at a merchant known for fraud or data breaches."),
    ]
    for name, desc in checks:
        checks_html += f"""
        <tr>
          <td style="padding:10px 14px;border-bottom:1px solid #1e1e1e;">
            <span style="color:#d4af37;font-weight:700;font-size:13px;">{name}</span><br>
            <span style="color:#a0998a;font-size:12px;">{desc}</span>
          </td>
        </tr>"""

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#0a0a0a;font-family:'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0a0a0a;padding:30px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0"
        style="background:#111;border-radius:12px;border:1px solid #d4af3730;overflow:hidden;">

        <tr>
          <td style="background:linear-gradient(135deg,#1a1400,#0a0a0a);padding:28px 32px;
              border-bottom:1px solid #d4af3740;">
            <span style="font-size:26px;font-weight:800;
              background:linear-gradient(90deg,#d4af37,#f5d060);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
              🛡️ SecureGuard AI
            </span>
            <div style="color:#a0998a;font-size:13px;margin-top:4px;">Card Health Reminder System</div>
          </td>
        </tr>

        <tr>
          <td style="background:#1a1200;padding:20px 32px;border-bottom:1px solid #d4af3730;">
            <div style="color:#f5d060;font-size:20px;font-weight:700;">
              ⏰ Time for Your Card Health Check
            </div>
            <div style="color:#a0998a;font-size:13px;margin-top:6px;">
              Hi <strong style="color:#fff;">{user_name}</strong>, it's been
              <strong style="color:#f5d060;">{days_str}</strong> since your last
              SecureGuard report. Your {freq_label.lower()} check is due today.
            </div>
          </td>
        </tr>

        <tr>
          <td style="padding:20px 32px;border-bottom:1px solid #222;">
            <div style="color:#d4af37;font-size:13px;font-weight:700;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
              Your Card Profile
            </div>
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td style="color:#a0998a;font-size:13px;padding:4px 0;">Card</td>
                <td style="color:#fff;font-size:14px;text-align:right;">**** **** **** {card_last4}</td>
              </tr>
              <tr>
                <td style="color:#a0998a;font-size:13px;padding:4px 0;">Registered Location</td>
                <td style="color:#2ec4b6;font-size:14px;text-align:right;">{location}</td>
              </tr>
              <tr>
                <td style="color:#a0998a;font-size:13px;padding:4px 0;">Daily Spend Limit</td>
                <td style="color:#fff;font-size:14px;text-align:right;">${daily_limit:,.0f}</td>
              </tr>
              <tr>
                <td style="color:#a0998a;font-size:13px;padding:4px 0;">Reminder Date</td>
                <td style="color:#fff;font-size:14px;text-align:right;">{ts}</td>
              </tr>
            </table>
          </td>
        </tr>

        <tr>
          <td style="padding:20px 32px;border-bottom:1px solid #222;">
            <div style="color:#d4af37;font-size:13px;font-weight:700;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
              8 Checks to Run Today
            </div>
            <table width="100%" cellpadding="0" cellspacing="0"
              style="border:1px solid #222;border-radius:8px;overflow:hidden;background:#0f0f0f;">
              {checks_html}
            </table>
          </td>
        </tr>

        <tr>
          <td style="padding:28px 32px;text-align:center;border-bottom:1px solid #222;">
            <a href="{APP_URL}"
              style="background:linear-gradient(90deg,#d4af37,#f5d060);
                color:#0a0a0a;font-weight:800;font-size:16px;
                padding:14px 40px;border-radius:8px;text-decoration:none;
                display:inline-block;letter-spacing:0.5px;">
              🛡️ Open SecureGuard &amp; Generate Report
            </a>
            <div style="color:#555;font-size:11px;margin-top:12px;">
              Go to Card Health Check → Generate Full Report
            </div>
          </td>
        </tr>

        <tr>
          <td style="padding:20px 32px;background:#0f1a0f;border-bottom:1px solid #222;">
            <div style="color:#2ec4b6;font-size:13px;line-height:1.8;">
              💡 <strong>Why this matters:</strong> Most card fraud goes undetected for
              <strong>14–30 days</strong>. A SecureGuard check takes under 60 seconds
              and can catch cloning and velocity attacks before your bank does.
            </div>
          </td>
        </tr>

        <tr>
          <td style="padding:18px 32px;background:#0a0a0a;">
            <div style="color:#555;font-size:11px;text-align:center;line-height:1.6;">
              You're receiving this because you set a {freq_label.lower()} reminder in SecureGuard.<br>
              Change frequency anytime in the app under 🔔 Reminders.<br>
              © 2026 SecureGuard AI · Fraud Detection System
            </div>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


# ── Send email ────────────────────────────────────────────────────────────────
def send_reminder(to_email: str, profile: dict, days_since: int) -> bool:
    freq       = profile.get("reminder_frequency", "15days")
    freq_label = {"weekly": "Weekly", "15days": "15-Day", "30days": "30-Day"}.get(freq, "15-Day")
    user_name  = profile.get("full_name", "User")

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"⏰ SecureGuard — Run Your {freq_label} Card Health Check"
        msg["From"]    = f"SecureGuard AI <{GMAIL_ADDRESS}>"
        msg["To"]      = to_email

        plain = (
            f"Hi {user_name},\n\n"
            f"It's been {days_since} days since your last card health check.\n"
            f"Visit SecureGuard to run your 8-point health check:\n{APP_URL}\n\n"
            f"— SecureGuard AI"
        )
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(build_html(profile, days_since), "html"))

        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
            server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"   ❌ Email failed: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n🛡️  SecureGuard Daily Reminder — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    if not SUPABASE_SERVICE:
        print("❌ SUPABASE_SERVICE_KEY not set. Exiting.")
        sys.exit(1)
    if not GMAIL_ADDRESS or not GMAIL_APP_PASS:
        print("❌ Gmail credentials not set. Exiting.")
        sys.exit(1)

    profiles = fetch_all_profiles()
    print(f"👥 Found {len(profiles)} user profiles")

    sent = 0
    skipped = 0

    for profile in profiles:
        email = profile.get("email", "")
        name  = profile.get("full_name", "User")

        if not email:
            print(f"   ⚠️  Skipping {name} — no email set")
            skipped += 1
            continue

        overdue, days_since = is_overdue(profile)
        freq = profile.get("reminder_frequency", "15days")

        if overdue:
            print(f"   📧 Sending to {email} ({days_since} days since last report, freq={freq})")
            ok = send_reminder(email, profile, days_since)
            if ok:
                sent += 1
                print(f"   ✅ Sent!")
        else:
            freq_days = {"weekly": 7, "15days": 15, "30days": 30}.get(freq, 15)
            days_left = freq_days - days_since
            print(f"   ✓  {email} — {days_left} days until next reminder")
            skipped += 1

    print("=" * 60)
    print(f"✅ Done — {sent} reminder(s) sent, {skipped} skipped")


if __name__ == "__main__":
    main()

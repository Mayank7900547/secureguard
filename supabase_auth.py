"""
supabase_auth.py
================
SecureGuard — Supabase Auth + Database via REST API
No SDK needed — pure requests calls to Supabase REST endpoints.
"""

import requests
import json
from datetime import datetime
from typing import Optional

# ── Config ─────────────────────────────────────────────────────────────────────
SUPABASE_URL  = "https://jeesqtofaccdvztwgnhc.supabase.co"
SUPABASE_ANON = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImplZXNxdG9mYWNjZHZ6dHdnbmhjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgxNzg0MTgsImV4cCI6MjA5Mzc1NDQxOH0.xoij0Bdfd_F3BbRtlKXWDPmBg196k0gzl50NzZ-Q7PU"

AUTH_URL = f"{SUPABASE_URL}/auth/v1"
DB_URL   = f"{SUPABASE_URL}/rest/v1"

# ── Headers ─────────────────────────────────────────────────────────────────────
def _anon_headers():
    return {
        "apikey":       SUPABASE_ANON,
        "Content-Type": "application/json",
    }

def _auth_headers(access_token: str):
    return {
        "apikey":        SUPABASE_ANON,
        "Authorization": f"Bearer {access_token}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════════

def sign_up(email: str, password: str, full_name: str) -> dict:
    """Register a new user. Returns {ok, user, session, error}."""
    r = requests.post(
        f"{AUTH_URL}/signup",
        headers=_anon_headers(),
        json={
            "email":    email,
            "password": password,
            "data":     {"full_name": full_name},
        },
        timeout=10,
    )
    data = r.json()
    if r.status_code in (200, 201) and data.get("user"):
        return {"ok": True, "user": data["user"], "session": data.get("session")}
    # Supabase sometimes returns 200 with identities=[] meaning user already exists
    if r.status_code in (200, 201):
        return {"ok": True, "user": data.get("user", {}), "session": data.get("session")}
    err = data.get("msg") or data.get("error_description") or data.get("error") or "Signup failed"
    return {"ok": False, "error": str(err)}


def sign_in(email: str, password: str) -> dict:
    """Sign in existing user. Returns {ok, user, session, access_token, error}."""
    r = requests.post(
        f"{AUTH_URL}/token?grant_type=password",
        headers=_anon_headers(),
        json={"email": email, "password": password},
        timeout=10,
    )
    data = r.json()
    if r.status_code == 200 and data.get("access_token"):
        return {
            "ok":           True,
            "access_token": data["access_token"],
            "user":         data.get("user", {}),
            "session":      data,
        }
    return {"ok": False, "error": data.get("error_description") or data.get("msg") or str(data)}


def sign_out(access_token: str) -> bool:
    """Sign out the current user."""
    r = requests.post(
        f"{AUTH_URL}/logout",
        headers=_auth_headers(access_token),
        timeout=10,
    )
    return r.status_code in (200, 204)


# ══════════════════════════════════════════════════════════════════════════════
#  USER PROFILES TABLE
# ══════════════════════════════════════════════════════════════════════════════

def upsert_profile(access_token: str, user_id: str, profile: dict) -> dict:
    """Create or update a user's card profile."""
    payload = {
        "user_id":               user_id,
        "full_name":             profile.get("full_name", ""),
        "card_last4":            profile.get("card_last4", ""),
        "registered_location":   profile.get("registered_location", "India"),
        "daily_spend_limit":     float(profile.get("daily_spend_limit", 80)),
        "max_transactions_day":  int(profile.get("max_transactions_day", 10)),
        "email":                 profile.get("email", ""),
        "card_expiry_month":     profile.get("card_expiry_month"),
        "card_expiry_year":      profile.get("card_expiry_year"),
        "ifsc_code":             profile.get("ifsc_code", ""),
        "account_name":          profile.get("account_name", ""),
        "updated_at":            datetime.utcnow().isoformat(),
    }
    # Try update first, then insert if not exists
    # PATCH existing record
    patch = requests.patch(
        f"{DB_URL}/user_profiles",
        headers={**_auth_headers(access_token), "Prefer": "return=representation"},
        params={"user_id": f"eq.{user_id}"},
        json=payload,
        timeout=10,
    )
    if patch.status_code == 200:
        data = patch.json()
        if data:  # record existed and was updated
            return {"ok": True, "profile": data[0] if isinstance(data, list) else data}

    # No existing record — insert fresh
    ins = requests.post(
        f"{DB_URL}/user_profiles",
        headers={**_auth_headers(access_token), "Prefer": "return=representation"},
        json=payload,
        timeout=10,
    )
    if ins.status_code in (200, 201):
        data = ins.json()
        return {"ok": True, "profile": data[0] if isinstance(data, list) else data}
    return {"ok": False, "error": ins.text}


def get_profile(access_token: str, user_id: str) -> Optional[dict]:
    """Fetch a user's profile. Returns profile dict or None."""
    r = requests.get(
        f"{DB_URL}/user_profiles",
        headers=_auth_headers(access_token),
        params={"user_id": f"eq.{user_id}", "limit": "1"},
        timeout=10,
    )
    if r.status_code == 200:
        data = r.json()
        return data[0] if data else None
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSACTION SESSIONS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def log_session_transaction(access_token: str, user_id: str, session_data: dict) -> dict:
    """Log an active session transaction (from Panel 2 / attacker simulation)."""
    payload = {
        "user_id":          user_id,
        "session_location": session_data.get("location", "Unknown"),
        "amount":           float(session_data.get("amount", 0)),
        "num_transactions": int(session_data.get("num_transactions", 1)),
        "merchant_type":    session_data.get("merchant_type", "Normal"),
        "flagged":          session_data.get("flagged", False),
        "flags":            json.dumps(session_data.get("flags", [])),
        "timestamp":        datetime.utcnow().isoformat(),
    }
    r = requests.post(
        f"{DB_URL}/session_transactions",
        headers=_auth_headers(access_token),
        json=payload,
        timeout=10,
    )
    if r.status_code in (200, 201):
        return {"ok": True}
    return {"ok": False, "error": r.text}


def get_session_history(access_token: str, user_id: str, limit: int = 15) -> list:
    """Get recent session transactions for a user."""
    r = requests.get(
        f"{DB_URL}/session_transactions",
        headers=_auth_headers(access_token),
        params={
            "user_id": f"eq.{user_id}",
            "order":   "timestamp.desc",
            "limit":   str(limit),
        },
        timeout=10,
    )
    if r.status_code == 200:
        return r.json()
    return []


# ══════════════════════════════════════════════════════════════════════════════
#  ALERTS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def log_alert(access_token: str, user_id: str, alert: dict) -> dict:
    """Log a fraud/health alert to the database."""
    payload = {
        "user_id":     user_id,
        "alert_type":  alert.get("type", "fraud"),
        "severity":    alert.get("severity", "High"),
        "title":       alert.get("title", ""),
        "description": alert.get("description", ""),
        "flags":       json.dumps(alert.get("flags", [])),
        "timestamp":   datetime.utcnow().isoformat(),
        "read":        False,
    }
    r = requests.post(
        f"{DB_URL}/alerts",
        headers=_auth_headers(access_token),
        json=payload,
        timeout=10,
    )
    if r.status_code in (200, 201):
        return {"ok": True}
    return {"ok": False, "error": r.text}
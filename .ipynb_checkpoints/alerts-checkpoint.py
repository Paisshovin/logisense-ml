"""
Alert System — Email & SMS for High Risk Shipments
====================================================
Features:
  - Email alerts via Gmail SMTP (free)
  - SMS alerts via Twilio (free trial)
  - Auto-trigger when delay probability > 70%
  - Daily digest email of all high-risk shipments
  - Alert history saved to SQLite database

Install:
  pip install twilio python-dotenv schedule

Setup:
  1. Create .env file with your credentials (see below)
  2. Run: python alerts.py
"""

import os
import json
import smtplib
import sqlite3
import schedule
import time
import requests
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()  # loads from .env file

# ─────────────────────────────────────────────
# CONFIGURATION — set these in your .env file
# ─────────────────────────────────────────────

# Gmail settings
GMAIL_USER     = os.getenv("GMAIL_USER", "your_email@gmail.com")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD", "your_app_password")
ALERT_TO_EMAIL = os.getenv("ALERT_TO_EMAIL", "recipient@gmail.com")

# Twilio SMS settings
TWILIO_SID   = os.getenv("TWILIO_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN", "")
TWILIO_FROM  = os.getenv("TWILIO_FROM", "")   # your Twilio number e.g. +12345678901
ALERT_TO_SMS = os.getenv("ALERT_TO_SMS", "")  # recipient number e.g. +919876543210

# Alert thresholds
HIGH_RISK_THRESHOLD  = float(os.getenv("HIGH_RISK_THRESHOLD", "0.70"))
SMS_THRESHOLD        = float(os.getenv("SMS_THRESHOLD", "0.85"))  # only SMS for extreme risk
API_URL              = os.getenv("API_URL", "http://localhost:8000")


# ─────────────────────────────────────────────
# DATABASE — store alert history
# ─────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect("alerts.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alert_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            shipment_id     TEXT,
            delay_prob      REAL,
            risk_level      TEXT,
            alert_type      TEXT,
            recipient       TEXT,
            sent_at         TEXT,
            status          TEXT,
            error_message   TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_alert(shipment_id, delay_prob, risk_level, alert_type, recipient, status, error=None):
    conn = sqlite3.connect("alerts.db")
    conn.execute("""
        INSERT INTO alert_history
        (shipment_id, delay_prob, risk_level, alert_type, recipient, sent_at, status, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (shipment_id, delay_prob, risk_level, alert_type, recipient,
          datetime.now().isoformat(), status, error))
    conn.commit()
    conn.close()

def get_alert_history(limit=50):
    conn = sqlite3.connect("alerts.db")
    rows = conn.execute(
        "SELECT * FROM alert_history ORDER BY sent_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return rows


# ─────────────────────────────────────────────
# EMAIL ALERTS
# ─────────────────────────────────────────────

def build_email_html(shipment_id, prob, risk_level, factors, recommendation, est_delay):
    """Build a nicely formatted HTML email."""
    color = "#ef4444" if risk_level == "HIGH" else "#f59e0b"
    factors_html = "".join(f"<li style='margin:4px 0;color:#374151'>{f}</li>" for f in factors)

    return f"""
    <html><body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;background:#f9fafb;padding:20px">
      <div style="background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1)">

        <!-- Header -->
        <div style="background:{color};padding:24px 28px">
          <h1 style="color:white;margin:0;font-size:20px">⚠️ Delivery Delay Alert</h1>
          <p style="color:rgba(255,255,255,0.85);margin:6px 0 0;font-size:14px">
            LogiSense ML Forecasting System
          </p>
        </div>

        <!-- Body -->
        <div style="padding:28px">
          <table style="width:100%;border-collapse:collapse;margin-bottom:20px">
            <tr>
              <td style="padding:10px 0;border-bottom:1px solid #e5e7eb;font-size:13px;color:#6b7280">Shipment ID</td>
              <td style="padding:10px 0;border-bottom:1px solid #e5e7eb;font-size:13px;font-weight:bold;color:#111827">{shipment_id}</td>
            </tr>
            <tr>
              <td style="padding:10px 0;border-bottom:1px solid #e5e7eb;font-size:13px;color:#6b7280">Delay Probability</td>
              <td style="padding:10px 0;border-bottom:1px solid #e5e7eb;font-size:20px;font-weight:bold;color:{color}">{prob:.1%}</td>
            </tr>
            <tr>
              <td style="padding:10px 0;border-bottom:1px solid #e5e7eb;font-size:13px;color:#6b7280">Risk Level</td>
              <td style="padding:10px 0;border-bottom:1px solid #e5e7eb">
                <span style="background:{color};color:white;padding:3px 10px;border-radius:4px;font-size:12px;font-weight:bold">{risk_level}</span>
              </td>
            </tr>
            <tr>
              <td style="padding:10px 0;font-size:13px;color:#6b7280">Estimated Delay</td>
              <td style="padding:10px 0;font-size:13px;font-weight:bold;color:#111827">+{est_delay} days</td>
            </tr>
          </table>

          <div style="background:#fef3c7;border-left:4px solid #f59e0b;padding:12px 16px;border-radius:4px;margin-bottom:20px">
            <p style="margin:0;font-size:13px;font-weight:bold;color:#92400e">Key Risk Factors</p>
            <ul style="margin:8px 0 0;padding-left:18px">
              {factors_html}
            </ul>
          </div>

          <div style="background:#eff6ff;border-left:4px solid #3b82f6;padding:12px 16px;border-radius:4px;margin-bottom:24px">
            <p style="margin:0;font-size:13px;font-weight:bold;color:#1e40af">Recommendation</p>
            <p style="margin:6px 0 0;font-size:13px;color:#1e40af">{recommendation}</p>
          </div>

          <a href="{API_URL.replace('8000','')}/dashboard.html"
             style="background:#3b82f6;color:white;padding:12px 24px;border-radius:8px;text-decoration:none;font-size:13px;font-weight:bold;display:inline-block">
            Open LogiSense Dashboard →
          </a>
        </div>

        <!-- Footer -->
        <div style="background:#f3f4f6;padding:16px 28px;font-size:11px;color:#9ca3af">
          Sent by LogiSense ML Forecasting System · {datetime.now().strftime("%Y-%m-%d %H:%M")}
          <br>This alert was triggered automatically when delay probability exceeded {HIGH_RISK_THRESHOLD:.0%}.
        </div>
      </div>
    </body></html>
    """

def send_email_alert(prediction: dict) -> bool:
    """Send an email alert for a high-risk shipment."""
    shipment_id = prediction.get("shipment_id", "Unknown")
    prob        = prediction.get("delay_probability", 0)
    risk_level  = prediction.get("risk_level", "HIGH")
    factors     = prediction.get("key_risk_factors", [])
    rec         = prediction.get("recommendation", "")
    est_delay   = prediction.get("estimated_delay_days", 0)

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🚨 [{risk_level}] Shipment {shipment_id} — {prob:.1%} Delay Risk"
        msg["From"]    = GMAIL_USER
        msg["To"]      = ALERT_TO_EMAIL

        # Plain text fallback
        text = f"""
DELAY ALERT — {risk_level} RISK
Shipment: {shipment_id}
Delay Probability: {prob:.1%}
Estimated Delay: +{est_delay} days
Risk Factors: {', '.join(factors)}
Recommendation: {rec}
        """
        msg.attach(MIMEText(text, "plain"))
        msg.attach(MIMEText(build_email_html(shipment_id, prob, risk_level, factors, rec, est_delay), "html"))

        # Send via Gmail SMTP
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_USER, ALERT_TO_EMAIL, msg.as_string())

        print(f"  ✅ Email sent → {ALERT_TO_EMAIL} | {shipment_id} | {prob:.1%}")
        log_alert(shipment_id, prob, risk_level, "email", ALERT_TO_EMAIL, "sent")
        return True

    except Exception as e:
        print(f"  ❌ Email failed: {e}")
        log_alert(shipment_id, prob, risk_level, "email", ALERT_TO_EMAIL, "failed", str(e))
        return False


# ─────────────────────────────────────────────
# SMS ALERTS via Twilio
# ─────────────────────────────────────────────

def send_sms_alert(prediction: dict) -> bool:
    """Send SMS alert for extreme-risk shipments via Twilio."""
    if not TWILIO_SID or not TWILIO_TOKEN:
        print("  ⚠️  Twilio not configured — skipping SMS")
        return False

    shipment_id = prediction.get("shipment_id", "Unknown")
    prob        = prediction.get("delay_probability", 0)
    risk_level  = prediction.get("risk_level", "HIGH")
    est_delay   = prediction.get("estimated_delay_days", 0)
    factors     = prediction.get("key_risk_factors", [])[:2]  # top 2 only for SMS

    message = (
        f"🚨 LogiSense ALERT\n"
        f"Shipment: {shipment_id}\n"
        f"Delay Risk: {prob:.1%} ({risk_level})\n"
        f"Est. Delay: +{est_delay} days\n"
        f"Cause: {', '.join(factors)}\n"
        f"Action required immediately."
    )

    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=TWILIO_FROM,
            to=ALERT_TO_SMS
        )
        print(f"  ✅ SMS sent → {ALERT_TO_SMS} | SID: {msg.sid}")
        log_alert(shipment_id, prob, risk_level, "sms", ALERT_TO_SMS, "sent")
        return True

    except Exception as e:
        print(f"  ❌ SMS failed: {e}")
        log_alert(shipment_id, prob, risk_level, "sms", ALERT_TO_SMS, "failed", str(e))
        return False


# ─────────────────────────────────────────────
# DAILY DIGEST EMAIL
# ─────────────────────────────────────────────

def send_daily_digest(predictions: list):
    """Send a daily summary email of all high-risk shipments."""
    high_risk = [p for p in predictions if p.get("delay_probability", 0) >= HIGH_RISK_THRESHOLD]
    if not high_risk:
        print("  No high-risk shipments today — digest skipped.")
        return

    rows_html = ""
    for p in sorted(high_risk, key=lambda x: x["delay_probability"], reverse=True):
        color = "#ef4444" if p["risk_level"] == "HIGH" else "#f59e0b"
        rows_html += f"""
        <tr>
          <td style="padding:10px;border-bottom:1px solid #e5e7eb;font-size:12px">{p.get('shipment_id','—')}</td>
          <td style="padding:10px;border-bottom:1px solid #e5e7eb;font-size:14px;font-weight:bold;color:{color}">{p['delay_probability']:.1%}</td>
          <td style="padding:10px;border-bottom:1px solid #e5e7eb">
            <span style="background:{color};color:white;padding:2px 8px;border-radius:3px;font-size:11px">{p['risk_level']}</span>
          </td>
          <td style="padding:10px;border-bottom:1px solid #e5e7eb;font-size:11px;color:#6b7280">{p.get('estimated_delay_days',0)}d</td>
        </tr>"""

    html = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:700px;margin:0 auto">
      <div style="background:#1a2235;padding:24px;border-radius:12px 12px 0 0">
        <h1 style="color:white;margin:0;font-size:18px">📊 LogiSense Daily Digest</h1>
        <p style="color:#94a3b8;margin:4px 0 0;font-size:13px">{datetime.now().strftime("%A, %B %d %Y")}</p>
      </div>
      <div style="background:white;padding:24px;border-radius:0 0 12px 12px">
        <div style="display:flex;gap:16px;margin-bottom:24px">
          <div style="background:#fef2f2;padding:16px;border-radius:8px;flex:1;text-align:center">
            <div style="font-size:28px;font-weight:bold;color:#ef4444">{len(high_risk)}</div>
            <div style="font-size:12px;color:#6b7280">High Risk</div>
          </div>
          <div style="background:#f0fdf4;padding:16px;border-radius:8px;flex:1;text-align:center">
            <div style="font-size:28px;font-weight:bold;color:#10b981">{len(predictions)-len(high_risk)}</div>
            <div style="font-size:12px;color:#6b7280">On Track</div>
          </div>
          <div style="background:#eff6ff;padding:16px;border-radius:8px;flex:1;text-align:center">
            <div style="font-size:28px;font-weight:bold;color:#3b82f6">{len(predictions)}</div>
            <div style="font-size:12px;color:#6b7280">Total Scored</div>
          </div>
        </div>
        <h3 style="font-size:14px;color:#374151;margin-bottom:12px">High Risk Shipments</h3>
        <table style="width:100%;border-collapse:collapse">
          <thead>
            <tr style="background:#f9fafb">
              <th style="padding:10px;text-align:left;font-size:11px;color:#6b7280">SHIPMENT</th>
              <th style="padding:10px;text-align:left;font-size:11px;color:#6b7280">PROBABILITY</th>
              <th style="padding:10px;text-align:left;font-size:11px;color:#6b7280">RISK</th>
              <th style="padding:10px;text-align:left;font-size:11px;color:#6b7280">EST. DELAY</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </body></html>
    """

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"📊 LogiSense Daily Digest — {len(high_risk)} high-risk shipments · {datetime.now().strftime('%b %d')}"
        msg["From"]    = GMAIL_USER
        msg["To"]      = ALERT_TO_EMAIL
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_USER, ALERT_TO_EMAIL, msg.as_string())

        print(f"  ✅ Daily digest sent — {len(high_risk)} high-risk shipments")
    except Exception as e:
        print(f"  ❌ Digest failed: {e}")


# ─────────────────────────────────────────────
# MAIN ALERT CHECKER
# ─────────────────────────────────────────────

# Track already-alerted shipments to avoid duplicates
alerted_shipments = set()

def check_and_alert(shipments: list):
    """
    Score a list of shipments and send alerts for high-risk ones.
    Call this from your main application or scheduler.
    """
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking {len(shipments)} shipments...")

    try:
        # Batch predict via API
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"shipments": shipments},
            timeout=30
        )
        if not response.ok:
            print(f"  ❌ API error: {response.status_code}")
            return

        batch = response.json()
        print(f"  Scored: {batch['total']} | Delayed: {batch['delayed']} | High risk: {batch['high_risk']}")

        for pred in batch["predictions"]:
            sid  = pred.get("shipment_id", "unknown")
            prob = pred.get("delay_probability", 0)

            # Skip already alerted
            if sid in alerted_shipments:
                continue

            # Email for high risk
            if prob >= HIGH_RISK_THRESHOLD:
                print(f"  🚨 Alert triggered: {sid} — {prob:.1%}")
                send_email_alert(pred)
                alerted_shipments.add(sid)

            # SMS for extreme risk only
            if prob >= SMS_THRESHOLD:
                send_sms_alert(pred)

    except requests.exceptions.ConnectionError:
        print("  ❌ Cannot connect to API — is the server running?")
    except Exception as e:
        print(f"  ❌ Error: {e}")


# ─────────────────────────────────────────────
# ADD ALERT ENDPOINT TO FASTAPI
# Paste this into api.py to trigger alerts from predictions
# ─────────────────────────────────────────────

FASTAPI_ADDON = '''
# ── Add this to api.py ────────────────────────────────────────
# At the top: from alerts import send_email_alert, send_sms_alert, HIGH_RISK_THRESHOLD, SMS_THRESHOLD

@app.post("/predict/alert", response_model=PredictionResult)
def predict_with_alert(shipment: ShipmentInput, send_alert: bool = True):
    """Predict and automatically send alerts if high risk."""
    try:
        X   = preprocessor.transform(build_features(shipment)[feature_names])
        prob = float(model.predict_proba(X)[0, 1])
        result = build_result(shipment, prob)

        if send_alert and prob >= HIGH_RISK_THRESHOLD:
            send_email_alert(result.dict())
        if send_alert and prob >= SMS_THRESHOLD:
            send_sms_alert(result.dict())

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''


# ─────────────────────────────────────────────
# SCHEDULER — run checks automatically
# ─────────────────────────────────────────────

def run_scheduler():
    """
    Runs alert checks on a schedule.
    Modify the sample shipments list with your real data source.
    """
    print("=" * 55)
    print("  LogiSense Alert System — Starting scheduler")
    print("=" * 55)
    print(f"  Email alerts  : {ALERT_TO_EMAIL}")
    print(f"  SMS alerts    : {ALERT_TO_SMS or 'not configured'}")
    print(f"  High risk     : >{HIGH_RISK_THRESHOLD:.0%}")
    print(f"  SMS threshold : >{SMS_THRESHOLD:.0%}")
    print(f"  API           : {API_URL}")
    print()

    # Sample shipments to check — replace with real DB query
    sample_shipments = [
        {
            "shipment_id": "SH-4821", "order_date": "2024-11-15",
            "carrier": "MSC", "transport_mode": "ocean",
            "cargo_type": "hazmat", "customs_complexity": "high_scrutiny",
            "origin": "Shanghai", "destination": "Chicago",
            "weight_kg": 4800, "distance_km": 12500,
            "carrier_otr_30d": 0.68, "port_congestion": 0.78,
            "weather_severity": 0.72, "planned_lead_days": 5, "is_peak_season": 1,
        },
        {
            "shipment_id": "SH-3907", "order_date": "2024-11-15",
            "carrier": "Maersk", "transport_mode": "ocean",
            "cargo_type": "general", "customs_complexity": "standard",
            "origin": "Rotterdam", "destination": "Mumbai",
            "weight_kg": 2100, "distance_km": 9800,
            "carrier_otr_30d": 0.75, "port_congestion": 0.62,
            "weather_severity": 0.55, "planned_lead_days": 18, "is_peak_season": 1,
        },
        {
            "shipment_id": "SH-8871", "order_date": "2024-11-10",
            "carrier": "FedEx", "transport_mode": "air",
            "cargo_type": "general", "customs_complexity": "standard",
            "origin": "Los Angeles", "destination": "Chicago",
            "weight_kg": 80, "distance_km": 2800,
            "carrier_otr_30d": 0.95, "port_congestion": 0.10,
            "weather_severity": 0.08, "planned_lead_days": 14, "is_peak_season": 0,
        },
    ]

    # Check every 30 minutes
    schedule.every(30).minutes.do(check_and_alert, shipments=sample_shipments)

    # Daily digest at 8:00 AM
    def daily_digest_job():
        check_and_alert(sample_shipments)
        # Collect all predictions and send digest
        try:
            response = requests.post(f"{API_URL}/predict/batch", json={"shipments": sample_shipments})
            if response.ok:
                preds = response.json().get("predictions", [])
                send_daily_digest(preds)
        except Exception as e:
            print(f"Digest error: {e}")

    schedule.every().day.at("08:00").do(daily_digest_job)

    # Run immediately on start
    check_and_alert(sample_shipments)

    print("\n  Scheduler running:")
    print("  · Alert check : every 30 minutes")
    print("  · Daily digest: every day at 08:00")
    print("  Press Ctrl+C to stop\n")

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    init_db()
    run_scheduler()

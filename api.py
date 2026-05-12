"""
api_with_auth.py — FastAPI with JWT Authentication
===================================================
Updated api.py that includes:
  - JWT login system
  - Role-based endpoint protection
  - User management endpoints
  - All original prediction endpoints (now protected)

Usage:
  1. Rename this to api.py (replace old api.py)
  2. pip install python-jose[cryptography] passlib[bcrypt] python-multipart
  3. uvicorn api:app --reload --port 8000

Default accounts:
  admin   / admin123
  manager / manager123
  viewer  / viewer123
"""

import os
import smtplib
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from auth import (
    router as auth_router,
    get_current_user,
    require_role,
    init_users_db,
)

load_dotenv()

GMAIL_USER     = os.getenv("GMAIL_USER", "")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD", "")

def send_shipment_alert_email(to_email: str, user_name: str, prediction: dict):
    """Send single shipment high-risk alert email to the user."""
    if not GMAIL_USER or not GMAIL_PASSWORD:
        print("  Email not configured — skipping")
        return

    sid   = prediction.get("shipment_id","Unknown")
    prob  = prediction.get("delay_probability", 0)
    risk  = prediction.get("risk_level","HIGH")
    est   = prediction.get("estimated_delay_days", 0)
    rec   = prediction.get("recommendation","")
    facts = prediction.get("key_risk_factors",[])
    color = "#ef4444" if risk=="HIGH" else "#f59e0b"
    facts_html = "".join(f"<li style='margin:4px 0;color:#374151'>{f}</li>" for f in facts)

    html = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:580px;margin:0 auto;background:#f9fafb;padding:20px">
      <div style="background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08)">
        <div style="background:{color};padding:22px 28px">
          <h1 style="color:white;margin:0;font-size:18px">🚨 High Risk Delivery Alert</h1>
          <p style="color:rgba(255,255,255,0.85);margin:4px 0 0;font-size:13px">LogiSense ML Forecasting · Auto-alert</p>
        </div>
        <div style="padding:24px 28px">
          <p style="font-size:14px;color:#374151;margin-bottom:20px">Hi <b>{user_name}</b>, a shipment you scored has been flagged as high risk.</p>
          <table style="width:100%;border-collapse:collapse;margin-bottom:18px">
            <tr><td style="padding:9px 0;border-bottom:1px solid #e5e7eb;font-size:12px;color:#6b7280">Shipment ID</td><td style="padding:9px 0;border-bottom:1px solid #e5e7eb;font-size:13px;font-weight:bold">{sid}</td></tr>
            <tr><td style="padding:9px 0;border-bottom:1px solid #e5e7eb;font-size:12px;color:#6b7280">Delay Probability</td><td style="padding:9px 0;border-bottom:1px solid #e5e7eb;font-size:20px;font-weight:bold;color:{color}">{prob:.1%}</td></tr>
            <tr><td style="padding:9px 0;border-bottom:1px solid #e5e7eb;font-size:12px;color:#6b7280">Risk Level</td><td style="padding:9px 0;border-bottom:1px solid #e5e7eb"><span style="background:{color};color:white;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:bold">{risk}</span></td></tr>
            <tr><td style="padding:9px 0;font-size:12px;color:#6b7280">Estimated Delay</td><td style="padding:9px 0;font-size:13px;font-weight:bold">+{est} days</td></tr>
          </table>
          <div style="background:#fef3c7;border-left:4px solid #f59e0b;padding:12px 16px;border-radius:4px;margin-bottom:16px">
            <p style="margin:0;font-size:12px;font-weight:bold;color:#92400e">Key Risk Factors</p>
            <ul style="margin:6px 0 0;padding-left:16px;font-size:12px">{facts_html}</ul>
          </div>
          <div style="background:#eff6ff;border-left:4px solid #3b82f6;padding:12px 16px;border-radius:4px;margin-bottom:20px">
            <p style="margin:0;font-size:12px;font-weight:bold;color:#1e40af">Recommendation</p>
            <p style="margin:6px 0 0;font-size:12px;color:#1e40af">{rec}</p>
          </div>
        </div>
        <div style="background:#f3f4f6;padding:14px 28px;font-size:11px;color:#9ca3af">
          Sent automatically by LogiSense · {datetime.now().strftime("%Y-%m-%d %H:%M")} · Alert triggered when delay probability exceeds 70%
        </div>
      </div>
    </body></html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🚨 [{risk}] Shipment {sid} — {prob:.1%} Delay Risk"
    msg["From"]    = GMAIL_USER
    msg["To"]      = to_email
    msg.attach(MIMEText(f"Shipment {sid} has {prob:.1%} delay probability. Risk: {risk}. Estimated delay: +{est} days.", "plain"))
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_PASSWORD)
        server.sendmail(GMAIL_USER, to_email, msg.as_string())
    print(f"  ✅ Alert email sent → {to_email} for {sid}")


def send_batch_alert_email(to_email: str, user_name: str, high_risk: list, total: int):
    """Send batch scan digest email showing all high-risk shipments found."""
    if not GMAIL_USER or not GMAIL_PASSWORD:
        print("  Email not configured — skipping batch email")
        return

    rows = ""
    for p in sorted(high_risk, key=lambda x: x["delay_probability"], reverse=True):
        color = "#ef4444" if p["risk_level"]=="HIGH" else "#f59e0b"
        rows += f"""<tr>
          <td style="padding:8px 10px;border-bottom:1px solid #e5e7eb;font-size:12px;font-family:monospace">{p.get('shipment_id','—')}</td>
          <td style="padding:8px 10px;border-bottom:1px solid #e5e7eb;font-size:14px;font-weight:bold;color:{color}">{p['delay_probability']:.1%}</td>
          <td style="padding:8px 10px;border-bottom:1px solid #e5e7eb"><span style="background:{color};color:white;padding:2px 7px;border-radius:3px;font-size:10px">{p['risk_level']}</span></td>
          <td style="padding:8px 10px;border-bottom:1px solid #e5e7eb;font-size:11px;color:#6b7280">+{p.get('estimated_delay_days',0)}d</td>
        </tr>"""

    html = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:620px;margin:0 auto">
      <div style="background:#1a2235;padding:22px 28px;border-radius:12px 12px 0 0">
        <h1 style="color:white;margin:0;font-size:18px">📊 Batch Scan Alert Digest</h1>
        <p style="color:#94a3b8;margin:4px 0 0;font-size:12px">LogiSense · {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
      </div>
      <div style="background:white;padding:24px 28px;border-radius:0 0 12px 12px">
        <p style="font-size:14px;color:#374151;margin-bottom:18px">Hi <b>{user_name}</b>, your batch scan of <b>{total}</b> shipments found <b style="color:#ef4444">{len(high_risk)}</b> high-risk delays.</p>
        <div style="display:flex;gap:12px;margin-bottom:20px">
          <div style="background:#fef2f2;padding:14px 20px;border-radius:8px;flex:1;text-align:center">
            <div style="font-size:26px;font-weight:bold;color:#ef4444">{len(high_risk)}</div>
            <div style="font-size:11px;color:#6b7280">High Risk</div>
          </div>
          <div style="background:#f0fdf4;padding:14px 20px;border-radius:8px;flex:1;text-align:center">
            <div style="font-size:26px;font-weight:bold;color:#10b981">{total-len(high_risk)}</div>
            <div style="font-size:11px;color:#6b7280">On Track</div>
          </div>
          <div style="background:#eff6ff;padding:14px 20px;border-radius:8px;flex:1;text-align:center">
            <div style="font-size:26px;font-weight:bold;color:#3b82f6">{total}</div>
            <div style="font-size:11px;color:#6b7280">Total Scanned</div>
          </div>
        </div>
        <table style="width:100%;border-collapse:collapse">
          <thead><tr style="background:#f9fafb">
            <th style="padding:9px 10px;text-align:left;font-size:10px;color:#6b7280;text-transform:uppercase">Shipment</th>
            <th style="padding:9px 10px;text-align:left;font-size:10px;color:#6b7280;text-transform:uppercase">Probability</th>
            <th style="padding:9px 10px;text-align:left;font-size:10px;color:#6b7280;text-transform:uppercase">Risk</th>
            <th style="padding:9px 10px;text-align:left;font-size:10px;color:#6b7280;text-transform:uppercase">Est. Delay</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
        <p style="font-size:11px;color:#9ca3af;margin-top:16px">Showing top {len(high_risk)} high-risk shipments from your scan.</p>
      </div>
    </body></html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"📊 LogiSense Batch Alert — {len(high_risk)} high-risk shipments found in {total} scanned"
    msg["From"]    = GMAIL_USER
    msg["To"]      = to_email
    msg.attach(MIMEText(f"Batch scan found {len(high_risk)} high-risk shipments out of {total} scanned.", "plain"))
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_PASSWORD)
        server.sendmail(GMAIL_USER, to_email, msg.as_string())
    print(f"  ✅ Batch digest email sent → {to_email} ({len(high_risk)} high-risk)")


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "model_bundle.pkl")

try:
    bundle        = joblib.load(MODEL_PATH)
    model         = bundle["model"]
    preprocessor  = bundle["preprocessor"]
    feature_names = bundle["feature_names"]
    threshold     = bundle["threshold"]
    trained_at    = bundle["trained_at"]
    metrics       = bundle["metrics"]
    print(f"✅ Model loaded — AUC: {metrics.get('test_roc_auc', 'N/A')}")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model = None

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="LogiSense — Delivery Delay Forecasting API",
    description="""
ML-powered delivery delay predictions with role-based authentication.

## Authentication
1. POST `/auth/login` with username + password
2. Copy the `access_token` from response
3. Click **Authorize** button above and paste: `Bearer <token>`
4. All protected endpoints now work

## Default Accounts
| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | Admin |
| manager | manager123 | Manager |
| viewer | viewer123 | Viewer |
    """,
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize users database on startup
@app.on_event("startup")
async def startup():
    init_users_db()
    print("✅ Users database initialized")

# Include auth routes
app.include_router(auth_router)

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class ShipmentInput(BaseModel):
    shipment_id:          Optional[str]  = None
    order_date:           Optional[str]  = None
    carrier:              str
    transport_mode:       str
    cargo_type:           str
    customs_complexity:   str
    origin:               str
    destination:          str
    weight_kg:            float
    distance_km:          float
    carrier_otr_30d:      float
    port_congestion:      float
    weather_severity:     float
    planned_lead_days:    int
    is_peak_season:       Optional[int]  = None

class PredictionResult(BaseModel):
    shipment_id:           Optional[str]
    delay_probability:     float
    delay_probability_pct: str
    prediction:            str
    risk_level:            str
    estimated_delay_days:  float
    confidence:            str
    key_risk_factors:      List[str]
    recommendation:        str
    predicted_at:          str
    predicted_by:          Optional[str] = None  # username

class BatchInput(BaseModel):
    shipments: List[ShipmentInput]

class BatchResult(BaseModel):
    total:       int
    delayed:     int
    on_time:     int
    high_risk:   int
    predictions: List[PredictionResult]

# ─────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────

def build_features(inp: ShipmentInput) -> pd.DataFrame:
    try:    order_dt = pd.to_datetime(inp.order_date) if inp.order_date else datetime.now()
    except: order_dt = datetime.now()

    is_peak = inp.is_peak_season if inp.is_peak_season is not None else int(order_dt.month in [10,11,12])
    dist    = inp.distance_km
    wt      = inp.weight_kg

    dist_bucket   = "local" if dist<1000 else "regional" if dist<3000 else "continental" if dist<7000 else "intercontinental" if dist<12000 else "ultra_long"
    weight_bucket = "tiny"  if wt<100   else "small"    if wt<500   else "medium"       if wt<2000  else "large"            if wt<10000 else "heavy"

    row = {
        "carrier": inp.carrier, "transport_mode": inp.transport_mode,
        "cargo_type": inp.cargo_type, "customs_complexity": inp.customs_complexity,
        "origin": inp.origin, "destination": inp.destination,
        "distance_bucket": dist_bucket, "weight_bucket": weight_bucket,
        "weight_kg": wt, "distance_km": dist,
        "log_weight": np.log1p(wt), "log_distance": np.log1p(dist),
        "carrier_otr_30d": inp.carrier_otr_30d, "port_congestion": inp.port_congestion,
        "weather_severity": inp.weather_severity, "planned_lead_days": inp.planned_lead_days,
        "lead_time_buffer": inp.planned_lead_days - (dist/400),
        "composite_risk_score": inp.weather_severity*0.35 + inp.port_congestion*0.30 + (1-inp.carrier_otr_30d)*0.20 + is_peak*0.15,
        "weather_x_congestion": inp.weather_severity * inp.port_congestion,
        "route_historical_delay_rate": 0.30,
        "carrier_historical_delay_rate": 1 - inp.carrier_otr_30d,
        "order_dayofweek": order_dt.weekday(), "order_month": order_dt.month,
        "order_quarter": (order_dt.month-1)//3+1, "order_weekofyear": order_dt.isocalendar()[1],
        "is_peak_season": is_peak, "is_monday_order": int(order_dt.weekday()==0),
        "is_friday_order": int(order_dt.weekday()==4), "tight_lead": int(inp.planned_lead_days<7),
        "ocean_congestion_flag": int(inp.transport_mode=="ocean" and inp.port_congestion>0.5),
        "hazmat_customs_flag": int(inp.cargo_type=="hazmat" and inp.customs_complexity!="standard"),
        "peak_ocean_flag": int(is_peak==1 and inp.transport_mode=="ocean"),
    }
    return pd.DataFrame([row])

def build_result(inp: ShipmentInput, prob: float, username: str = None) -> PredictionResult:
    if prob>=0.70:   risk,pred="HIGH","DELAYED"
    elif prob>=0.50: risk,pred="MEDIUM","DELAYED"
    elif prob>=0.35: risk,pred="MEDIUM","ON_TIME"
    else:            risk,pred="LOW","ON_TIME"

    est  = round(prob*5.5,1) if pred=="DELAYED" else 0.0
    d    = abs(prob-threshold)
    conf = "High" if d>0.30 else "Medium" if d>0.15 else "Low"

    factors=[]
    if inp.weather_severity>0.6:        factors.append(f"High weather severity ({inp.weather_severity:.0%})")
    if inp.port_congestion>0.6:         factors.append(f"High port congestion ({inp.port_congestion:.0%})")
    if inp.carrier_otr_30d<0.75:        factors.append(f"Low carrier on-time rate ({inp.carrier_otr_30d:.0%})")
    if inp.customs_complexity!="standard": factors.append(f"Customs: {inp.customs_complexity}")
    if inp.transport_mode=="ocean":     factors.append("Ocean freight variability")
    if inp.planned_lead_days<7:         factors.append("Tight lead time")
    if not factors:                     factors=["No major risk factors identified"]

    rec = ("Escalate immediately. Consider expedited routing." if risk=="HIGH"
           else "Monitor closely. Prepare contingency routing." if risk=="MEDIUM"
           else "No action needed.")

    return PredictionResult(
        shipment_id=inp.shipment_id, delay_probability=round(prob,4),
        delay_probability_pct=f"{prob:.1%}", prediction=pred, risk_level=risk,
        estimated_delay_days=est, confidence=conf, key_risk_factors=factors,
        recommendation=rec, predicted_at=datetime.now().isoformat(),
        predicted_by=username,
    )

# ─────────────────────────────────────────────
# PUBLIC ENDPOINTS (no auth required)
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok" if model else "model_not_loaded", "version": "2.0.0"}

# ─────────────────────────────────────────────
# PROTECTED ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/model/info", tags=["System"])
def model_info(current_user: dict = Depends(get_current_user)):
    """Model metadata — requires login."""
    if not model:
        raise HTTPException(503, "Model not loaded.")
    return {
        "model_name":  "logistics-delay-xgboost",
        "trained_at":  trained_at,
        "metrics":     metrics,
        "threshold":   threshold,
        "n_features":  len(feature_names),
        "requested_by": current_user["username"],
        "requester_role": current_user["role"],
    }

@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(
    shipment: ShipmentInput,
    current_user: dict = Depends(get_current_user)
):
    """Predict delay — requires login. Auto-sends email if high risk."""
    if not model:
        raise HTTPException(503, "Model not loaded.")
    try:
        X    = preprocessor.transform(build_features(shipment)[feature_names])
        prob = float(model.predict_proba(X)[0, 1])
        result = build_result(shipment, prob, username=current_user["username"])

        # Auto-send email if high risk
        if prob >= 0.70 and current_user.get("email"):
            try:
                send_shipment_alert_email(
                    to_email  = current_user["email"],
                    user_name = current_user.get("full_name") or current_user["username"],
                    prediction= result.dict(),
                )
            except Exception as e:
                print(f"Email send failed: {e}")

        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict/batch", response_model=BatchResult, tags=["Prediction"])
def predict_batch(
    batch: BatchInput,
    current_user: dict = Depends(require_role("admin", "manager"))
):
    """Batch predict — requires Manager or Admin role. Emails high-risk alerts."""
    if not model:
        raise HTTPException(503, "Model not loaded.")
    if len(batch.shipments) > 500:
        raise HTTPException(400, "Max 500 shipments per request.")
    results = []
    for s in batch.shipments:
        X    = preprocessor.transform(build_features(s)[feature_names])
        prob = float(model.predict_proba(X)[0, 1])
        results.append(build_result(s, prob, username=current_user["username"]))
    delayed   = [r for r in results if r.prediction=="DELAYED"]
    high_risk = [r for r in results if r.risk_level=="HIGH"]

    # Send digest email for high-risk shipments
    if high_risk and current_user.get("email"):
        try:
            send_batch_alert_email(
                to_email  = current_user["email"],
                user_name = current_user.get("full_name") or current_user["username"],
                high_risk = [r.dict() for r in high_risk[:10]],
                total     = len(results),
            )
        except Exception as e:
            print(f"Batch email failed: {e}")

    return BatchResult(
        total=len(results), delayed=len(delayed),
        on_time=len(results)-len(delayed),
        high_risk=len(high_risk),
        predictions=results,
    )

@app.get("/admin/stats", tags=["Admin"])
def admin_stats(current_user: dict = Depends(require_role("admin"))):
    """Admin-only statistics endpoint."""
    return {
        "model_metrics":  metrics,
        "total_features": len(feature_names),
        "trained_at":     trained_at,
        "api_version":    "2.0.0",
        "accessed_by":    current_user["username"],
    }


# ── AI Chatbot endpoint ────────────────────────
import httpx

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE    = "https://api.groq.com/openai/v1"

CHATBOT_SYSTEM = """You are LogiSense AI Assistant — an expert logistics intelligence assistant built into the LogiSense ML-based delivery delay forecasting platform.

You help logistics operations teams with:
1. Understanding shipment delay risks and probabilities
2. Interpreting ML model predictions and risk factors
3. Recommending actions for high-risk shipments
4. Explaining analytics trends (delay rates by carrier, transport mode, route)
5. Answering questions about the LogiSense platform features

Key facts about LogiSense:
- ML model: XGBoost with 93.4% accuracy, 0.971 AUC-ROC
- Trained on 180,519 real shipments from DataCo dataset
- 32 features: weather severity, port congestion, carrier OTR, customs complexity, transport mode, cargo type, lead time, distance, weight, peak season
- Risk levels: HIGH (>70%), MEDIUM (35-70%), LOW (<35%)
- Carriers: Maersk, MSC, CMA CGM, DHL Express, FedEx, UPS, DB Schenker
- Transport modes: ocean, air, road, rail, multimodal
- Highest delay carrier: MSC (34%) | Lowest: FedEx (8%)
- Highest delay mode: Ocean (31%) | Lowest: Air (7%)
- Peak delay months: October-December (Q4 surge)
- Riskiest route: Shanghai → Chicago (91% avg probability)

When risk is HIGH: recommend expediting, rerouting, or customer notification.
When risk is MEDIUM: recommend monitoring and preparing contingency plans.
When risk is LOW: confirm on track, routine monitoring.

Be concise, professional, and actionable. Use bullet points for recommendations.
Always respond in the same language the user writes in."""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.post("/chat", tags=["AI Assistant"])
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """AI chatbot endpoint — powered by Groq (free)."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(
                f"{GROQ_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       "llama3-8b-8192",
                    "max_tokens":  1000,
                    "temperature": 0.7,
                    "messages": [
                        {"role": "system", "content": CHATBOT_SYSTEM},
                        *[{"role": m.role, "content": m.content} for m in request.messages]
                    ],
                }
            )
            data = res.json()
            reply = data.get("choices", [{}])[0].get("message", {}).get("content", "Sorry, I could not get a response.")
            return {"reply": reply}
    except Exception as e:
        raise HTTPException(500, f"AI error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)

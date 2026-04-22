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
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from auth import (
    router as auth_router,
    get_current_user,
    require_role,
    init_users_db,
)

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
    current_user: dict = Depends(get_current_user)  # all roles can predict
):
    """Predict delay — requires login (any role)."""
    if not model:
        raise HTTPException(503, "Model not loaded.")
    try:
        X    = preprocessor.transform(build_features(shipment)[feature_names])
        prob = float(model.predict_proba(X)[0, 1])
        return build_result(shipment, prob, username=current_user["username"])
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict/batch", response_model=BatchResult, tags=["Prediction"])
def predict_batch(
    batch: BatchInput,
    current_user: dict = Depends(require_role("admin", "manager"))  # viewer cannot batch predict
):
    """Batch predict — requires Manager or Admin role."""
    if not model:
        raise HTTPException(503, "Model not loaded.")
    if len(batch.shipments) > 500:
        raise HTTPException(400, "Max 500 shipments per request.")
    results = []
    for s in batch.shipments:
        X    = preprocessor.transform(build_features(s)[feature_names])
        prob = float(model.predict_proba(X)[0, 1])
        results.append(build_result(s, prob, username=current_user["username"]))
    delayed = [r for r in results if r.prediction=="DELAYED"]
    return BatchResult(
        total=len(results), delayed=len(delayed),
        on_time=len(results)-len(delayed),
        high_risk=sum(1 for r in results if r.risk_level=="HIGH"),
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)

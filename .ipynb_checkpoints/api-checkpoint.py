
import joblib, numpy as np, pandas as pd
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

bundle       = joblib.load("model_bundle.pkl")
model        = bundle["model"]
preprocessor = bundle["preprocessor"]
feature_names= bundle["feature_names"]
threshold    = bundle["threshold"]
trained_at   = bundle["trained_at"]
metrics      = bundle["metrics"]

app = FastAPI(title="LogiSense Delay Forecasting API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ShipmentInput(BaseModel):
    shipment_id:        Optional[str]  = None
    order_date:         Optional[str]  = None
    carrier:            str
    transport_mode:     str
    cargo_type:         str
    customs_complexity: str
    origin:             str
    destination:        str
    weight_kg:          float
    distance_km:        float
    carrier_otr_30d:    float
    port_congestion:    float
    weather_severity:   float
    planned_lead_days:  int
    is_peak_season:     Optional[int] = None

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

class BatchInput(BaseModel):
    shipments: List[ShipmentInput]

class BatchResult(BaseModel):
    total: int; delayed: int; on_time: int; high_risk: int
    predictions: List[PredictionResult]

def build_features(inp):
    try:    order_dt = pd.to_datetime(inp.order_date) if inp.order_date else datetime.now()
    except: order_dt = datetime.now()
    is_peak = inp.is_peak_season if inp.is_peak_season is not None else int(order_dt.month in [10,11,12])
    dist = inp.distance_km
    dist_bucket = "local" if dist<1000 else "regional" if dist<3000 else "continental" if dist<7000 else "intercontinental" if dist<12000 else "ultra_long"
    wt = inp.weight_kg
    weight_bucket = "tiny" if wt<100 else "small" if wt<500 else "medium" if wt<2000 else "large" if wt<10000 else "heavy"
    row = {
        "carrier": inp.carrier, "transport_mode": inp.transport_mode,
        "cargo_type": inp.cargo_type, "customs_complexity": inp.customs_complexity,
        "origin": inp.origin, "destination": inp.destination,
        "distance_bucket": dist_bucket, "weight_bucket": weight_bucket,
        "weight_kg": inp.weight_kg, "distance_km": dist,
        "log_weight": np.log1p(inp.weight_kg), "log_distance": np.log1p(dist),
        "carrier_otr_30d": inp.carrier_otr_30d, "port_congestion": inp.port_congestion,
        "weather_severity": inp.weather_severity, "planned_lead_days": inp.planned_lead_days,
        "lead_time_buffer": inp.planned_lead_days - (dist/400),
        "composite_risk_score": inp.weather_severity*0.35 + inp.port_congestion*0.30 + (1-inp.carrier_otr_30d)*0.20 + is_peak*0.15,
        "weather_x_congestion": inp.weather_severity * inp.port_congestion,
        "route_historical_delay_rate": 0.30, "carrier_historical_delay_rate": 1-inp.carrier_otr_30d,
        "order_dayofweek": order_dt.weekday(), "order_month": order_dt.month,
        "order_quarter": (order_dt.month-1)//3+1, "order_weekofyear": order_dt.isocalendar()[1],
        "is_peak_season": is_peak, "is_monday_order": int(order_dt.weekday()==0),
        "is_friday_order": int(order_dt.weekday()==4), "tight_lead": int(inp.planned_lead_days<7),
        "ocean_congestion_flag": int(inp.transport_mode=="ocean" and inp.port_congestion>0.5),
        "hazmat_customs_flag": int(inp.cargo_type=="hazmat" and inp.customs_complexity!="standard"),
        "peak_ocean_flag": int(is_peak==1 and inp.transport_mode=="ocean"),
    }
    return pd.DataFrame([row])

def build_result(inp, prob):
    if prob>=0.70:   risk,pred="HIGH","DELAYED"
    elif prob>=0.50: risk,pred="MEDIUM","DELAYED"
    elif prob>=0.35: risk,pred="MEDIUM","ON_TIME"
    else:            risk,pred="LOW","ON_TIME"
    est = round(prob*5.5,1) if pred=="DELAYED" else 0.0
    d = abs(prob-threshold)
    conf = "High" if d>0.30 else "Medium" if d>0.15 else "Low"
    factors=[]
    if inp.weather_severity>0.6:   factors.append(f"High weather severity ({inp.weather_severity:.0%})")
    if inp.port_congestion>0.6:    factors.append(f"High port congestion ({inp.port_congestion:.0%})")
    if inp.carrier_otr_30d<0.75:   factors.append(f"Low carrier on-time rate ({inp.carrier_otr_30d:.0%})")
    if inp.customs_complexity!="standard": factors.append(f"Customs: {inp.customs_complexity}")
    if inp.transport_mode=="ocean": factors.append("Ocean freight variability")
    if inp.planned_lead_days<7:    factors.append("Tight lead time")
    if not factors: factors=["No major risk factors"]
    rec = ("Escalate immediately. Consider expedited routing." if risk=="HIGH"
           else "Monitor closely. Prepare contingency." if risk=="MEDIUM"
           else "No action needed.")
    return PredictionResult(shipment_id=inp.shipment_id, delay_probability=round(prob,4),
        delay_probability_pct=f"{prob:.1%}", prediction=pred, risk_level=risk,
        estimated_delay_days=est, confidence=conf, key_risk_factors=factors,
        recommendation=rec, predicted_at=datetime.now().isoformat())

@app.get("/health")
def health(): return {"status":"ok","trained_at":trained_at,"metrics":metrics}

@app.get("/model/info")
def info(): return {"model_name":"logistics-delay-xgboost","trained_at":trained_at,"metrics":metrics,"threshold":threshold,"n_features":len(feature_names)}

@app.post("/predict", response_model=PredictionResult)
def predict(s: ShipmentInput):
    try:
        X = preprocessor.transform(build_features(s)[feature_names])
        return build_result(s, float(model.predict_proba(X)[0,1]))
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/predict/batch", response_model=BatchResult)
def predict_batch(b: BatchInput):
    if len(b.shipments)>500: raise HTTPException(400,"Max 500 shipments")
    results=[]
    for s in b.shipments:
        X = preprocessor.transform(build_features(s)[feature_names])
        results.append(build_result(s, float(model.predict_proba(X)[0,1])))
    delayed=[r for r in results if r.prediction=="DELAYED"]
    return BatchResult(total=len(results),delayed=len(delayed),on_time=len(results)-len(delayed),
        high_risk=sum(1 for r in results if r.risk_level=="HIGH"),predictions=results)

if __name__=="__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

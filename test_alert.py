from alerts import init_db, check_and_alert

init_db()

shipments = [
    {
        "shipment_id": "SH-4821",
        "order_date": "2024-11-15",
        "carrier": "MSC",
        "transport_mode": "ocean",
        "cargo_type": "hazmat",
        "customs_complexity": "high_scrutiny",
        "origin": "Shanghai",
        "destination": "Chicago",
        "weight_kg": 4800,
        "distance_km": 12500,
        "carrier_otr_30d": 0.68,
        "port_congestion": 0.78,
        "weather_severity": 0.72,
        "planned_lead_days": 5,
        "is_peak_season": 1
    }
]

check_and_alert(shipments)
print("Done! Check your Gmail inbox.")
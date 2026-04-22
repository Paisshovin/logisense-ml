@echo off
echo Starting LogiSense ML System...
call conda activate py310
cd /d C:\Users\Shovin\logistics_ml
start "" "C:\Users\Shovin\logistics_ml\dashboard.html"
start /b uvicorn api:app --port 8000
echo API running at http://localhost:8000
pause
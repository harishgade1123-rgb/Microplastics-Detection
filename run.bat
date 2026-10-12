@echo off
title MicroDetect AI - Starting Server
color 0A
echo.
echo  =======================================================
echo   MicroDetect AI - Microplastic Detection System
echo   Starting Flask backend on http://127.0.0.1:5000 ...
echo  =======================================================
echo.

cd /d "%~dp0"

echo  [INFO] Launching Python server...
start cmd /k "python server.py"

echo  [INFO] Initializing YOLO model and API server...
echo  [INFO] Waiting for server to become ready...

powershell -Command "for ($i=0; $i -lt 30; $i++) { try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:5000/api/health' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { break } } catch {}; Start-Sleep -Seconds 1 }"

echo.
echo  [INFO] Server ready! Opening application in browser...
start chrome "http://127.0.0.1:5000"



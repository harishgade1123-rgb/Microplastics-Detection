@echo off
title Push Microplastics Detector to GitHub
set "PATH=%LOCALAPPDATA%\Programs\Git\cmd;%LOCALAPPDATA%\GitHubDesktop\app-3.6.6\resources\app\git\cmd;%PATH%"

echo ======================================================
echo Pushing Microplastics Detector to GitHub (origin main)...
echo Account: harishgade1123-rgb (harishgade1123@gmail.com)
echo Target:  https://github.com/harishgade1123-rgb/microplastics-detector
echo ======================================================
echo.

git push -u origin main

echo.
if %errorlevel% equ 0 (
    echo ======================================================
    echo Push SUCCESSFUL!
    echo View your code at: https://github.com/harishgade1123-rgb/microplastics-detector
    echo ======================================================
) else (
    echo ======================================================
    echo Push failed. Please check the error above.
    echo If repository was not found, make sure you created
    echo "microplastics-detector" at https://github.com/new
    echo under account "harishgade1123-rgb".
    echo ======================================================
)

pause

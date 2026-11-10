@echo off
title SMART ML PARAMETER TUNER - AutoML Mini Lab
echo =======================================================
echo   SMART ML PARAMETER TUNER - AutoML Mini Lab
echo =======================================================
echo.
echo Launching Streamlit web application on http://localhost:8501 ...
echo.

cd /d "%~dp0"
python -m streamlit run app.py

pause

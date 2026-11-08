@echo off
title Harvestify - Smart Agriculture Website
cd /d "%~dp0"
echo ========================================================
echo  Starting Harvestify AI Agriculture Web Application...
echo ========================================================
echo.
python -m streamlit run streamlit_app.py
pause

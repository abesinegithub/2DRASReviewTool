@echo off
setlocal
cd /d "%~dp0"
py -m streamlit run app.py
if errorlevel 1 (
  echo.
  echo If Streamlit is not installed, run: py -m pip install -r requirements.txt
  pause
)

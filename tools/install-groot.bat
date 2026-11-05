@echo off
title Download Groot 19B-A3B Q4_K_M (12.5 GB)
cd /d "G:\PolyStrata"
echo ========================================================
echo   Downloading Groot 19B-A3B (Bartowski Q4_K_M)
echo   Target: G:\Strata-3060\models\groot
echo ========================================================
python setup.py --family groot --model Q4_K_M --context 32768 --vision no --yes --no-start
echo.
echo Setup finished! You can now launch Groot via:
echo G:\Strata-3060\am2strata\launchers\start-groot.bat
echo.
pause

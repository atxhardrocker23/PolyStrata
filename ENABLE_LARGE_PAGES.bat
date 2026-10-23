@echo off
title Configure Windows Large Pages (SeLockMemoryPrivilege)
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting Administrator privileges to grant SeLockMemoryPrivilege...
    powershell -Command "Start-Process cmd -ArgumentList '/c', '\"%~f0\"' -Verb runAs"
    exit /b
)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows\enable_large_pages.ps1"
echo.
pause

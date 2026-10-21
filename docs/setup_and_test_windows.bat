@echo off
setlocal
cd /d "%~dp0"

echo Installing the lightweight inference dependency...
py -3.11 -m pip install -r skill\hidden-bites-macau-food-agent\requirements.txt
if errorlevel 1 goto :failed

echo.
echo Running Hidden Bites Skill self-check...
py -3.11 skill\hidden-bites-macau-food-agent\scripts\self_check.py
if errorlevel 1 goto :failed

echo.
echo Running parsing and UTF-8 regression tests...
py -3.11 skill\hidden-bites-macau-food-agent\scripts\regression_test.py
if errorlevel 1 goto :failed

echo.
echo PASS. Upload hidden_bites_qwenpaw_skill.zip in QwenPaw:
echo Workspace ^> Skills ^> Add Skill ^> Upload ZIP
pause
exit /b 0

:failed
echo.
echo Setup or self-check failed. Read README_DEPLOY.md for troubleshooting.
pause
exit /b 1

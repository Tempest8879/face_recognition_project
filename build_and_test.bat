@echo off
cd /d "%~dp0"
call build.bat
if errorlevel 1 exit /b 1
echo.
echo Running tests...
python python\test_setup.py

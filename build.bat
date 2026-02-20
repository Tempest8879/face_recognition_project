@echo off
REM ============================================================================
REM Build script for Face Processor C++ module (uses MSVC)
REM ============================================================================

echo [1/3] Setting up Visual Studio environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul 2>&1
if errorlevel 1 (
    echo ERROR: Could not find Visual Studio Build Tools.
    exit /b 1
)

echo [2/3] Getting Python paths...
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_path('include'))"') do set PYTHON_INCLUDE=%%i
for /f "tokens=*" %%i in ('python -c "import pybind11; print(pybind11.get_include())"') do set PYBIND11_INCLUDE=%%i
for /f "tokens=*" %%i in ('python -c "import sysconfig, os; print(os.path.join(os.path.dirname(sysconfig.get_path('stdlib')), 'libs'))"') do set PYTHON_LIBS=%%i
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"') do set EXT_SUFFIX=%%i

echo    Python Include: %PYTHON_INCLUDE%
echo    Pybind11 Include: %PYBIND11_INCLUDE%
echo    Python Libs: %PYTHON_LIBS%
echo    Extension Suffix: %EXT_SUFFIX%

echo [3/3] Compiling face_processor_cpp%EXT_SUFFIX% ...
cl /O2 /EHsc /std:c++17 /LD ^
    /I"%PYTHON_INCLUDE%" ^
    /I"%PYBIND11_INCLUDE%" ^
    cpp\bindings.cpp ^
    /Fe:python\face_processor_cpp%EXT_SUFFIX% ^
    /link /LIBPATH:"%PYTHON_LIBS%"

if errorlevel 1 (
    echo.
    echo ERROR: Compilation failed!
    exit /b 1
)

REM Clean up intermediate files
del *.obj *.exp *.lib 2>nul

echo.
echo ============================================
echo   BUILD SUCCESSFUL!
echo   Output: python\face_processor_cpp%EXT_SUFFIX%
echo ============================================

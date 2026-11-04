@echo off
setlocal
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Project Python environment missing. See Windows setup in README.md.
    pause
    exit /b 1
)
if not "%~1"=="" goto arguments
"%PYTHON_EXE%" "%~dp0arm4_twin.py" --demo
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" pause
exit /b %RESULT%

:arguments
"%PYTHON_EXE%" "%~dp0arm4_twin.py" %*
exit /b %ERRORLEVEL%

@echo off
setlocal
set "PYTHON_EXE=%LOCALAPPDATA%\arm4-mj316\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
if not "%~1"=="" goto arguments
"%PYTHON_EXE%" "%~dp0arm4_twin.py" --demo
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" pause
exit /b %RESULT%

:arguments
"%PYTHON_EXE%" "%~dp0arm4_twin.py" %*
exit /b %ERRORLEVEL%

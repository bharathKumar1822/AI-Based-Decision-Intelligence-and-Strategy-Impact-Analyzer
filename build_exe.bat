@echo off
echo ================================================
echo   Building Decision Intelligence Analyzer EXE
echo ================================================

REM Activate venv
call .venv\Scripts\activate.bat

REM Install PyInstaller if not present
pip install pyinstaller --quiet

REM Clean previous build
if exist dist\DecisionIntelligenceAnalyzer rmdir /s /q dist\DecisionIntelligenceAnalyzer
if exist build rmdir /s /q build

REM Build using the spec file
echo.
echo [1/2] Running PyInstaller (this may take 3-5 minutes)...
pyinstaller decision_intelligence.spec --noconfirm

echo.
echo [2/2] Done!
echo.
echo ================================================
echo   EXE LOCATION:
echo   dist\DecisionIntelligenceAnalyzer\DecisionIntelligenceAnalyzer.exe
echo.
echo   Share the entire folder:
echo   dist\DecisionIntelligenceAnalyzer\
echo ================================================
pause

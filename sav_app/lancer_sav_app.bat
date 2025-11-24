@echo off
REM Script de lancement de l'application SAV Tweets
title SAV Tweets - Lancement en cours...

echo ========================================
echo   SAV TWEETS - Support Client IA
echo ========================================
echo.
echo Demarrage de l'application...
echo.

REM Se placer dans le dossier de l'application
cd /d "%~dp0"

REM Activer l'environnement virtuel si disponible
if exist "venv\Scripts\activate.bat" (
    echo Activation de l'environnement virtuel...
    call "venv\Scripts\activate.bat"
) else if exist "..\..\.venv\Scripts\activate.bat" (
    echo Activation de l'environnement virtuel parent...
    call "..\..\.venv\Scripts\activate.bat"
) else (
    echo Environnement virtuel non trouve, utilisation de Python systeme...
)

echo.
echo Lancement de Streamlit...
echo L'application va s'ouvrir dans votre navigateur...
echo.
echo Pour arreter l'application, fermez cette fenetre ou appuyez sur Ctrl+C
echo ========================================
echo.

REM Lancer l'application Streamlit
python -m streamlit run app.py

REM Pause si erreur
if errorlevel 1 (
    echo.
    echo ERREUR: L'application n'a pas pu demarrer.
    echo Verifiez que Python et Streamlit sont installes.
    echo.
    pause
)

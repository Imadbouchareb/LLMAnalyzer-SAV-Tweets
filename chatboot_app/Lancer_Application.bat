@echo off
REM Raccourci pour lancer l'Assistant Free Mobile
cd /d "c:\Users\hallo\OneDrive\Bureau\IA Free Mobile\chatboot_app"
echo.
echo ================================================
echo  Lancement de l'Assistant Free Mobile...
echo ================================================
echo.
echo Demarrage en cours...
echo.

REM Activer l'environnement virtuel et lancer Streamlit
call venv\Scripts\activate.bat
streamlit run app.py

echo.
echo Application lancee !
echo L'application s'ouvrira dans votre navigateur
echo.
echo Pour arreter l'application, fermez cette fenetre
echo.
pause

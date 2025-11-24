# Script PowerShell pour lancer l'application SAV Tweets
# Pour exécuter: Clic droit > "Exécuter avec PowerShell"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SAV TWEETS - Support Client IA" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Se placer dans le dossier du script
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "Démarrage de l'application..." -ForegroundColor Yellow
Write-Host ""

# Chercher et activer l'environnement virtuel
$venvPaths = @(
    "venv\Scripts\Activate.ps1",
    ".venv\Scripts\Activate.ps1",
    "..\..\..venv\Scripts\Activate.ps1"
)

$venvActivated = $false
foreach ($venvPath in $venvPaths) {
    if (Test-Path $venvPath) {
        Write-Host "Activation de l'environnement virtuel..." -ForegroundColor Green
        & $venvPath
        $venvActivated = $true
        break
    }
}

if (-not $venvActivated) {
    Write-Host "Environnement virtuel non trouvé, utilisation de Python système..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Lancement de Streamlit..." -ForegroundColor Green
Write-Host "L'application va s'ouvrir dans votre navigateur..." -ForegroundColor Green
Write-Host ""
Write-Host "Pour arrêter l'application, fermez cette fenêtre ou appuyez sur Ctrl+C" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Lancer l'application
try {
    python -m streamlit run app.py
}
catch {
    Write-Host ""
    Write-Host "ERREUR: L'application n'a pas pu démarrer." -ForegroundColor Red
    Write-Host "Vérifiez que Python et Streamlit sont installés." -ForegroundColor Red
    Write-Host ""
    Read-Host "Appuyez sur Entrée pour fermer"
}
